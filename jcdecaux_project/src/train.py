"""Entraînement du modèle CNN multi-tâches.

Contrairement à la version précédente, ce script :
  - relit les artefacts d'encodage sur disque (pas de dépendance à l'ordre
    d'import entre encoding.py / excel_matching.py) ;
  - applique une perte pondérée par classe (inverse-fréquence) par tâche,
    pour compenser le déséquilibre observé sur plusieurs attributs
    (ex. "Personnages", "1er point d'accroche", "Tranche d'âge") ;
  - conserve le sanity check du premier batch et l'arrêt anticipé déjà
    en place dans les versions précédentes.

Prérequis : avoir exécuté `python -m src.encoding` au préalable.
"""
import gc
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import optim
from tqdm import tqdm

from config.config import (
    DEVICE, BACKBONE, EPOCHS, MODEL_PATH, ARTIFACTS_DIR, OUTPUT_DIR,
    ENCODERS_PATH, CLASS_WEIGHTS_PATH, LR, WEIGHT_DECAY, PATIENCE,
    USE_CLASS_WEIGHTS,
)
from src.model import MultiTaskModel
from src.dataset import build_dataloaders

LOGS_DIR = OUTPUT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Machine à mémoire limitée (~8 Go RAM, souvent < 1.5 Go libre observé) :
# limiter les threads CPU de torch réduit les buffers internes par thread.
torch.set_num_threads(2)


def load_encoded_data():
    df_path = OUTPUT_DIR / "df_labeled_with_enc.csv"
    if not df_path.exists():
        raise FileNotFoundError(f"{df_path} introuvable : lancer `python -m src.encoding` d'abord.")
    df = pd.read_csv(df_path, encoding="utf-8-sig")

    with open(ENCODERS_PATH, "rb") as f:
        enc_data = pickle.load(f)
    encoders, col_types, target_cols = enc_data["encoders"], enc_data["col_types"], enc_data["target_cols"]

    class_weights = {}
    if USE_CLASS_WEIGHTS and CLASS_WEIGHTS_PATH.exists():
        with open(CLASS_WEIGHTS_PATH, "rb") as f:
            class_weights = pickle.load(f)

    return df, encoders, col_types, target_cols, class_weights


def build_loss_fns(target_cols, class_weights):
    loss_fns = {}
    for c in target_cols:
        weight = None
        if c in class_weights:
            weight = torch.tensor(class_weights[c], dtype=torch.float32).to(DEVICE)
        loss_fns[c] = torch.nn.CrossEntropyLoss(weight=weight, reduction="mean")
    return loss_fns


def sanity_check_first_batch(model, train_loader, target_cols):
    print("Running sanity check on first batch...")
    imgs, targets, paths = next(iter(train_loader))
    print(" imgs.shape:", imgs.shape, "dtype:", imgs.dtype)
    print(" example paths:", paths[:3])
    for c in target_cols:
        t = targets[c]
        print(f" target '{c[:40]}': shape={t.shape}, min={t.min().item()}, max={t.max().item()}")
    model.eval()
    with torch.no_grad():
        outputs = model(imgs.to(DEVICE))
    for c, out in outputs.items():
        print(f" output '{c[:40]}': shape={out.shape}")
    print("Sanity check done.\n")


def run_epoch(model, loader, target_cols, loss_fns, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_batches = 0.0, 0
    per_task_loss = {c: 0.0 for c in target_cols}

    context = torch.enable_grad() if is_train else torch.no_grad()
    iterator = tqdm(loader, desc="Train" if is_train else "Val", leave=False)
    with context:
        for imgs, targets, _ in iterator:
            imgs = imgs.to(DEVICE)
            for c in target_cols:
                targets[c] = targets[c].long().to(DEVICE)

            if is_train:
                optimizer.zero_grad()
            outputs = model(imgs)

            task_losses = []
            for c in target_cols:
                loss_val = loss_fns[c](outputs[c], targets[c])
                task_losses.append(loss_val)
                per_task_loss[c] += float(loss_val.detach().cpu().item())

            loss = torch.stack(task_losses).mean()
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            n_batches += 1
            iterator.set_postfix({"loss": f"{total_loss / n_batches:.4f}"})

    n_batches = max(1, n_batches)
    avg_loss = total_loss / n_batches
    avg_per_task = {c: per_task_loss[c] / n_batches for c in target_cols}
    return avg_loss, avg_per_task


def main():
    df, encoders, col_types, target_cols, class_weights = load_encoded_data()
    print(f"{len(target_cols)} colonnes cibles, {len(df)} images.")

    train_loader, val_loader, test_loader = build_dataloaders(df, target_cols, col_types)

    model = MultiTaskModel(BACKBONE, target_cols, col_types, encoders).to(DEVICE)
    loss_fns = build_loss_fns(target_cols, class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    sanity_check_first_batch(model, train_loader, target_cols)

    best_val = float("inf")
    no_improve = 0
    history = []

    print("Starting training loop...")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_per_task = run_epoch(model, train_loader, target_cols, loss_fns, optimizer)
        val_loss, val_per_task = run_epoch(model, val_loader, target_cols, loss_fns, optimizer=None)
        scheduler.step(val_loss)
        gc.collect()
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{EPOCHS} — train_loss: {train_loss:.4f} — val_loss: {val_loss:.4f} — time: {elapsed:.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            **{f"train_{c}": train_per_task[c] for c in target_cols},
            **{f"val_{c}": val_per_task[c] for c in target_cols},
        })
        pd.DataFrame(history).to_csv(LOGS_DIR / "training_history.csv", index=False)

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, str(MODEL_PATH))
            print(" Nouveau meilleur modèle sauvegardé:", MODEL_PATH)
        else:
            no_improve += 1
            print(f" Pas d'amélioration depuis {no_improve} epoch(s).")
            if no_improve >= PATIENCE:
                print("Early stopping.")
                break

    print("Entraînement terminé. Meilleure val_loss:", best_val)


if __name__ == "__main__":
    main()
