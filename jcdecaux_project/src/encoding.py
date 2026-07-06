# src/train.py
import os
import time
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import optim

from config.config import (
    DEVICE, BACKBONE, EPOCHS, MODEL_PATH, ARTIFACTS_DIR,
    IMG_SIZE, BATCH_SIZE, WEIGHT_DECAY, LR, PATIENCE
)
from src.model import MultiTaskModel
from src.dataset import build_dataloaders
from src.excel_matching import df_labeled
from src.encoding import encoders, col_types, target_cols

# Output folders
LOGS_DIR = Path("outputs/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = Path(ARTIFACTS_DIR) if isinstance(ARTIFACTS_DIR, (str, Path)) else Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Utility: save history
def save_history(history, path=LOGS_DIR / "training_history.csv"):
    df = pd.DataFrame(history)
    df.to_csv(path, index=False)
    print("Training history saved to", path)

# Build dataloaders from df_labeled (df_labeled comes from src.excel_matching)
print("Building dataloaders...")
train_loader, val_loader, test_loader = build_dataloaders(df_labeled, target_cols, col_types)

# Instantiate model
print("Instantiating model...")
model = MultiTaskModel(BACKBONE, target_cols, col_types, encoders).to(DEVICE)

# Loss functions per task (robuste)
loss_fns = {}
for c in target_cols:
    ttype = col_types.get(c, "categorical")
    if ttype == "categorical":
        loss_fns[c] = torch.nn.CrossEntropyLoss(reduction="mean")
    elif ttype == "binary":
        loss_fns[c] = torch.nn.BCEWithLogitsLoss(reduction="mean")
    else:
        loss_fns[c] = torch.nn.MSELoss(reduction="mean")

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5, verbose=True)

# Debug / sanity check on first batch: shapes, dtypes, ranges
def sanity_check_first_batch():
    print("Running sanity check on first batch...")
    batch = next(iter(train_loader))
    imgs, targets, paths = batch
    print(" imgs.shape:", imgs.shape, "dtype:", imgs.dtype)
    print(" example paths:", paths[:3])
    for c in target_cols:
        t = targets[c]
        print(f" target '{c}': shape={t.shape}, dtype={t.dtype}, min={t.min().item()}, max={t.max().item()}, unique_sample={np.unique(t.cpu().numpy())[:10]}")
    # forward pass shapes
    model.eval()
    with torch.no_grad():
        imgs_dev = imgs.to(DEVICE)
        outputs = model(imgs_dev)
    for c, out in outputs.items():
        print(f" output '{c}': shape={out.shape}, dtype={out.dtype}, min={float(out.min()):.4f}, max={float(out.max()):.4f}")
    print("Sanity check done.\n")

# Run sanity check once
try:
    sanity_check_first_batch()
except Exception as e:
    print("Sanity check failed. Inspect shapes/types above. Error:", e)
    raise

# Training loop with robust loss handling and logging per task
best_val = float("inf")
no_improve = 0
history = []

print("Starting training loop...")
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    train_epoch_loss = 0.0
    train_task_loss_acc = {c: 0.0 for c in target_cols}
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)
    for imgs, targets, _ in pbar:
        imgs = imgs.to(DEVICE)
        # cast and move targets safely
        for c in target_cols:
            if col_types[c] == "categorical":
                targets[c] = targets[c].long().to(DEVICE)
            else:
                targets[c] = targets[c].float().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)

        # compute per-task losses
        task_losses = []
        task_losses_values = {}
        for c in target_cols:
            out = outputs[c]
            ttype = col_types[c]

            # Ensure shapes are compatible
            if ttype == "categorical":
                # out expected shape: (B, C)
                if out.dim() == 1:
                    out = out.unsqueeze(0)
                # targets shape: (B,)
                loss_val = loss_fns[c](out, targets[c])
            elif ttype == "binary":
                # out can be (B,1) or (B,)
                if out.dim() == 2 and out.size(1) == 1:
                    out_flat = out.view(-1)
                else:
                    out_flat = out.view(-1)
                loss_val = loss_fns[c](out_flat, targets[c].view(-1))
            else:  # numeric / regression
                loss_val = loss_fns[c](out.view(-1), targets[c].view(-1))

            task_losses.append(loss_val)
            task_losses_values[c] = float(loss_val.detach().cpu().item())

        # average across tasks to avoid huge sums
        loss = torch.stack(task_losses).mean()
        loss.backward()
        optimizer.step()

        train_epoch_loss += float(loss.detach().cpu().item())
        for c in target_cols:
            train_task_loss_acc[c] += task_losses_values[c]
        n_batches += 1

        pbar.set_postfix({"loss": f"{train_epoch_loss / n_batches:.4f}"})

    avg_train_loss = train_epoch_loss / max(1, n_batches)
    avg_task_train = {c: train_task_loss_acc[c] / max(1, n_batches) for c in target_cols}

    # Validation
    model.eval()
    val_epoch_loss = 0.0
    val_task_loss_acc = {c: 0.0 for c in target_cols}
    n_val_batches = 0
    with torch.no_grad():
        for imgs, targets, _ in val_loader:
            imgs = imgs.to(DEVICE)
            for c in target_cols:
                if col_types[c] == "categorical":
                    targets[c] = targets[c].long().to(DEVICE)
                else:
                    targets[c] = targets[c].float().to(DEVICE)

            outputs = model(imgs)
            task_losses = []
            for c in target_cols:
                out = outputs[c]
                ttype = col_types[c]
                if ttype == "categorical":
                    if out.dim() == 1:
                        out = out.unsqueeze(0)
                    loss_val = loss_fns[c](out, targets[c])
                elif ttype == "binary":
                    if out.dim() == 2 and out.size(1) == 1:
                        out_flat = out.view(-1)
                    else:
                        out_flat = out.view(-1)
                    loss_val = loss_fns[c](out_flat, targets[c].view(-1))
                else:
                    loss_val = loss_fns[c](out.view(-1), targets[c].view(-1))
                task_losses.append(loss_val)
                val_task_loss_acc[c] += float(loss_val.detach().cpu().item())

            loss_val_batch = torch.stack(task_losses).mean()
            val_epoch_loss += float(loss_val_batch.detach().cpu().item())
            n_val_batches += 1

    avg_val_loss = val_epoch_loss / max(1, n_val_batches)
    avg_task_val = {c: val_task_loss_acc[c] / max(1, n_val_batches) for c in target_cols}

    # Scheduler step
    scheduler.step(avg_val_loss)

    elapsed = time.time() - t0
    print(f"Epoch {epoch}/{EPOCHS} — train_loss: {avg_train_loss:.4f} — val_loss: {avg_val_loss:.4f} — time: {elapsed:.1f}s")
    # Print per-task summary (concise)
    per_task_str = " | ".join([f"{c}: train={avg_task_train[c]:.4f} val={avg_task_val[c]:.4f}" for c in target_cols])
    print(" Per-task losses:", per_task_str)

    # Save history
    history.append({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        **{f"train_{c}": avg_task_train[c] for c in target_cols},
        **{f"val_{c}": avg_task_val[c] for c in target_cols}
    })
    save_history(history)

    # Early stopping & save best
    if avg_val_loss < best_val:
        best_val = avg_val_loss
        no_improve = 0
        torch.save({"model_state": model.state_dict(), "epoch": epoch}, str(MODEL_PATH))
        print(" New best model saved to", MODEL_PATH)
    else:
        no_improve += 1
        print(f" No improvement for {no_improve} epoch(s).")
        if no_improve >= PATIENCE:
            print("Early stopping triggered. Stopping training.")
            break

print("Training finished. Best val loss:", best_val)
save_history(history)
