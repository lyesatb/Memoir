# src/train.py
import os
from pathlib import Path
import time
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from src.utils import set_seed, ensure_dir, save_model_checkpoint
from src.data import make_dataloaders
from src.model import MultiTaskModel

def safe_class_weights(counts):
    import numpy as np
    counts = np.array(counts, dtype=float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (counts * len(counts))
    return weights.tolist()

def compute_loss(outputs, targets, col_types, encoders, device):
    total_loss = 0.0
    losses = {}
    for c, out in outputs.items():
        ttype = col_types.get(c)
        tgt = targets[c].to(device)
        if ttype == "categorical":
            loss = F.cross_entropy(out, tgt.long())
        elif ttype == "binary":
            loss = F.binary_cross_entropy_with_logits(out.view(-1), tgt.float())
        else:
            loss = F.mse_loss(out.view(-1), tgt.float())
        losses[c] = loss.item()
        total_loss += loss
    return total_loss, losses

def train_loop(model, train_loader, val_loader, device, encoders, col_types, target_cols,
               output_folder, epochs=20, lr=1e-4, patience=5):
    ensure_dir(output_folder)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5, verbose=True)
    best_val = float("inf")
    history = []
    no_improve = 0

    for epoch in range(1, epochs+1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for imgs, targets, _ in tqdm(train_loader, desc=f"Train epoch {epoch}"):
            imgs = imgs.to(device)
            # move targets to device
            for k in targets:
                targets[k] = targets[k].to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss, _ = compute_loss(outputs, targets, col_types, encoders, device)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() if hasattr(loss, "item") else float(loss)

        train_loss /= len(train_loader)
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets, _ in val_loader:
                imgs = imgs.to(device)
                for k in targets:
                    targets[k] = targets[k].to(device)
                outputs = model(imgs)
                loss, _ = compute_loss(outputs, targets, col_types, encoders, device)
                val_loss += loss.item() if hasattr(loss, "item") else float(loss)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f"Epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} time={elapsed:.1f}s")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # early stopping & save best
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            ckpt_path = Path(output_folder) / f"best_model_epoch{epoch}.pth"
            save_model_checkpoint({"model_state": model.state_dict(), "epoch": epoch}, str(ckpt_path))
            print("Saved best model:", ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    # save history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(Path(output_folder) / "training_history.csv", index=False)
    return model, hist_df

if __name__ == "__main__":
    # Exemple d'utilisation minimal (adapter les chemins et DataFrames)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ATTENTION: remplace ces DataFrames par tes propres train/val/test
    import pandas as pd
    train_df = pd.read_excel("data/metadata/train.xlsx")
    val_df = pd.read_excel("data/metadata/val.xlsx")
    test_df = pd.read_excel("data/metadata/test.xlsx")
    target_cols = [c for c in train_df.columns if c not in ("filename",)]
    img_root = Path("data/raw")
    train_loader, val_loader, test_loader = make_dataloaders(train_df, val_df, test_df, img_root, target_cols,
                                                            img_size=224, batch_size=16, num_workers=4)
    # encoders.pkl doit exister (créé depuis utils d'encodage)
    import pickle
    enc = pickle.load(open("data/processed/encoders.pkl","rb"))
    encoders = enc["encoders"]
    col_types = enc["col_types"]
    BACKBONE = enc.get("BACKBONE","efficientnet_b0")
    model = MultiTaskModel(BACKBONE, col_types, target_cols, encoders).to(device)
    out_folder = "outputs/models"
    model, hist = train_loop(model, train_loader, val_loader, device, encoders, col_types, target_cols,
                             out_folder, epochs=20, lr=1e-4, patience=5)
