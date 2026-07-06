import torch
import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    mean_squared_error
)

from config.config import DEVICE, MODEL_PATH
from src.dataset import build_dataloaders
from src.model import MultiTaskModel
from src.excel_matching import df_labeled
from src.encoding import target_cols, col_types, encoders

OUTPUT_DIR = "data/output/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def evaluate():
    _, _, test_loader = build_dataloaders(df_labeled, target_cols, col_types)

    model = MultiTaskModel(
        "efficientnet_b0",
        target_cols,
        col_types,
        encoders
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    metrics = []

    all_preds = {c: [] for c in target_cols}
    all_targets = {c: [] for c in target_cols}

    with torch.no_grad():
        for imgs, targets, _ in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)

            for c in target_cols:
                out = outputs[c].cpu().numpy()
                tgt = targets[c].numpy()

                if col_types[c] == "categorical":
                    pred = np.argmax(out, axis=1)
                else:
                    pred = out.squeeze()

                all_preds[c].extend(pred)
                all_targets[c].extend(tgt)

    for c in target_cols:
        y_true = np.array(all_targets[c])
        y_pred = np.array(all_preds[c])

        if col_types[c] == "categorical":
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted")
            metrics.append([c, "classification", acc, f1, None])

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 6))
            plt.imshow(cm)
            plt.title(f"Confusion Matrix – {c}")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/cm_{c}.png")
            plt.close()

        else:
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            metrics.append([c, "regression", None, None, rmse])

    df_metrics = pd.DataFrame(
        metrics,
        columns=["Attribut", "Type", "Accuracy", "F1-score", "RMSE"]
    )
    df_metrics.to_excel(f"{OUTPUT_DIR}/metrics.xlsx", index=False)

    print("Évaluation terminée ✔")
    print(df_metrics)

if __name__ == "__main__":
    evaluate()
