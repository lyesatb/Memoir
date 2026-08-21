"""Évaluation du modèle sur le jeu de test.

Ajoute la précision/rappel par classe (pas seulement l'accuracy et le F1
pondéré) : sur des attributs déséquilibrés (ex. "Présence d'un QR Code"),
l'accuracy seule est trompeuse (une classe majoritaire suffit à l'obtenir),
le rapport par classe permet de voir où le modèle échoue réellement.
"""
import pickle

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from config.config import DEVICE, MODEL_PATH, BACKBONE, OUTPUT_DIR, ENCODERS_PATH
from src.dataset import build_dataloaders
from src.model import MultiTaskModel

EVAL_DIR = OUTPUT_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def load_encoded_data():
    df_path = OUTPUT_DIR / "df_labeled_with_enc.csv"
    if not df_path.exists():
        raise FileNotFoundError(f"{df_path} introuvable : lancer `python -m src.encoding` d'abord.")
    df = pd.read_csv(df_path, encoding="utf-8-sig")
    with open(ENCODERS_PATH, "rb") as f:
        enc_data = pickle.load(f)
    return df, enc_data["encoders"], enc_data["col_types"], enc_data["target_cols"]


def evaluate():
    df, encoders, col_types, target_cols = load_encoded_data()
    _, _, test_loader = build_dataloaders(df, target_cols, col_types)

    model = MultiTaskModel(BACKBONE, target_cols, col_types, encoders)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.to(DEVICE)
    model.eval()

    all_preds = {c: [] for c in target_cols}
    all_targets = {c: [] for c in target_cols}

    with torch.no_grad():
        for imgs, targets, _ in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            for c in target_cols:
                pred = np.argmax(outputs[c].cpu().numpy(), axis=1)
                all_preds[c].extend(pred)
                all_targets[c].extend(targets[c].numpy())

    summary_rows = []
    per_class_rows = []

    for c in target_cols:
        y_true = np.array(all_targets[c])
        y_pred = np.array(all_preds[c])
        classes = encoders[c].classes_

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        summary_rows.append({"Attribut": c, "Accuracy": acc, "F1_weighted": f1, "n_classes": len(classes)})

        report = classification_report(
            y_true, y_pred, labels=range(len(classes)), target_names=classes,
            output_dict=True, zero_division=0,
        )
        for label, vals in report.items():
            if label in ("accuracy", "macro avg", "weighted avg"):
                continue
            per_class_rows.append({
                "Attribut": c, "Classe": label,
                "Precision": vals["precision"], "Rappel": vals["recall"],
                "F1": vals["f1-score"], "Support": vals["support"],
            })

        cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
        plt.figure(figsize=(6, 6))
        plt.imshow(cm)
        plt.title(f"Confusion Matrix – {c}"[:60])
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in c)[:50]
        plt.savefig(EVAL_DIR / f"cm_{safe_name}.png")
        plt.close()

    df_summary = pd.DataFrame(summary_rows)
    df_per_class = pd.DataFrame(per_class_rows)

    with pd.ExcelWriter(EVAL_DIR / "metrics.xlsx") as writer:
        df_summary.to_excel(writer, sheet_name="resume_par_attribut", index=False)
        df_per_class.to_excel(writer, sheet_name="precision_rappel_par_classe", index=False)

    print("Évaluation terminée.")
    print(df_summary.to_string(index=False))
    return df_summary, df_per_class


if __name__ == "__main__":
    evaluate()
