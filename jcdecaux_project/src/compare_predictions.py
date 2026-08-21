"""Compare, pour quelques images déjà labellisées, la prédiction du modèle
à la valeur réelle (celle de la base Excel) — utile pour vérifier "à l'œil"
la qualité du modèle sans attendre de nouvelles images.

Usage : python -m src.compare_predictions [nombre_images]
"""
import pickle
import sys

import pandas as pd

from config.config import OUTPUT_DIR, ENCODERS_PATH
from src.inference import predict_image


def compare(n=5, seed=42):
    df = pd.read_csv(OUTPUT_DIR / "df_labeled_with_enc.csv", encoding="utf-8-sig")
    with open(ENCODERS_PATH, "rb") as f:
        target_cols = pickle.load(f)["target_cols"]

    sample = df.sample(n=min(n, len(df)), random_state=seed)
    rows = []
    for _, row in sample.iterrows():
        pred = predict_image(row["image_path"])
        for c in target_cols:
            rows.append({
                "image": row["Nom_fichier_visuel"],
                "attribut": c,
                "predit": pred[c],
                "reel": row[c],
                "correct": str(pred[c]).strip() == str(row[c]).strip(),
            })

    df_compare = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "comparaison_predit_vs_reel.xlsx"
    df_compare.to_excel(out_path, index=False)

    print(f"Comparaison exportée : {out_path}")
    print(f"Taux d'accord global sur {n} image(s) : {df_compare['correct'].mean():.1%}")
    print("\nNote : une valeur réelle rare (peu d'exemples dans la base) peut avoir été "
          "fusionnée en 'Autre' pendant l'entraînement ; le modèle prédira alors 'Autre' "
          "même si la valeur réelle affichée ici est plus précise — ce n'est pas une erreur.")
    return df_compare


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    compare(n=n)
