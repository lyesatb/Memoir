"""Encodage des attributs cibles pour l'entraînement multi-tâches.

Ce module :
  1. détermine les colonnes cibles (tous les attributs qualité, hors
     métadonnées et hors colonnes exclues comme "Notoriété (base YouGov)") ;
  2. fusionne les classes rares (< RARE_CLASS_MIN occurrences) dans une
     catégorie "Autre" par colonne, pour limiter le surapprentissage sur
     des classes à quelques exemplaires seulement ;
  3. encode chaque colonne cible avec un LabelEncoder ;
  4. calcule des poids de classe (inverse-fréquence) par tâche, utilisés
     par l'entraînement pour compenser le déséquilibre ;
  5. sauvegarde le DataFrame encodé et tous les artefacts sur disque, afin
     que train.py / evaluate.py les rechargent sans dépendre d'un ordre
     d'import particulier.
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config.config import (
    OUTPUT_DIR, ARTIFACTS_DIR, ENCODERS_PATH, ENCODERS_XLSX,
    CLASS_WEIGHTS_PATH, METADATA_COLS, EXCLUDED_TARGET_COLS,
    RARE_CLASS_MIN, RARE_CLASS_LABEL,
)
from src.excel_matching import df_labeled as _df_labeled

NA_LABEL = "Non renseigné"


def get_target_cols(df):
    return [c for c in df.columns if c not in METADATA_COLS and c not in EXCLUDED_TARGET_COLS]


def clean_column(series):
    return series.fillna(NA_LABEL).astype(str).str.strip().replace("", NA_LABEL)


def merge_rare_classes(series, min_count=RARE_CLASS_MIN, rare_label=RARE_CLASS_LABEL):
    counts = series.value_counts()
    rare = counts[counts < min_count].index
    # Ne fusionne pas si ça ne laisse qu'une seule classe restante.
    if len(rare) == 0 or len(counts) - len(rare) < 2:
        return series, 0
    merged = series.where(~series.isin(rare), rare_label)
    return merged, len(rare)


def build_encoding(df_labeled):
    df = df_labeled.copy().reset_index(drop=True)
    target_cols = get_target_cols(df)

    encoders = {}
    col_types = {}
    class_weights = {}
    merge_report = []

    for c in target_cols:
        cleaned = clean_column(df[c])
        merged, n_rare = merge_rare_classes(cleaned)
        if n_rare:
            merge_report.append((c, n_rare, cleaned.nunique(), merged.nunique()))

        enc = LabelEncoder()
        df[c + "_enc"] = enc.fit_transform(merged)
        encoders[c] = enc
        col_types[c] = "categorical"

        counts = np.bincount(df[c + "_enc"].values, minlength=len(enc.classes_)).astype(float)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (counts * len(counts))
        class_weights[c] = weights.tolist()

    return df, target_cols, encoders, col_types, class_weights, merge_report


def save_artifacts(df, target_cols, encoders, col_types, class_weights):
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump({"encoders": encoders, "col_types": col_types, "target_cols": target_cols}, f)

    with open(CLASS_WEIGHTS_PATH, "wb") as f:
        pickle.dump(class_weights, f)

    rows = []
    for c, enc in encoders.items():
        for label, class_id in zip(enc.classes_, range(len(enc.classes_))):
            rows.append({"column": c, "label": label, "class_id": class_id})
    pd.DataFrame(rows).to_excel(ENCODERS_XLSX, index=False)

    out_path = OUTPUT_DIR / "df_labeled_with_enc.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


# Exécuté à l'import, même convention que src/excel_matching.py : les
# modules qui en dépendent (train.py, evaluate.py) relisent ensuite les
# artefacts sur disque plutôt que de dépendre de l'ordre des imports.
df_encoded, target_cols, encoders, col_types, class_weights, merge_report = build_encoding(_df_labeled)
out_path = save_artifacts(df_encoded, target_cols, encoders, col_types, class_weights)

print(f"Encodage terminé : {len(target_cols)} colonnes cibles.")
if merge_report:
    print(f"Fusion des classes rares (< {RARE_CLASS_MIN} occurrences) :")
    for c, n_rare, before, after in merge_report:
        print(f"  - {c}: {n_rare} classe(s) rare(s) fusionnée(s) ({before} -> {after} classes)")
print(f"DataFrame encodé sauvegardé dans {out_path}")
print(f"Encodeurs sauvegardés dans {ENCODERS_PATH}")
