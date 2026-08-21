"""Export au format de la base d'origine (BDDCréaJCDecaux.xlsx), pour que
l'équipe puisse réutiliser les prédictions dans son propre processus
d'étiquetage, sans changer d'outil.

Contrairement au rapport détaillé de `predict_batch` (scores, OCR, logo...),
ce fichier ne contient QUE les colonnes de la base d'origine, avec les
attributs prédits par le modèle à la place d'une saisie manuelle.

Colonnes non déductibles d'une image seule (ID JCDECAUX, Reftest, FAMILLE,
CLASSE, Nombre de visuels) restent vides — à compléter manuellement si
besoin. "Notoriété (base YouGov)" reste vide aussi : exclue du modèle (cf.
ANALYSE_MEMOIRE.md, §5).

Le fichier de sortie est incrémental : à chaque appel, les nouvelles lignes
s'ajoutent à celles déjà présentes ; une image déjà présente (même
Nom_fichier_visuel) est mise à jour, pas dupliquée — comme demandé, "à
chaque fois qu'on insère des images, ça s'ajoute au fichier et il se met
à jour".
"""
import re
from pathlib import Path

import pandas as pd

from config.config import OUTPUT_DIR

TEAM_FORMAT_PATH = OUTPUT_DIR / "BDD_nouvelles_affiches_a_valider.xlsx"

ORIGINAL_COLUMNS = [
    "ID JCDECAUX", "Marques", "Année", "Reftest", "Nom_fichier_visuel",
    "FAMILLE", "CLASSE", "Nombre de visuels",
    "Type de campagne", "Couleur dominante", "Couleur propre à la marque ? ",
    "1er point d'accroche", "Personnages",
    "Si Egérie : est-elle propre à la marque ou également sur d'autres marques ? ",
    "Utilisation de l'égérie dans le temps", "Genre", "Tranche d'âge",
    "Quantité/Place du texte dans le visuel", "Discours utilisé",
    "Style de texte \n(Lettres)", "Taille du logo", "Eléments de branding",
    "Présence d'un QR Code", "Langue du claim", "Contraste", "Place du logo",
    "Notoriété (base YouGov)", "Mise en avant prix", "Charge visuelle",
]


def guess_annee_from_path(image_rel_path):
    """Cherche une année (4 chiffres, ex. "2024") dans le chemin relatif de l'image
    (nos nouvelles images sont rangées par dossier d'année : nv_images/2024/...)."""
    m = re.search(r"(19|20)\d{2}", str(image_rel_path))
    return int(m.group(0)) if m else None


def build_team_rows(df_report):
    """Convertit le DataFrame produit par predict_batch au format de la base d'origine."""
    rows = []
    for _, r in df_report.iterrows():
        if "erreur" in df_report.columns and pd.notna(r.get("erreur")):
            continue  # image non traitée (erreur de lecture), rien à reporter

        row = {c: None for c in ORIGINAL_COLUMNS}
        row["Marques"] = r.get("marque_declaree")
        row["Année"] = guess_annee_from_path(r.get("image"))
        row["Nom_fichier_visuel"] = r.get("image")
        for c in ORIGINAL_COLUMNS:
            if c in df_report.columns and row.get(c) is None:
                row[c] = r.get(c)
        rows.append(row)
    return pd.DataFrame(rows, columns=ORIGINAL_COLUMNS)


def export_team_format(df_report, out_path=TEAM_FORMAT_PATH):
    df_new = build_team_rows(df_report)

    if Path(out_path).exists():
        df_existing = pd.read_excel(out_path)
        combined = pd.concat([df_existing, df_new], ignore_index=True)
        combined = combined.drop_duplicates(subset="Nom_fichier_visuel", keep="last")
    else:
        combined = df_new

    combined.to_excel(out_path, index=False)
    print(f"Format équipe mis à jour : {out_path} ({len(combined)} ligne(s) au total)")
    return combined
