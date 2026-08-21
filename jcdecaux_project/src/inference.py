"""Inférence sur une image unique ou un lot de nouvelles images.

`predict_batch` est le point d'entrée pensé pour la prochaine étape du
projet : traiter une nouvelle base d'images (pas encore vues à
l'entraînement) et produire, pour chacune, les 20 attributs de qualité
visuelle, le texte OCR, la cohérence du logo, un score composite et des
recommandations — le tout consolidé dans un rapport Excel.
"""
import pickle
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torchvision import transforms

from config.config import (
    DEVICE, MODEL_PATH, IMG_SIZE, BACKBONE, ENCODERS_PATH, OUTPUT_DIR,
    VALID_IMAGE_EXTS, LOGO_REFERENCE_PATH,
)
from src.model import MultiTaskModel
from src.img_utils import safe_open_rgb
from src import ocr as ocr_module
from src import logo_match
from src import recommend
from src import team_export

_model = None
_encoders_data = None

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_model():
    global _model, _encoders_data
    if _model is None:
        with open(ENCODERS_PATH, "rb") as f:
            _encoders_data = pickle.load(f)
        model = MultiTaskModel(
            BACKBONE, _encoders_data["target_cols"], _encoders_data["col_types"], _encoders_data["encoders"]
        )
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
        model.to(DEVICE).eval()
        _model = model
    return _model, _encoders_data


def predict_image(img_path):
    """Prédit les attributs de qualité visuelle pour une image, décodés en libellés lisibles."""
    model, enc_data = get_model()
    encoders, target_cols = enc_data["encoders"], enc_data["target_cols"]

    img = _transform(safe_open_rgb(img_path)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)

    predictions = {}
    for c in target_cols:
        idx = int(torch.argmax(outputs[c], dim=1).item())
        predictions[c] = encoders[c].classes_[idx]
    return predictions


def guess_brand_from_filename(stem):
    """Devine la marque à partir du nom de fichier (ex. "BOMPARD (1)" -> "BOMPARD"),
    utile quand aucun Excel de déclaration de marque n'est fourni : pour la plupart
    des nouvelles affiches, le nom de fichier EST le nom de la marque, éventuellement
    suivi d'un suffixe de numérotation ("(1)", "-2", " 3"...).

    Heuristique volontairement prudente : ne retire que les suffixes numériques
    précédés d'un séparateur (espace/tiret/underscore), pour ne pas tronquer les
    marques dont le nom se termine légitimement par un chiffre (ex. "TF1", "M6").
    """
    s = stem.strip()
    s = re.sub(r"\s*\(\d+\)\s*$", "", s)
    s = re.sub(r"[\s_-]+\d+\s*$", "", s)
    return s.strip()


def predict_batch(images_dir, excel_path=None, brand_col="Marques", out_path=None):
    """
    images_dir : dossier contenant les nouvelles images à évaluer.
    excel_path : Excel optionnel avec au moins une colonne `Nom_fichier_visuel`
                 et `brand_col`, pour activer la vérification de cohérence logo.
                 À défaut, la marque est devinée depuis le nom de fichier
                 (voir `guess_brand_from_filename`).
    """
    images_dir = Path(images_dir)
    image_paths = sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in VALID_IMAGE_EXTS
    )
    if not image_paths:
        raise FileNotFoundError(f"Aucune image trouvée dans {images_dir}")

    declared_brand_by_stem = {}
    if excel_path is not None:
        df_new = pd.read_excel(excel_path)
        name_col = next((c for c in df_new.columns if "nom_fichier" in c.lower()), None)
        if name_col and brand_col in df_new.columns:
            for _, row in df_new.iterrows():
                declared_brand_by_stem[str(row[name_col]).strip()] = row[brand_col]

    reference = None
    if LOGO_REFERENCE_PATH.exists():
        reference = logo_match.load_logo_reference()

    rows = []
    for i, img_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {img_path.relative_to(images_dir)}", flush=True)
        try:
            predictions = predict_image(img_path)
        except Exception as e:
            rows.append({"image": str(img_path.relative_to(images_dir)), "erreur": str(e)})
            continue

        try:
            ocr_result = ocr_module.extract_text(img_path)
            ocr_result["qr_detected"] = ocr_module.detect_qr_code(img_path)
        except Exception as e:
            ocr_result = {"ocr_error": str(e)}

        declared_brand = declared_brand_by_stem.get(img_path.stem) or guess_brand_from_filename(img_path.stem)
        logo_result = {}
        if reference:
            try:
                logo_result = logo_match.match_logo(
                    img_path,
                    declared_brand=declared_brand,
                    place_du_logo_label=predictions.get("Place du logo"),
                    reference=reference,
                )
            except Exception as e:
                logo_result = {"logo_error": str(e)}

        scoring = recommend.score_poster(predictions, ocr_result=ocr_result, logo_result=logo_result)

        row = {"image": str(img_path.relative_to(images_dir)), "marque_declaree": declared_brand}
        row.update(predictions)
        row.update(ocr_result)
        row.update(logo_result)
        row.update(scoring)
        rows.append(row)

    df_report = pd.DataFrame(rows)

    if out_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"new_images_report_{timestamp}.xlsx"
    df_report.to_excel(out_path, index=False)
    print(f"Rapport exporté : {out_path} ({len(df_report)} images)")

    team_export.export_team_format(df_report)

    return df_report
