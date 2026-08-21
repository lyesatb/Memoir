"""Vérification de cohérence logo <-> marque déclarée.

Aucune base publique de logos ne couvre les ~188 marques annonceurs
spécifiques de ce projet (Free, BUT, Cartier, Lacoste...) : entraîner un
détecteur de logo générique serait long (CPU) et peu pertinent avec aussi
peu d'images par marque. Approche retenue : matching par référence.

  1. L'attribut déjà labellisé "Place du logo" (7 zones) sert à recadrer
     automatiquement la zone probable du logo, sans entraînement.
  2. Le backbone EfficientNet-B0 déjà entraîné (src/model.py) sert
     d'extracteur de features pour ces recadrages.
  3. Une bibliothèque de référence par marque est construite en moyennant
     les embeddings des images existantes de chaque marque (>= 2 images).
  4. Une nouvelle image est comparée par similarité cosinus à la marque
     déclarée (et à la meilleure correspondance toutes marques confondues)
     pour détecter une incohérence logo/marque.
"""
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from config.config import (
    DEVICE, MODEL_PATH, IMG_SIZE, BACKBONE, ENCODERS_PATH,
    LOGO_REFERENCE_PATH, LOGO_MATCH_THRESHOLD, LOGO_MIN_IMAGES_PER_BRAND,
    PLACE_LOGO_BBOX_MAP,
)
from src.model import MultiTaskModel
from src.img_utils import safe_open_rgb

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_backbone = None


def crop_logo_region(img_pil, place_du_logo_label):
    w, h = img_pil.size
    bbox = PLACE_LOGO_BBOX_MAP.get(place_du_logo_label)
    if bbox is None:
        return img_pil
    x0, y0, x1, y1 = bbox
    box = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
    if box[2] <= box[0] or box[3] <= box[1]:
        return img_pil
    return img_pil.crop(box)


def get_backbone():
    global _backbone
    if _backbone is None:
        with open(ENCODERS_PATH, "rb") as f:
            enc_data = pickle.load(f)
        model = MultiTaskModel(BACKBONE, enc_data["target_cols"], enc_data["col_types"], enc_data["encoders"])
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
        model.to(DEVICE).eval()
        _backbone = model.backbone
    return _backbone


def embed_crop(img_pil):
    x = _transform(img_pil.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = get_backbone()(x)
    v = feat.squeeze(0).cpu().numpy()
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_logo_reference(df_labeled, brand_col="Marques", place_col="Place du logo", image_col="image_path"):
    embeddings_by_brand = defaultdict(list)
    n_skipped = 0

    for _, row in df_labeled.iterrows():
        brand = row.get(brand_col)
        img_path = row.get(image_col)
        place = row.get(place_col)
        if pd.isna(brand) or not isinstance(img_path, str) or not img_path:
            n_skipped += 1
            continue
        try:
            img = safe_open_rgb(img_path)
        except Exception:
            n_skipped += 1
            continue
        crop = crop_logo_region(img, place)
        embeddings_by_brand[brand].append(embed_crop(crop))

    reference = {
        brand: np.mean(embs, axis=0)
        for brand, embs in embeddings_by_brand.items()
        if len(embs) >= LOGO_MIN_IMAGES_PER_BRAND
    }

    with open(LOGO_REFERENCE_PATH, "wb") as f:
        pickle.dump(reference, f)

    print(f"Référence logo construite : {len(reference)}/{len(embeddings_by_brand)} marques "
          f"retenues (>= {LOGO_MIN_IMAGES_PER_BRAND} images), {n_skipped} lignes ignorées.")
    return reference


def load_logo_reference():
    with open(LOGO_REFERENCE_PATH, "rb") as f:
        return pickle.load(f)


def match_logo(image_path, declared_brand=None, place_du_logo_label=None, reference=None):
    if reference is None:
        reference = load_logo_reference()

    img = safe_open_rgb(image_path)
    crop = crop_logo_region(img, place_du_logo_label)
    emb = embed_crop(crop)

    sims = {brand: cosine_sim(emb, ref_emb) for brand, ref_emb in reference.items()}
    if not sims:
        return {"logo_best_match": None, "logo_best_score": None,
                "logo_declared_score": None, "logo_is_consistent": None}

    best_brand = max(sims, key=sims.get)
    declared_score = sims.get(declared_brand) if declared_brand else None

    return {
        "logo_best_match": best_brand,
        "logo_best_score": round(sims[best_brand], 4),
        "logo_declared_score": round(declared_score, 4) if declared_score is not None else None,
        "logo_is_consistent": (declared_score >= LOGO_MATCH_THRESHOLD) if declared_score is not None else None,
    }


if __name__ == "__main__":
    from config.config import OUTPUT_DIR
    df = pd.read_csv(OUTPUT_DIR / "df_labeled_with_enc.csv", encoding="utf-8-sig")
    build_logo_reference(df)
