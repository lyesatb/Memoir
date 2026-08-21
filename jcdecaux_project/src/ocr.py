"""Extraction de texte (OCR) et détection de QR code sur une affiche.

Tesseract (cité dans la bibliographie du mémoire) nécessite un binaire
externe non installé sur cette machine ; EasyOCR est utilisé à la place
(pur Python, déjà disponible), pour les mêmes besoins : mesurer la
présence/quantité de texte et vérifier la lisibilité annoncée par le
modèle de classification.

La détection de QR code utilise `cv2.QRCodeDetector`, déterministe et
sans entraînement : elle sert de vérification indépendante de l'attribut
"Présence d'un QR Code", pour lequel le modèle CNN a un F1 faible malgré
une bonne accuracy (classe très déséquilibrée).
"""
import sys

import cv2
import numpy as np

from config.config import OCR_LANGS, OCR_GPU
from src.img_utils import safe_open_rgb

# Limite la résolution interne de traitement d'EasyOCR (par défaut 2560px) :
# sur les grandes images (ex. affiches portrait haute résolution), la valeur
# par défaut provoque un Segmentation Fault sur cette machine à mémoire
# limitée (~8 Go RAM). 1024 évite le crash sans dégrader la détection du
# texte principal d'une affiche publicitaire.
OCR_CANVAS_SIZE = 480

# La console Windows par défaut (cp1252) plante sur les caractères Unicode
# (ex. "█") utilisés par la barre de progression du téléchargement des
# poids EasyOCR. On force stdout/stderr en UTF-8 pour éviter ce crash.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

_reader = None


def get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(OCR_LANGS, gpu=OCR_GPU)
    return _reader


def _load_bgr(image_path):
    img = safe_open_rgb(image_path)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def extract_text(image_path):
    """Retourne texte détecté, nb de caractères, ratio de surface occupée
    par le texte, et proportion de caractères en majuscule (proxy du
    style de texte)."""
    img_bgr = _load_bgr(image_path)
    h, w = img_bgr.shape[:2]
    img_area = max(1, h * w)

    reader = get_reader()
    results = reader.readtext(img_bgr, canvas_size=OCR_CANVAS_SIZE, mag_ratio=1.0)

    texts, text_area = [], 0
    for box, text, conf in results:
        texts.append(text)
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        text_area += max(0, max(xs) - min(xs)) * max(0, max(ys) - min(ys))

    full_text = " ".join(texts).strip()
    letters = [ch for ch in full_text if ch.isalpha()]
    upper_ratio = (sum(1 for ch in letters if ch.isupper()) / len(letters)) if letters else 0.0

    return {
        "ocr_text": full_text,
        "ocr_n_chars": len(full_text),
        "ocr_n_boxes": len(results),
        "ocr_text_area_ratio": round(text_area / img_area, 4),
        "ocr_upper_ratio": round(upper_ratio, 4),
    }


def detect_qr_code(image_path):
    img_bgr = _load_bgr(image_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(gray)
    return bool(points is not None and len(data) > 0)
