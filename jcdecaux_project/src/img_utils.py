"""Ouverture d'image robuste, partagée par tous les modules du pipeline.

Certaines images fournies (scans haute définition) dépassent la limite de
sécurité par défaut de Pillow (protection "decompression bomb", ~179
mégapixels) et sont rejetées par un simple `Image.open(...).convert("RGB")`.
On lève cette limite mais on utilise `Image.draft()` pour que le décodeur
JPEG downscale l'image *pendant* le décodage (au lieu de décoder à pleine
résolution puis redimensionner) — indispensable sur une machine à mémoire
limitée où décoder une image de 289 Mpx en RAM ferait à nouveau planter
le processus.
"""
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # images fournies par un usage interne connu, pas une source non fiable


def safe_open_rgb(path, max_dim=1600):
    img = Image.open(path)
    try:
        img.draft("RGB", (max_dim, max_dim))
    except Exception:
        pass  # draft() ne s'applique qu'aux JPEG ; on retombe sur un décodage classique
    return img.convert("RGB")
