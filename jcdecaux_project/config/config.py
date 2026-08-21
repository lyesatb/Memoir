import os
import torch
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

EXCEL_PATH = INPUT_DIR / "BDDCréaJCDecaux.xlsx"
SHEET_NAME = "Feuil2"
IMAGE_FOLDER = INPUT_DIR / "Visuale"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# --- Training params ---
IMG_SIZE = 160   # réduit de 224 (machine ~8 Go RAM, MemoryError observées à 224/batch=8)
BACKBONE = "efficientnet_b0"   # "efficientnet_b0" ou "resnet50"
BATCH_SIZE = 4   # réduit (machine à mémoire limitée, ~1 Go RAM libre observé)
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 6
SEED = 42
CONF_THRESH = 0.6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Files outputs ---
MODEL_PATH = ARTIFACTS_DIR / "best_model.pth"
ENCODERS_PATH = ARTIFACTS_DIR / "encoders.pkl"
ENCODERS_XLSX = ARTIFACTS_DIR / "encoders_mapping.xlsx"
METRICS_XLSX = OUTPUT_DIR / "evaluation_metrics.xlsx"
PRED_XLSX = OUTPUT_DIR / "predictions_non_matchees.xlsx"

# --- Image extensions ---
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".jfif"}

# --- Random seeds ---
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
import numpy as np
import random
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Split / sampler ---
SPLIT = (0.8, 0.1, 0.1)
USE_WEIGHTED_SAMPLER = False
SAMPLER_TARGET = None

# --- Colonnes de métadonnées (jamais des cibles de prédiction) ---
METADATA_COLS = [
    "ID JCDECAUX", "Marques", "Année", "Reftest", "Nom_fichier_visuel",
    "FAMILLE", "CLASSE", "Nombre de visuels", "image_path",
]

# --- Target column exclusions ---
# "Notoriété (base YouGov)" est une mesure externe de notoriété de marque
# (donnée YouGov), pas un critère de qualité visuelle : hors périmètre.
EXCLUDED_TARGET_COLS = ["Notoriété (base YouGov)"]

# --- Robustesse classes rares ---
RARE_CLASS_MIN = 15          # classes avec < N occurrences fusionnées dans "Autre"
RARE_CLASS_LABEL = "Autre"
USE_CLASS_WEIGHTS = True     # pondération inverse-fréquence par classe et par tâche
CLASS_WEIGHTS_PATH = ARTIFACTS_DIR / "class_weights.pkl"

# --- OCR ---
OCR_LANGS = ["fr", "en"]
OCR_GPU = False

# --- Détection / matching de logo ---
LOGO_REFERENCE_PATH = ARTIFACTS_DIR / "logo_reference.pkl"
LOGO_MATCH_THRESHOLD = 0.5
LOGO_MIN_IMAGES_PER_BRAND = 2

# Zones relatives (x0, y0, x1, y1) en fraction de l'image, dérivées du
# libellé humain "Place du logo" déjà présent dans la base labellisée.
PLACE_LOGO_BBOX_MAP = {
    "Haut gauche":             (0.0, 0.0, 0.35, 0.35),
    "Haut milieu":             (0.30, 0.0, 0.70, 0.35),
    "Haut droit":               (0.65, 0.0, 1.0, 0.35),
    "Bas gauche":               (0.0, 0.65, 0.35, 1.0),
    "Bas milieu":               (0.30, 0.65, 0.70, 1.0),
    "Bas droit":                 (0.65, 0.65, 1.0, 1.0),
    "Dans le corps du visuel":  (0.25, 0.25, 0.75, 0.75),
}

# --- num_workers ---
import platform
NUM_WORKERS = 0 if platform.system().lower().startswith("win") else 4
