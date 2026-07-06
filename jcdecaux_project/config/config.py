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
IMG_SIZE = 224
BACKBONE = "efficientnet_b0"   # "efficientnet_b0" ou "resnet50"
BATCH_SIZE = 16
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
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

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

# --- num_workers ---
import platform
NUM_WORKERS = 0 if platform.system().lower().startswith("win") else 4
