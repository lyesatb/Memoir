# src/utils.py
import os
import json
import random
import torch
import numpy as np
from pathlib import Path

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_pickle(obj, path):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)

def save_model_checkpoint(state, path):
    ensure_dir(Path(path).parent)
    torch.save(state, path)
