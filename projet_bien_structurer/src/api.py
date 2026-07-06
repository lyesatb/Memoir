# src/api.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io, os, pickle
from pathlib import Path
from PIL import Image
import torch
import pandas as pd
from typing import List

from src.model import MultiTaskModel
from src.data import get_transforms

OUTPUT_FOLDER = Path("outputs")
MODELS_FOLDER = OUTPUT_FOLDER / "models"
ENCODERS_PATH = Path("data/processed/encoders.pkl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not ENCODERS_PATH.exists():
    raise RuntimeError(f"Encoders file not found: {ENCODERS_PATH}")

with open(ENCODERS_PATH, "rb") as f:
    meta = pickle.load(f)
encoders = meta["encoders"]
col_types = meta["col_types"]
target_cols = meta["target_cols"]
BACKBONE = meta.get("BACKBONE", "efficientnet_b0")
IMG_SIZE = int(meta.get("IMG_SIZE", 224))

# load latest model
candidates = sorted(MODELS_FOLDER.glob("best_model*.pth"))
if not candidates:
    raise RuntimeError(f"No model found in {MODELS_FOLDER}")
ckpt = torch.load(str(candidates[-1]), map_location=DEVICE)
model = MultiTaskModel(BACKBONE, col_types, target_cols, encoders)
if "model_state" in ckpt:
    model.load_state_dict(ckpt["model_state"])
else:
    model.load_state_dict(ckpt)
model.to(DEVICE)
model.eval()

app = FastAPI(title="Predict API")

val_transforms = get_transforms(IMG_SIZE, train=False)

def preprocess_pil(img_pil):
    img = val_transforms(img_pil).unsqueeze(0).to(DEVICE)
    return img

def decode_single_output(outputs):
    row = {}
    for c in target_cols:
        out = outputs[c][0]
        ttype = col_types.get(c)
        if ttype == "categorical":
            idx = int(torch.argmax(out).cpu().item())
            label = encoders[c].classes_[idx] if c in encoders else str(idx)
            row[c + "_pred_label"] = label
        elif ttype == "binary":
            prob = float(torch.sigmoid(out.view(-1)).cpu().item())
            row[c + "_pred_prob"] = prob
            row[c + "_pred"] = int(prob >= 0.5)
        else:
            val = float(out.view(-1).cpu().item())
            row[c + "_pred"] = val
    return row

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict_single")
async def predict_single(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        inp = preprocess_pil(img)
        with torch.no_grad():
            outputs = model(inp)
        row = decode_single_output(outputs)
        return JSONResponse(content=row)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    rows = []
    try:
        for f in files:
            contents = await f.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            inp = preprocess_pil(img)
            with torch.no_grad():
                outputs = model(inp)
            row = decode_single_output(outputs)
            row["filename"] = f.filename
            rows.append(row)
        df_out = pd.DataFrame(rows)
        out_path = OUTPUT_FOLDER / "predictions" / f"predictions_api_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_excel(out_path, index=False)
        return JSONResponse(content={"n_images": len(rows), "excel_path": str(out_path)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
