# src/eval.py
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.utils import load_pickle
from src.data import get_transforms
from src.model import MultiTaskModel

def decode_outputs(outputs, col_types, encoders):
    row = {}
    for c, out in outputs.items():
        ttype = col_types.get(c)
        if ttype == "categorical":
            idx = int(torch.argmax(out, dim=-1).cpu().item())
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

def run_inference(model, test_df, img_root, col_types, encoders, img_size=224, device="cpu"):
    model.eval()
    results = []
    transforms = get_transforms(img_size, train=False)
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            from PIL import Image
            img = Image.open(Path(img_root)/row["filename"]).convert("RGB")
            x = transforms(img).unsqueeze(0).to(device)
            outputs = model(x)
            decoded = decode_outputs(outputs, col_types, encoders)
            decoded["filename"] = row["filename"]
            results.append(decoded)
    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = load_pickle("data/processed/encoders.pkl")
    encoders = enc["encoders"]; col_types = enc["col_types"]; target_cols = enc["target_cols"]
    model_ckpt = "outputs/models/best_model_epoch10.pth"
    ckpt = torch.load(model_ckpt, map_location=device)
    model = MultiTaskModel(enc.get("BACKBONE","efficientnet_b0"), col_types, target_cols, encoders)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    test_df = pd.read_excel("data/metadata/test.xlsx")
    df_pred = run_inference(model, test_df, "data/raw", col_types, encoders, img_size=enc.get("IMG_SIZE",224), device=device)
    df_pred.to_excel("outputs/predictions/predictions_test_decoded.xlsx", index=False)
