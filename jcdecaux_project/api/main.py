from fastapi import FastAPI, UploadFile, File
import shutil, os, pickle, torch
from PIL import Image
from torchvision import transforms
from config.config import DEVICE, IMG_SIZE, MODEL_PATH
from src.model import MultiTaskModel

app = FastAPI(title="JCDecaux Creative AI API")

with open("artifacts/encoders.pkl", "rb") as f:
    data = pickle.load(f)

model = MultiTaskModel(
    "efficientnet_b0",
    data["target_cols"],
    data["col_types"],
    data["encoders"]
)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
model.to(DEVICE).eval()

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    img = tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)

    response = {}
    for c, out in outputs.items():
        if data["col_types"][c] == "categorical":
            idx = int(torch.argmax(out))
            response[c] = data["encoders"][c].classes_[idx]
        else:
            response[c] = float(torch.sigmoid(out).item())

    return response
