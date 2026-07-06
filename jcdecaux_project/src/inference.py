import torch
import pickle
from PIL import Image
from torchvision import transforms
from config.config import DEVICE, MODEL_PATH, IMG_SIZE
from src.model import MultiTaskModel

def predict_image(img_path):
    with open("artifacts/encoders.pkl", "rb") as f:
        data = pickle.load(f)

    model = MultiTaskModel(
        "efficientnet_b0",
        data["target_cols"],
        data["col_types"],
        data["encoders"]
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    img = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)

    return outputs
