import torch
import pickle
from config.config import DEVICE, BACKBONE, EPOCHS, MODEL_PATH
from src.model import MultiTaskModel
from src.dataset import build_dataloaders
from src.excel_matching import df_labeled
from src.encoding import encoders, col_types, target_cols

train_loader, val_loader, test_loader = build_dataloaders(df_labeled, target_cols, col_types)

model = MultiTaskModel(BACKBONE, target_cols, col_types, encoders).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
losses = {
    c: torch.nn.CrossEntropyLoss() if col_types[c]=="categorical"
    else torch.nn.BCEWithLogitsLoss()
    for c in target_cols
}

best_loss = 1e9
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, targets, _ in train_loader:
        imgs = imgs.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = 0
        for c in target_cols:
            out = outputs[c].squeeze()
            tgt = targets[c].to(DEVICE)
            loss += losses[c](out, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss {total_loss:.4f}")
    if total_loss < best_loss:
        best_loss = total_loss
        torch.save(model.state_dict(), MODEL_PATH)
