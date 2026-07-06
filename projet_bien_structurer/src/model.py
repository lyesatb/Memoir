# src/model.py
import torch.nn as nn
from torchvision import models

class MultiTaskModel(nn.Module):
    def __init__(self, backbone_name: str, col_types: dict, target_cols: list, encoders: dict):
        super().__init__()
        self.col_types = col_types
        self.target_cols = target_cols

        if backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=True)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        else:
            backbone = models.efficientnet_b0(pretrained=True)
            # efficientnet_b0 classifier is usually Sequential; remove it
            try:
                feat_dim = backbone.classifier[1].in_features
            except Exception:
                feat_dim = 1280
            backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.heads = nn.ModuleDict()
        for c in target_cols:
            t = col_types.get(c, "categorical")
            if t == "categorical":
                n_classes = len(encoders[c].classes_)
                self.heads[c] = nn.Sequential(
                    nn.Linear(feat_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(512, n_classes)
                )
            elif t == "binary":
                self.heads[c] = nn.Sequential(
                    nn.Linear(feat_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(256, 1)
                )
            else:
                self.heads[c] = nn.Sequential(
                    nn.Linear(feat_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(256, 1)
                )

    def forward(self, x):
        feats = self.backbone(x)
        out = {}
        for c in self.target_cols:
            out[c] = self.heads[c](feats)
        return out
