import torch.nn as nn
from torchvision import models

class MultiTaskModel(nn.Module):
    def __init__(self, backbone_name, target_cols, col_types, encoders):
        super().__init__()
        self.target_cols = target_cols
        self.col_types = col_types

        if backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=True)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        else:
            backbone = models.efficientnet_b0(pretrained=True)
            feat_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.heads = nn.ModuleDict()

        for c in target_cols:
            if col_types[c] == "categorical":
                self.heads[c] = nn.Linear(feat_dim, len(encoders[c].classes_))
            else:
                self.heads[c] = nn.Linear(feat_dim, 1)

    def forward(self, x):
        feats = self.backbone(x)
        return {c: self.heads[c](feats) for c in self.target_cols}
