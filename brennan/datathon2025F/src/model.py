import torch.nn as nn
import torchvision.models as tv
import timm

"""
model.py
---------
Defines the MosquitoNet model architecture.
Uses a configurable backbone (default: ResNet-34) with a custom classification head.
"""

class MosquitoNet(nn.Module):
    def __init__(self, backbone="resnet34", num_classes=10, pretrained=True, pool="avg"):
        super().__init__()
        if hasattr(tv, backbone):
            m = getattr(tv, backbone)(weights="DEFAULT" if pretrained else None)
            # torchvision resnets have .fc
            in_features = m.fc.in_features if hasattr(m, "fc") else m.classifier[1].in_features
            if hasattr(m, "fc"): m.fc = nn.Identity()
            else: m.classifier[-1] = nn.Identity()
            self.backbone = m; feat_dim = in_features
        else:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool=pool)
            feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, num_classes)
    def forward(self, x):
        return self.head(self.backbone(x))
