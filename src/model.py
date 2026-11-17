from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ReidNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        layer4 = base.layer4[0]
        if layer4.downsample is not None:
            layer4.downsample[0].stride = (1, 1)
        layer4.conv2.stride = (1, 1)
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(2048)
        self.bn.bias.requires_grad_(False)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)
        self.classifier = nn.Linear(2048, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, images: torch.Tensor):
        features = self.pool(self.backbone(images)).flatten(1)
        bottleneck = self.bn(features)
        if not self.training:
            return F.normalize(bottleneck, dim=1)
        logits = self.classifier(bottleneck)
        return features, logits


def build_model(backbone: str, num_classes: int) -> nn.Module:
    if backbone == "resnet50":
        return ReidNet(num_classes)
    raise ValueError(f"unknown backbone {backbone}")


def batch_hard_triplet(features: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    distance = torch.cdist(features, features, p=2)
    same = labels.view(-1, 1).eq(labels.view(1, -1))
    different = ~same
    eye = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    positive = distance.masked_fill(~same | eye, -1.0).max(dim=1).values
    negative = distance.masked_fill(~different, 1e6).min(dim=1).values
    return F.relu(positive - negative + margin).mean()
