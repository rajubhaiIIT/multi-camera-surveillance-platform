"""Re-ID network: ResNet backbone (ImageNet-pretrained) + BNNeck, the 'bag of tricks' baseline.

forward() in train mode  -> (class logits, raw feature)   [ID loss on logits, triplet loss on feature]
forward() in eval mode   -> BNNeck feature (use L2-normalised cosine similarity to compare people)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from .data import IMG_H, IMG_W, MEAN, STD


class ReIDNet(nn.Module):
    def __init__(self, backbone: str = "resnet50", num_classes: int = 751, feat_dim: int | None = 512,
                 pretrained: bool = True, last_stride_1: bool = True):
        super().__init__()
        net = getattr(torchvision.models, backbone)(weights="DEFAULT" if pretrained else None)
        native = net.fc.in_features
        if last_stride_1:           # keeps a larger feature map: a well-known Re-ID accuracy boost
            blk = net.layer4[0]
            (blk.conv2 if hasattr(blk, "conv3") else blk.conv1).stride = (1, 1)
            blk.downsample[0].stride = (1, 1)
        self.body = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                  net.layer1, net.layer2, net.layer3, net.layer4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # optional linear layer to shrink 2048 -> 512 (smaller DB, faster matching, fits a pgvector index)
        self.reduce = nn.Linear(native, feat_dim, bias=False) if feat_dim and feat_dim != native else nn.Identity()
        self.feat_dim = feat_dim or native
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.feat_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        if isinstance(self.reduce, nn.Linear):
            nn.init.kaiming_normal_(self.reduce.weight, mode="fan_out")

    def forward(self, x):
        f = self.reduce(self.pool(self.body(x)).flatten(1))
        fb = self.bn(f)
        return (self.classifier(fb), f) if self.training else fb


def save_checkpoint(path, model: ReIDNet, backbone: str, num_classes: int, epoch: int, metrics: dict | None):
    torch.save({"state_dict": model.state_dict(), "backbone": backbone, "feat_dim": model.feat_dim,
                "num_classes": num_classes, "img_size": (IMG_H, IMG_W), "mean": MEAN, "std": STD,
                "epoch": epoch, "metrics": metrics}, path)


def load_reid_model(path, device: str = "cpu") -> tuple[ReIDNet, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ReIDNet(ck["backbone"], ck["num_classes"], ck["feat_dim"], pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), {k: v for k, v in ck.items() if k != "state_dict"}
