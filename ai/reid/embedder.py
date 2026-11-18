"""Turn person crops into L2-normalised appearance embeddings (torch or ONNX Runtime)."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .data import IMG_H, IMG_W, MEAN, STD
from .model import load_reid_model

_MEAN = np.array(MEAN, np.float32)
_STD = np.array(STD, np.float32)


def preprocess(crops: list[np.ndarray]) -> np.ndarray:
    """BGR uint8 crops -> float32 NCHW batch, same preprocessing as training."""
    out = np.empty((len(crops), 3, IMG_H, IMG_W), np.float32)
    for i, c in enumerate(crops):
        x = cv2.cvtColor(cv2.resize(c, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        out[i] = ((x.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)
    return out


def crop_person(frame: np.ndarray, bbox, min_h: int = 48, min_w: int = 20):
    """Cut a person out of a frame. Returns None when the box is too small to describe reliably."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if y2 - y1 < min_h or x2 - x1 < min_w:
        return None
    return frame[y1:y2, x1:x2]


class ReIDEmbedder:
    def __init__(self, ckpt: str, device: str = "auto", fp16: bool | None = None, batch_size: int = 64):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.fp16 = self.device.startswith("cuda") if fp16 is None else fp16
        self.batch_size = batch_size
        self.model, self.meta = load_reid_model(ckpt, self.device)
        self.dim = self.model.feat_dim

    @torch.inference_mode()
    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        outs = []
        for i in range(0, len(crops), self.batch_size):
            x = torch.from_numpy(preprocess(crops[i:i + self.batch_size])).to(self.device)
            with torch.autocast(device_type="cuda", enabled=self.fp16 and self.device.startswith("cuda")):
                f = self.model(x)
            outs.append(torch.nn.functional.normalize(f.float(), dim=1).cpu().numpy())
        return np.concatenate(outs)
