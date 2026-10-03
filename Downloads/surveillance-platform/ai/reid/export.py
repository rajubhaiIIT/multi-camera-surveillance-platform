#!/usr/bin/env python3
"""Export a Re-ID checkpoint to ONNX (embedding output is already L2-normalised).

    python -m ai.reid.export --ckpt models/reid/resnet50_market/best.pt --out models/reid/reid.onnx
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from .data import IMG_H, IMG_W
from .embedder import preprocess
from .model import load_reid_model


class _Normalised(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.nn.functional.normalize(self.m(x), dim=1)


def export_onnx(ckpt: str, out: str, opset: int = 17) -> str:
    model, _ = load_reid_model(ckpt, "cpu")
    wrapper = _Normalised(model).eval()
    dummy = torch.randn(2, 3, IMG_H, IMG_W)
    torch.onnx.export(wrapper, dummy, out, input_names=["images"], output_names=["embedding"],
                      dynamic_axes={"images": {0: "batch"}, "embedding": {0: "batch"}},
                      opset_version=opset, dynamo=False)
    return out


class OnnxEmbedder:
    """Same interface as ReIDEmbedder, but runs the exported ONNX file with ONNX Runtime."""

    def __init__(self, onnx_path: str, providers: list[str] | None = None, batch_size: int = 64):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=providers or ["CPUExecutionProvider"])
        self.batch_size = batch_size
        self.dim = self.sess.get_outputs()[0].shape[1]

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        outs = [self.sess.run(None, {"images": preprocess(crops[i:i + self.batch_size])})[0]
                for i in range(0, len(crops), self.batch_size)]
        return np.concatenate(outs).astype(np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    print("wrote", export_onnx(a.ckpt, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
