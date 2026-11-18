"""Face detection and embedding with InsightFace buffalo_l (SCRFD + ArcFace).

The InsightFace weights are non-commercial research-only.
State this in docs/ethics.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Default model cache next to the project so we don't re-download on every env rebuild.
DEFAULT_ROOT = str(Path(__file__).resolve().parents[2] / "models" / "insightface")


@dataclass
class FaceRecord:
    """One detected face in one frame."""
    bbox: tuple          # (x1, y1, x2, y2) in pixels
    det_score: float     # detection confidence 0..1
    embedding: np.ndarray  # L2-normalised 512-d ArcFace embedding
    kps: np.ndarray | None  # 5 keypoints (5, 2) or None

    def to_dict(self) -> dict:
        return {"bbox": [round(float(v), 1) for v in self.bbox],
                "det_score": round(float(self.det_score), 4),
                "embedding": self.embedding.tolist()}

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def face_height(self) -> float:
        return max(0.0, float(self.bbox[3]) - float(self.bbox[1]))


class FaceDetector:
    """Detect faces and extract ArcFace embeddings from a BGR frame.

    Parameters
    ----------
    det_size : (width, height) fed to SCRFD. Larger = finds smaller faces, slower.
    min_det_score : drop faces below this detection confidence.
    min_face_height : drop faces shorter than this many pixels (too small to be reliable).
    model_root : where to cache the buffalo_l model pack (~326 MB, downloaded once).
    providers : ONNX Runtime execution providers. ['CUDAExecutionProvider', 'CPUExecutionProvider']
                for GPU, or ['CPUExecutionProvider'] for CPU.
    """

    def __init__(self, det_size: tuple = (640, 640), min_det_score: float = 0.5,
                 min_face_height: float = 40.0, model_root: str = DEFAULT_ROOT,
                 providers: list | None = None):
        import insightface
        self.min_det_score = min_det_score
        self.min_face_height = min_face_height
        if providers is None:
            try:
                import onnxruntime as ort
                available = ort.get_available_providers()
                providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                             if "CUDAExecutionProvider" in available
                             else ["CPUExecutionProvider"])
            except Exception:
                providers = ["CPUExecutionProvider"]
        self.providers = providers
        self._app = insightface.app.FaceAnalysis(
            name="buffalo_l", root=model_root,
            providers=providers,
            allowed_modules=["detection", "recognition"])
        self._app.prepare(ctx_id=0 if any("CUDA" in p for p in providers) else -1,
                          det_size=det_size)

    def detect(self, image: np.ndarray) -> list[FaceRecord]:
        """Return detected faces in a BGR frame, filtered by score and size."""
        faces = self._app.get(image)
        out = []
        for f in faces:
            score = float(f.det_score)
            if score < self.min_det_score:
                continue
            bb = tuple(float(v) for v in f.bbox)
            if bb[3] - bb[1] < self.min_face_height:
                continue
            emb = np.asarray(f.embedding, np.float32)
            norm = np.linalg.norm(emb)
            emb = emb / (norm + 1e-12)                # L2-normalise
            kps = np.asarray(f.kps, np.float32) if f.kps is not None else None
            out.append(FaceRecord(bb, score, emb, kps))
        return out

    def draw(self, image: np.ndarray, faces: list[FaceRecord]) -> np.ndarray:
        """Draw bounding boxes and detection scores on a copy of the frame."""
        out = image.copy()
        for f in faces:
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
            label = f"{f.det_score:.2f}"
            cv2.putText(out, label, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
            if f.kps is not None:
                for pt in f.kps:
                    cv2.circle(out, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
        return out
