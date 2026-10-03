"""Market-1501 loading. File names look like 0002_c1s1_000451_03.jpg  ->  person 2, camera 1."""
from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, Sampler

NAME_RE = re.compile(r"^(-?\d+)_c(\d+)s\d+_\d+_\d+")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
IMG_H, IMG_W = 256, 128
FOLDERS = {"train": "bounding_box_train", "query": "query", "gallery": "bounding_box_test"}


def parse_market_name(name: str) -> tuple[int, int]:
    m = NAME_RE.match(Path(name).name)
    if not m:
        raise ValueError(f"not a Market-1501 file name: {name}")
    return int(m.group(1)), int(m.group(2))


def find_market_root(path: str | Path) -> Path:
    """Folder that directly contains bounding_box_train / bounding_box_test / query (searches below `path`)."""
    p = Path(path)
    hit = p if (p / "bounding_box_train").is_dir() else None
    if hit is None:
        found = next(p.rglob("bounding_box_train"), None) if p.exists() else None
        hit = found.parent if found else None
    if hit is None:
        raise FileNotFoundError(f"Market-1501 not found under {p}. Expected a folder with bounding_box_train/.")
    return hit


def load_split(root: Path, split: str) -> list[tuple[str, int, int]]:
    """Return [(image_path, person_id, camera_id)]. Gallery contains junk (-1) and distractor (0) ids by design."""
    folder = root / FOLDERS[split]
    items = []
    for f in sorted(folder.glob("*.jpg")):
        pid, cam = parse_market_name(f.name)
        items.append((str(f), pid, cam))
    if not items:
        raise FileNotFoundError(f"no .jpg files in {folder}")
    return items


def train_transform():
    return T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.RandomHorizontalFlip(),
        T.Pad(10),
        T.RandomCrop((IMG_H, IMG_W)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),   # helps across lighting conditions
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.5, value=0),                                 # simulates occlusion
    ])


def test_transform():
    return T.Compose([T.Resize((IMG_H, IMG_W)), T.ToTensor(), T.Normalize(MEAN, STD)])


class ReIDDataset(Dataset):
    def __init__(self, items, transform, relabel: bool = False):
        self.items, self.transform = items, transform
        self.pid2label = None
        if relabel:
            self.pid2label = {p: i for i, p in enumerate(sorted({pid for _, pid, _ in items}))}

    @property
    def num_classes(self) -> int:
        return len(self.pid2label or {})

    @property
    def labels(self) -> list[int]:
        return [self.pid2label[pid] if self.pid2label else pid for _, pid, _ in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, pid, cam = self.items[i]
        img = Image.open(path).convert("RGB")
        label = self.pid2label[pid] if self.pid2label else pid
        return self.transform(img), label, cam


class PKSampler(Sampler):
    """Each batch = P identities x K images each (needed for triplet loss)."""

    def __init__(self, labels, p: int, k: int, num_batches: int | None = None, seed: int = 0):
        self.by = defaultdict(list)
        for i, l in enumerate(labels):
            self.by[l].append(i)
        self.ids = sorted(self.by)
        self.p, self.k, self.seed, self.epoch = p, k, seed, 0
        self.n = num_batches or max(1, len(labels) // (p * k))
        if len(self.ids) < p:
            raise ValueError(f"need at least {p} identities, got {len(self.ids)}")

    def __len__(self):
        return self.n

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.n):
            batch = []
            for ident in rng.sample(self.ids, self.p):
                pool = self.by[ident]
                batch += rng.sample(pool, self.k) if len(pool) >= self.k else rng.choices(pool, k=self.k)
            yield batch
