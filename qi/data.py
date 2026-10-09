"""Dataset discovery and identical train/val/test splits for both frameworks.

Both Keras and PyTorch receive the SAME file lists (same seed, same split),
so their results can be compared fairly.
"""
from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import CLASS_DIRS, DATA_DIR

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

KAGGLE_SLUG = "ravirajsinh45/real-life-industrial-dataset-of-casting-product"


@dataclass
class Split:
    paths: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.paths)

    def counts(self) -> dict:
        return {"ok": self.labels.count(0), "defective": self.labels.count(1)}


def list_images(folder: str | Path) -> Split:
    """folder/ok_front/* -> label 0, folder/def_front/* -> label 1."""
    folder = Path(folder)
    paths, labels = [], []
    for label, cls in enumerate(CLASS_DIRS):
        d = folder / cls
        if not d.is_dir():
            raise FileNotFoundError(f"Missing folder {d}. Expected {CLASS_DIRS} inside {folder}")
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
        paths += [str(p) for p in files]
        labels += [label] * len(files)
    return Split(paths, labels)


def train_val_split(split: Split, val_fraction: float, seed: int) -> tuple[Split, Split]:
    """Stratified split (same OK/defective ratio in train and val).

    For CAD images, all views of one physical part stay on the same side, so
    validation never contains a near-copy of a training image (avoids leakage).
    """
    rng = random.Random(seed)
    train, val = Split([], []), Split([], [])
    for label in (0, 1):
        items = [p for p, l in zip(split.paths, split.labels) if l == label]
        groups: dict[str, list[str]] = {}
        for p in items:
            name = Path(p).name
            key = name.rsplit("_v", 1)[0] if name.startswith("cad_") else name
            groups.setdefault(key, []).append(p)
        keys = sorted(groups)
        rng.shuffle(keys)
        n_val = max(1, round(len(keys) * val_fraction))
        for i, k in enumerate(keys):
            target = val if i < n_val else train
            target.paths += groups[k]
            target.labels += [label] * len(groups[k])
    return train, val


def find_real_dataset(root: str | Path = DATA_DIR) -> Path:
    """Locate the Kaggle casting dataset folder that has train/ and test/."""
    root = Path(root)
    for train_dir in sorted(root.rglob("train")):
        if (train_dir / "def_front").is_dir() and (train_dir.parent / "test" / "def_front").is_dir():
            return train_dir.parent
    raise FileNotFoundError(
        f"Real dataset not found under {root}. Run: python scripts/download_data.py "
        f"(Kaggle dataset '{KAGGLE_SLUG}')")


def subsample(split: Split, fraction: float, seed: int) -> Split:
    """Keep a stratified fraction of a split (e.g. 10% of real images)."""
    rng = random.Random(seed)
    out = Split([], [])
    for label in (0, 1):
        items = [p for p, l in zip(split.paths, split.labels) if l == label]
        rng.shuffle(items)
        keep = sorted(items[: max(1, round(len(items) * fraction))])
        out.paths += keep
        out.labels += [label] * len(keep)
    return out


def merge(*splits: Split) -> Split:
    out = Split([], [])
    for s in splits:
        out.paths += s.paths
        out.labels += s.labels
    return out


def materialize(split: Split, out_dir: str | Path) -> Path:
    """Write a split as ok_front/ def_front/ folders (symlinks, copies on Windows)."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for cls in CLASS_DIRS:
        (out_dir / cls).mkdir(parents=True)
    for p, l in zip(split.paths, split.labels):
        dst = out_dir / CLASS_DIRS[l] / Path(p).name
        try:
            os.symlink(os.path.abspath(p), dst)
        except OSError:
            shutil.copy2(p, dst)
    return out_dir
