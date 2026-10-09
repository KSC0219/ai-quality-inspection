"""All settings in one place. Override any value from the command line, e.g.
    python scripts/train.py --framework keras ... --epochs-head 3 --img-size 160
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Folders can be redirected with environment variables (the tests use this).
DATA_DIR = Path(os.environ.get("QI_DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(os.environ.get("QI_MODELS_DIR", ROOT / "models"))
RESULTS_DIR = Path(os.environ.get("QI_RESULTS_DIR", ROOT / "results"))

# Label convention used EVERYWHERE (Keras, PyTorch, TFLite, API):
#   0 = ok_front (good part), 1 = def_front (defective part)
CLASS_DIRS = ["ok_front", "def_front"]
CLASS_NAMES = ["OK", "DEFECTIVE"]


@dataclass
class TrainConfig:
    img_size: int = 224
    batch_size: int = 32
    epochs_head: int = 5          # phase 1: train only the new classifier head
    epochs_finetune: int = 10     # phase 2: also update the top of the backbone
    lr_head: float = 1e-3
    lr_finetune: float = 1e-4
    patience: int = 3             # early stopping on validation loss
    val_fraction: float = 0.2     # carved out of the training folder
    target_recall: float = 0.98   # choose threshold so >=98% of defects are caught
    weights: str = "imagenet"     # "imagenet" (transfer learning) or "none"
    seed: int = 42
    workers: int = 2
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def add_train_args(parser) -> None:
    """Expose every TrainConfig field as a --flag."""
    for name, default in TrainConfig().to_dict().items():
        if name == "extra":
            continue
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)


def config_from_args(args) -> TrainConfig:
    return TrainConfig(**{k: getattr(args, k) for k in TrainConfig().to_dict() if k != "extra"})
