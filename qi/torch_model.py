"""PyTorch model: ResNet18 transfer learning with a hand-written training loop.

Architecture
    image -> resize + ImageNet normalisation -> ResNet18 backbone (11M params)
    -> Dropout(0.3) -> Linear(512, 1) -> logit;  sigmoid(logit) = P(defective)

Training
    Loss: BCEWithLogitsLoss with pos_weight (handles class imbalance and is
          numerically more stable than sigmoid + BCE separately).
    Phase 1: only the new head (fc) trains.  Phase 2: layer3 + layer4 + head
    fine-tune with AdamW and a cosine learning-rate schedule.
    Early stopping keeps the weights with the lowest validation loss.
    Mixed precision (AMP) is used automatically on a GPU.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .config import TrainConfig
from .data import Split

FRAMEWORK = "pytorch"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ data
def make_transform(size: int, training: bool) -> T.Compose:
    steps = [T.Resize((size, size))]
    if training:
        steps += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(180),
                  T.RandomAffine(degrees=0, scale=(0.9, 1.1)), T.ColorJitter(brightness=0.15, contrast=0.2)]
    steps += [T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(steps)


class ImageListDataset(Dataset):
    def __init__(self, split: Split, transform):
        self.split, self.transform = split, transform

    def __len__(self):
        return len(self.split)

    def __getitem__(self, i):
        img = Image.open(self.split.paths[i]).convert("RGB")
        return self.transform(img), torch.tensor(self.split.labels[i], dtype=torch.float32)


def make_loader(split: Split, cfg: TrainConfig, training: bool) -> DataLoader:
    ds = ImageListDataset(split, make_transform(cfg.img_size, training))
    g = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=training, generator=g,
                      num_workers=cfg.workers, pin_memory=torch.cuda.is_available())


# ------------------------------------------------------------------ model
def build_model(cfg: TrainConfig) -> nn.Module:
    weights = None if cfg.weights == "none" else torchvision.models.ResNet18_Weights.DEFAULT
    m = torchvision.models.resnet18(weights=weights)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 1))
    return m


def _set_trainable(model: nn.Module, phase: int, scratch: bool = False) -> list:
    for p in model.parameters():
        p.requires_grad = scratch          # from scratch: everything trains
    blocks = [model.fc] if phase == 1 else [model.layer3, model.layer4, model.fc]
    for b in blocks:
        for p in b.parameters():
            p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def _run_epoch(model, loader, loss_fn, dev, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    use_amp = dev.type == "cuda"
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                logits = model(x).squeeze(1)
                loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(y)
            correct += ((logits.detach() > 0).float() == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def train(train_split: Split, val_split: Split, cfg: TrainConfig, log=print) -> tuple[nn.Module, dict]:
    torch.manual_seed(cfg.seed)
    dev = device()
    model = build_model(cfg).to(dev)
    tr, va = make_loader(train_split, cfg, True), make_loader(val_split, cfg, False)
    c = train_split.counts()
    pos_weight = torch.tensor(max(c["ok"], 1) / max(c["defective"], 1), device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best = {"val_loss": float("inf"), "state": copy.deepcopy(model.state_dict())}

    for phase, epochs, lr in ((1, cfg.epochs_head, cfg.lr_head), (2, cfg.epochs_finetune, cfg.lr_finetune)):
        if epochs <= 0:
            continue
        if phase == 2:
            history["phase2_start"] = len(history["loss"])
            model.load_state_dict(best["state"])
        params = _set_trainable(model, phase, scratch=cfg.weights == "none")
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        what = "whole network from scratch" if cfg.weights == "none" else (
            "head only" if phase == 1 else "layer3+layer4+head")
        log(f"[pytorch] phase {phase}: {what}, "
            f"{epochs} epochs on {dev}")
        bad_epochs = 0
        for ep in range(epochs):
            t0 = time.time()
            l, a = _run_epoch(model, tr, loss_fn, dev, opt, scaler)
            vl, vacc = _run_epoch(model, va, loss_fn, dev)
            sched.step()
            for k, v in zip(history, (l, a, vl, vacc)):
                history[k].append(float(v))
            log(f"  epoch {ep + 1}/{epochs} loss {l:.4f} acc {a:.4f} val_loss {vl:.4f} "
                f"val_acc {vacc:.4f} ({time.time() - t0:.0f}s)")
            if vl < best["val_loss"] - 1e-4:
                best = {"val_loss": vl, "state": copy.deepcopy(model.state_dict())}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    log("  early stopping")
                    break
    history.setdefault("phase2_start", len(history["loss"]))
    model.load_state_dict(best["state"])
    return model.eval(), history


@torch.no_grad()
def predict_probs(model: nn.Module, split: Split, cfg: TrainConfig) -> np.ndarray:
    dev = device()
    model.eval().to(dev)
    out = [torch.sigmoid(model(x.to(dev)).squeeze(1)).float().cpu().numpy()
           for x, _ in make_loader(split, cfg, False)]
    return np.concatenate(out)


def _to_tensor(img_rgb: np.ndarray, size: int) -> torch.Tensor:
    return make_transform(size, False)(Image.fromarray(img_rgb).convert("RGB"))[None]


@torch.no_grad()
def predict_one(model: nn.Module, img_rgb: np.ndarray, size: int) -> float:
    dev = next(model.parameters()).device
    return float(torch.sigmoid(model.eval()(_to_tensor(img_rgb, size).to(dev))).item())


# ------------------------------------------------------------------ Grad-CAM
def gradcam(model: nn.Module, img_rgb: np.ndarray, size: int) -> np.ndarray:
    """Grad-CAM on the last conv block (layer4) using forward/backward hooks."""
    model.eval()
    dev = next(model.parameters()).device
    store = {}
    h1 = model.layer4.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = model.layer4.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    try:
        x = _to_tensor(img_rgb, size).to(dev).requires_grad_(True)
        model.zero_grad()
        model(x).squeeze().backward()
        w = store["grad"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * store["act"]).sum(1))[0].detach().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()
    return cam / (cam.max() + 1e-8)


# ------------------------------------------------------------------ save / stats
def save(model: nn.Module, out_base: Path) -> list[Path]:
    path = out_base.with_name(out_base.name + ".pt")
    torch.save({"arch": "resnet18", "state_dict": model.state_dict()}, path)
    return [path]


def load(path: str | Path) -> nn.Module:
    ckpt = torch.load(path, map_location=device(), weights_only=True)
    m = build_model(TrainConfig(weights="none"))
    m.load_state_dict(ckpt["state_dict"])
    return m.to(device()).eval()


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


@torch.no_grad()
def latency_ms(model, img_rgb: np.ndarray, size: int, runs: int = 30) -> float:
    dev = next(model.parameters()).device
    x = _to_tensor(img_rgb, size).to(dev)
    model.eval()(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(runs):
        model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / runs * 1000
