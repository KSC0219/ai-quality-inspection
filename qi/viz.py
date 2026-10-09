"""Plots and overlays shared by both frameworks."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"          # series 1 / series 2 (Keras / PyTorch)
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def overlay_heatmap(image, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Single-hue (orange) overlay: the more the model 'looked' at a spot,
    the more orange it gets. Returns RGB uint8."""
    img = np.asarray(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    heat = cv2.resize(heat.astype(np.float32), (img.shape[1], img.shape[0]))
    heat = np.clip(heat, 0, 1)[..., None] * alpha
    tint = np.array([235, 104, 52], np.float32)
    return (img.astype(np.float32) * (1 - heat) + tint * heat).astype(np.uint8)


def plot_history(history: dict, path: str | Path, title: str) -> None:
    """Training curves: loss and accuracy in two panels (never a dual axis)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    epochs = np.arange(1, len(history["loss"]) + 1)
    for ax, key, label in ((axes[0], "loss", "Loss (binary cross-entropy)"), (axes[1], "accuracy", "Accuracy")):
        ax.plot(epochs, history[key], color=BLUE, lw=2, label="train")
        ax.plot(epochs, history["val_" + key], color=ORANGE, lw=2, label="validation")
        ax.set_title(label, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("epoch", fontsize=9, color=MUTED)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        _style(ax)
        ax.legend(frameon=False, fontsize=9)
    if "phase2_start" in history:
        for ax in axes:
            ax.axvline(history["phase2_start"] + 0.5, color=MUTED, lw=1, ls="--")
            ax.text(history["phase2_start"] + 0.6, ax.get_ylim()[1], " fine-tuning", fontsize=8,
                    color=MUTED, va="top")
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(cm, path: str | Path, title: str) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap="Blues")
    labels = ["OK", "Defective"]
    ax.set_xticks([0, 1], [f"pred {l}" for l in labels], fontsize=9)
    ax.set_yticks([0, 1], [f"true {l}" for l in labels], fontsize=9)
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=13,
                    color="white" if v > cm.max() * 0.55 else INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def gradcam_grid(images, heats, titles, path: str | Path, cols: int = 4) -> None:
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, img, h, t in zip(np.ravel(axes), images, heats, titles):
        ax.imshow(overlay_heatmap(img, h))
        ax.set_title(t, fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
