"""One training + evaluation routine used for BOTH frameworks, so the
comparison is fair: same data split, same metrics, same reports.

Outputs for an experiment <exp> and framework <fw>:
    models/<exp>/<fw>_model.keras|.pt  + <fw>_meta.json (threshold, img size ...)
    results/<exp>/<fw>/metrics.json, history.json, history.png,
                       confusion_matrix.png, gradcam_examples.png, mistakes.csv
"""
from __future__ import annotations

import csv
import importlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .config import CLASS_NAMES, MODELS_DIR, RESULTS_DIR, TrainConfig
from .data import Split, list_images, train_val_split
from .metrics import full_report
from .viz import gradcam_grid, plot_confusion, plot_history

FRAMEWORKS = {"keras": "qi.keras_model", "pytorch": "qi.torch_model"}


def _rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _pick_examples(test: Split, probs: np.ndarray, thr: float, k: int = 8) -> list[tuple[int, str]]:
    y = np.array(test.labels)
    pred = probs >= thr
    picks = []
    groups = [("caught defect", np.where((y == 1) & pred)[0], -probs),
              ("missed defect", np.where((y == 1) & ~pred)[0], -probs),
              ("false alarm", np.where((y == 0) & pred)[0], -probs),
              ("correct OK", np.where((y == 0) & ~pred)[0], probs)]
    for name, idx, key in groups:
        idx = idx[np.argsort(key[idx])][: max(2, k // 4)]
        picks += [(int(i), name) for i in idx]
    return picks[:k]


def run(framework: str, train_dir: str | Path, test_dir: str | Path, experiment: str,
        cfg: TrainConfig, log=print) -> dict:
    fw = importlib.import_module(FRAMEWORKS[framework])
    full_train = list_images(train_dir)
    train_split, val_split = train_val_split(full_train, cfg.val_fraction, cfg.seed)
    test = list_images(test_dir)
    log(f"[{framework}] {experiment}: train {train_split.counts()} | val {val_split.counts()} | "
        f"test {test.counts()}")

    t0 = time.time()
    model, history = fw.train(train_split, val_split, cfg, log)
    train_seconds = time.time() - t0

    p_val = fw.predict_probs(model, val_split, cfg)
    p_test = fw.predict_probs(model, test, cfg)
    report = full_report(val_split.labels, p_val, test.labels, p_test, cfg.target_recall)
    thr = report["chosen_threshold"]

    model_dir = MODELS_DIR / experiment
    res_dir = RESULTS_DIR / experiment / framework
    model_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    files = fw.save(model, model_dir / f"{framework}_model")
    meta = {"framework": framework, "experiment": experiment, "threshold": thr,
            "img_size": cfg.img_size, "class_names": CLASS_NAMES,
            "label_meaning": "output = probability that the part is DEFECTIVE"}
    (model_dir / f"{framework}_meta.json").write_text(json.dumps(meta, indent=2))

    sample = _rgb(test.paths[0])
    report.update({
        "framework": framework, "experiment": experiment,
        "train_dir": str(train_dir), "test_dir": str(test_dir),
        "train_counts": train_split.counts(), "val_counts": val_split.counts(), "test_counts": test.counts(),
        "train_seconds": round(train_seconds, 1),
        "epochs_run": len(history["loss"]),
        "params": fw.count_params(model),
        "model_size_mb": round(sum(f.stat().st_size for f in files) / 1e6, 2),
        "latency_ms_per_image": round(fw.latency_ms(model, sample, cfg.img_size), 2),
        "config": cfg.to_dict(),
    })
    (res_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (res_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, res_dir / "history.png", f"{framework} - {experiment}")
    plot_confusion(report["test_at_chosen_threshold"]["confusion_matrix"], res_dir / "confusion_matrix.png",
                   f"{framework} - test set (threshold {thr:.2f})")

    picks = _pick_examples(test, p_test, thr)
    imgs = [_rgb(test.paths[i]) for i, _ in picks]
    heats = [fw.gradcam(model, im, cfg.img_size) for im in imgs]
    titles = [f"{name}\nP(defect)={p_test[i]:.2f}" for i, name in picks]
    gradcam_grid(imgs, heats, titles, res_dir / "gradcam_examples.png")

    y = np.array(test.labels)
    with open(res_dir / "mistakes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "true", "p_defective", "error"])
        for i in np.where((p_test >= thr) != (y == 1))[0]:
            w.writerow([test.paths[i], CLASS_NAMES[y[i]], round(float(p_test[i]), 4),
                        "missed defect" if y[i] == 1 else "false alarm"])

    t = report["test_at_chosen_threshold"]
    log(f"[{framework}] {experiment}: test accuracy {t['accuracy']:.4f}, defect recall "
        f"{t['recall_defective']:.4f}, precision {t['precision_defective']:.4f} at threshold {thr:.3f}")
    return report
