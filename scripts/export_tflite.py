"""Convert a trained Keras model to TensorFlow Lite and check it still agrees
with the original on the test set.

    python scripts/export_tflite.py                    # experiment "real"
    python scripts/export_tflite.py --experiment cad_plus_10
"""
import argparse
import json
import time

import _bootstrap  # noqa: F401
import numpy as np
from PIL import Image

from qi import keras_model as km
from qi.config import DATA_DIR, MODELS_DIR, RESULTS_DIR
from qi.data import find_real_dataset, list_images
from qi.metrics import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="real")
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--no-quantize", action="store_true")
    a = ap.parse_args()

    mdir = MODELS_DIR / a.experiment
    meta = json.loads((mdir / "keras_meta.json").read_text())
    model = km.load(mdir / "keras_model.keras")
    out = km.export_tflite(model, mdir / "keras_model.tflite", quantize=not a.no_quantize)
    tfl = km.TFLitePredictor(out)

    test = list_images(a.test_dir or find_real_dataset(DATA_DIR) / "test")
    size, thr = meta["img_size"], meta["threshold"]
    p_k, p_t, times = [], [], []
    for path in test.paths:
        img = np.array(Image.open(path).convert("RGB"))
        p_k.append(km.predict_one(model, img, size))
        t = time.perf_counter()
        p_t.append(tfl.predict_one(img))
        times.append((time.perf_counter() - t) * 1000)
    p_k, p_t = np.array(p_k), np.array(p_t)
    report = {
        "experiment": a.experiment,
        "keras_size_mb": round((mdir / "keras_model.keras").stat().st_size / 1e6, 2),
        "tflite_size_mb": round(out.stat().st_size / 1e6, 2),
        "quantized": not a.no_quantize,
        "same_decision_rate": round(float(np.mean((p_k >= thr) == (p_t >= thr))), 4),
        "max_prob_difference": round(float(np.abs(p_k - p_t).max()), 4),
        "tflite_latency_ms": round(float(np.median(times)), 2),
        "tflite_test_metrics": evaluate(test.labels, p_t, thr),
    }
    (RESULTS_DIR / a.experiment).mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / a.experiment / "tflite.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "tflite_test_metrics"}, indent=2))


if __name__ == "__main__":
    main()
