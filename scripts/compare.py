"""Collect every results/<experiment>/<framework>/metrics.json into one table + chart.

    python scripts/compare.py
Writes results/comparison.csv, results/comparison.md, results/comparison.png
"""
import json

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from qi.config import RESULTS_DIR  # noqa: E402
from qi.viz import BLUE, INK, MUTED, ORANGE, _style  # noqa: E402

ORDER = ["real", "cad_only", "cad_plus_10"]
LABELS = {"real": "Real photos", "cad_only": "CAD only", "cad_plus_10": "CAD + 10% real"}


def main():
    rows = []
    for f in sorted(RESULTS_DIR.glob("*/*/metrics.json")):
        m = json.loads(f.read_text())
        t = m["test_at_chosen_threshold"]
        rows.append({
            "experiment": m["experiment"], "framework": m["framework"],
            "accuracy": t["accuracy"], "recall_defective": t["recall_defective"],
            "precision_defective": t["precision_defective"], "f1_defective": t["f1_defective"],
            "roc_auc": t["roc_auc"], "missed_defects": t["missed_defects"], "false_alarms": t["false_alarms"],
            "threshold": m["chosen_threshold"], "accuracy_at_0.5": m["test_at_0.5"]["accuracy"],
            "params_millions": round(m["params"] / 1e6, 2), "model_size_mb": m["model_size_mb"],
            "latency_ms": m["latency_ms_per_image"], "train_minutes": round(m["train_seconds"] / 60, 1),
            "epochs": m["epochs_run"],
        })
    if not rows:
        raise SystemExit("No results yet. Run scripts/run_experiments.py first.")
    rows.sort(key=lambda r: (ORDER.index(r["experiment"]) if r["experiment"] in ORDER else 99, r["framework"]))

    cols = list(rows[0].keys())
    (RESULTS_DIR / "comparison.csv").write_text(
        "\n".join([",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows]) + "\n")
    md = ["| Training data | Framework | Accuracy | Defect recall | Precision | Missed defects | False alarms | Size (MB) | ms / image |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {LABELS.get(r['experiment'], r['experiment'])} | {r['framework']} | {r['accuracy']:.2%} | "
                  f"{r['recall_defective']:.2%} | {r['precision_defective']:.2%} | {r['missed_defects']} | "
                  f"{r['false_alarms']} | {r['model_size_mb']} | {r['latency_ms']} |")
    for tf in sorted(RESULTS_DIR.glob("*/tflite.json")):
        t = json.loads(tf.read_text())
        md.append(f"\nTFLite ({t['experiment']}): {t['keras_size_mb']} MB -> {t['tflite_size_mb']} MB, "
                  f"same decision as Keras on {t['same_decision_rate']:.1%} of test images, "
                  f"{t['tflite_latency_ms']} ms/image.")
    (RESULTS_DIR / "comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # Chart: defect recall per training set, Keras vs PyTorch (grouped bars, one axis).
    exps = [e for e in ORDER if any(r["experiment"] == e for r in rows)] or sorted({r["experiment"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, title in ((axes[0], "recall_defective", "Defect recall (share of bad parts caught)"),
                           (axes[1], "accuracy", "Accuracy")):
        x = np.arange(len(exps))
        for k, (fw, color) in enumerate((("keras", BLUE), ("pytorch", ORANGE))):
            vals = [next((r[key] for r in rows if r["experiment"] == e and r["framework"] == fw), np.nan)
                    for e in exps]
            bars = ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=color,
                          label="TensorFlow/Keras" if fw == "keras" else "PyTorch")
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x, [LABELS.get(e, e) for e in exps], fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper right", ncol=2)
    fig.text(0.01, 0.01, "All models tested on the same real test photos.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(RESULTS_DIR / "comparison.png", dpi=140)
    print("Saved results/comparison.csv, .md, .png")


if __name__ == "__main__":
    main()
