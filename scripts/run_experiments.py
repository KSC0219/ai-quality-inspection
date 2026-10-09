"""Run the full experiment grid: 3 training sets x 2 frameworks, all tested
on the SAME real test photos.

    real          train on real photos only                (baseline)
    cad_only      train on CAD-rendered images only         (sim-to-real: zero real photos)
    cad_plus_10   CAD images + only 10% of the real photos  (realistic: few real defect photos)

    python scripts/run_experiments.py                         # everything (~1 h on a Colab GPU)
    python scripts/run_experiments.py --experiments real --frameworks keras --epochs-finetune 3
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qi.config import DATA_DIR, RESULTS_DIR, ROOT, add_train_args, config_from_args
from qi.data import find_real_dataset, list_images, materialize, merge, subsample

EXPERIMENTS = ["real", "cad_only", "cad_plus_10"]


def build_train_dir(name: str, real_root, synthetic_dir, seed: int):
    if name == "real":
        return real_root / "train"
    real_train = list_images(real_root / "train")
    synth = list_images(synthetic_dir)
    if name == "cad_only":
        return synthetic_dir
    if name == "cad_plus_10":
        return materialize(merge(synth, subsample(real_train, 0.10, seed)), DATA_DIR / "experiments" / name)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", default=EXPERIMENTS, choices=EXPERIMENTS)
    ap.add_argument("--frameworks", nargs="+", default=["keras", "pytorch"], choices=["keras", "pytorch"])
    ap.add_argument("--real-root", default=None, help="folder with train/ and test/ (auto-detected)")
    ap.add_argument("--synthetic-dir", default=str(DATA_DIR / "synthetic"))
    add_train_args(ap)
    a = ap.parse_args()
    cfg = config_from_args(a)
    real_root = find_real_dataset(a.real_root or DATA_DIR)
    synthetic_dir = Path(a.synthetic_dir)
    if any(e != "real" for e in a.experiments) and not (synthetic_dir / "def_front").is_dir():
        raise SystemExit(f"No synthetic data in {synthetic_dir}. Run scripts/generate_cad_data.py first.")

    summary = []
    for exp in a.experiments:
        train_dir = build_train_dir(exp, real_root, synthetic_dir, cfg.seed)
        for fw in a.frameworks:
            # Each run in its own process: TensorFlow and PyTorch loaded into the
            # same Python process can crash (clashing native libraries).
            cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--framework", fw,
                   "--train-dir", str(train_dir), "--test-dir", str(real_root / "test"), "--experiment", exp]
            for k, v in cfg.to_dict().items():
                if k != "extra":
                    cmd += ["--" + k.replace("_", "-"), str(v)]
            subprocess.run(cmd, check=True)
            r = json.loads((RESULTS_DIR / exp / fw / "metrics.json").read_text())
            t = r["test_at_chosen_threshold"]
            summary.append({"experiment": exp, "framework": fw, "accuracy": t["accuracy"],
                            "recall_defective": t["recall_defective"], "missed_defects": t["missed_defects"],
                            "false_alarms": t["false_alarms"]})
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "experiments_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Next: python scripts/export_tflite.py && python scripts/compare.py")


if __name__ == "__main__":
    main()
