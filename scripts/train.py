"""Train one model on one dataset.

Examples
    python scripts/train.py --framework keras   --train-dir data/casting/train --test-dir data/casting/test
    python scripts/train.py --framework pytorch --train-dir data/synthetic --test-dir data/casting/test --experiment cad_only
"""
import argparse

import _bootstrap  # noqa: F401

from qi.config import add_train_args, config_from_args
from qi.runner import run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--framework", choices=["keras", "pytorch"], required=True)
    ap.add_argument("--train-dir", required=True, help="folder with ok_front/ and def_front/")
    ap.add_argument("--test-dir", required=True, help="folder with ok_front/ and def_front/")
    ap.add_argument("--experiment", default="real")
    add_train_args(ap)
    args = ap.parse_args()
    run(args.framework, args.train_dir, args.test_dir, args.experiment, config_from_args(args))


if __name__ == "__main__":
    main()
