"""Generate synthetic training images from the CAD model.

    python scripts/generate_cad_data.py                       # default: 300+300 parts x 4 views = 2400 images
    python scripts/generate_cad_data.py --ok-parts 50 --def-parts 50 --views 2   # quick test
    python scripts/generate_cad_data.py --defects crack blowhole                  # only some defect types

About 1.5 s per part per CPU core. On Colab (2 cores) the default takes ~8 minutes.
"""
import argparse

import _bootstrap  # noqa: F401

from qi.cad.generate import generate_dataset
from qi.cad.part import ALL_DEFECTS, VISUAL_DEFECTS
from qi.config import DATA_DIR


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DATA_DIR / "synthetic"))
    ap.add_argument("--ok-parts", type=int, default=300)
    ap.add_argument("--def-parts", type=int, default=300)
    ap.add_argument("--views", type=int, default=4, help="photos per physical part")
    ap.add_argument("--defects", nargs="+", default=VISUAL_DEFECTS, choices=ALL_DEFECTS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    generate_dataset(a.out, a.ok_parts, a.def_parts, a.views, a.defects, a.seed, a.workers)


if __name__ == "__main__":
    main()
