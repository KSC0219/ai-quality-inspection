"""Download the real dataset from Kaggle into data/casting/.

Dataset: "Casting product image data for quality inspection"
         https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
         Top-view photos of submersible-pump impellers, 300x300 grayscale,
         labelled ok_front / def_front, with a separate test folder.
         (The training log prints the exact image counts per class.)

One-time setup (either option works):
  A. New API token: kaggle.com/settings/api -> "Generate New Token" -> copy the token.
     Save it in ~/.kaggle/access_token  (Windows: C:\\Users\\<you>\\.kaggle\\access_token)
     or set the environment variable KAGGLE_API_TOKEN.
  B. Legacy key: same page -> "Create Legacy API Key" -> kaggle.json downloads.
     Save it as ~/.kaggle/kaggle.json   (Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json)
  On Colab the notebook asks for the token or the file.
  Then: python scripts/download_data.py
Manual alternative: download the zip from the page above and unzip it into data/casting/.
"""
import subprocess
import sys

import _bootstrap  # noqa: F401

from qi.config import DATA_DIR
from qi.data import KAGGLE_SLUG, find_real_dataset


def main():
    out = DATA_DIR / "casting"
    try:
        print("Already downloaded:", find_real_dataset(out))
        return
    except FileNotFoundError:
        pass
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", KAGGLE_SLUG, "-p", str(out), "--unzip"]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nDownload failed. Check: 'pip install -U kaggle' is done, your token / kaggle.json is in place,"
              "\nand you opened the dataset page on kaggle.com once while logged in.")
        print(__doc__)
        sys.exit(1)
    print("Dataset ready at:", find_real_dataset(out))


if __name__ == "__main__":
    main()
