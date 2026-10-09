import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    """A small CAD-generated dataset (train + test) used by the model tests."""
    from qi.cad.generate import generate_dataset
    base = tmp_path_factory.mktemp("data")
    train = generate_dataset(base / "train", 6, 6, 2, seed=1, workers=1, export_samples=False, progress=False)
    test = generate_dataset(base / "test", 3, 3, 2, seed=2, workers=1, export_samples=False, progress=False)
    return train, test
