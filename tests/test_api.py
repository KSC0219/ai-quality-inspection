"""End to end: train tiny Keras + PyTorch models (separate processes), export
TFLite, then call the REST API for every model type."""
import importlib
import os
import subprocess
import sys

import pytest

from conftest import ROOT


@pytest.fixture(scope="module")
def trained(tiny_dataset, tmp_path_factory):
    train, test = tiny_dataset
    out = tmp_path_factory.mktemp("out")
    env = dict(os.environ, QI_MODELS_DIR=str(out / "models"), QI_RESULTS_DIR=str(out / "results"))
    common = ["--train-dir", str(train), "--test-dir", str(test), "--experiment", "real",
              "--weights", "none", "--img-size", "64", "--epochs-head", "1", "--epochs-finetune", "1",
              "--batch-size", "8", "--workers", "0"]
    for fw in ("keras", "pytorch"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--framework", fw, *common],
                       check=True, env=env, capture_output=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "export_tflite.py"), "--test-dir", str(test)],
                   check=True, env=env, capture_output=True)
    os.environ.update(env)
    return out, test


def test_reports_written(trained):
    out, _ = trained
    for fw in ("keras", "pytorch"):
        d = out / "results" / "real" / fw
        for name in ("metrics.json", "history.png", "confusion_matrix.png", "gradcam_examples.png"):
            assert (d / name).exists(), name
    assert (out / "results" / "real" / "tflite.json").exists()


def test_api_all_models(trained):
    out, test = trained
    import qi.config
    importlib.reload(qi.config)
    import qi.inference
    importlib.reload(qi.inference)
    sys.path.insert(0, str(ROOT))
    import api.main
    importlib.reload(api.main)
    from fastapi.testclient import TestClient
    client = TestClient(api.main.app)
    assert client.get("/health").json()["models"] == ["keras", "pytorch", "tflite"]
    img = sorted((test / "def_front").glob("*.jpeg"))[0]
    for fw in ("keras", "pytorch", "tflite"):
        r = client.post(f"/inspect?framework={fw}&measure=true",
                        files={"file": ("p.jpeg", img.read_bytes(), "image/jpeg")})
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["prediction"] in ("OK", "DEFECTIVE") and 0 <= j["probability_defective"] <= 1
        assert ("gradcam_png_base64" in j) == (fw != "tflite")
        assert j["measurement"]["status"] in ("PASS", "FAIL", "UNCERTAIN", "NOT_FOUND")
    assert client.post("/inspect", files={"file": ("x.txt", b"nope", "text/plain")}).status_code == 400
    api.main.inspector.close()
