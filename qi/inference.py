"""One interface to run any trained model (Keras, PyTorch or TFLite) on an image.

    from qi.inference import Inspector
    insp = Inspector("models/real")             # finds whatever models exist there
    result = insp.inspect(image, framework="keras", explain=True, measure=True)

Process isolation: each framework runs in its own worker process. TensorFlow
and PyTorch loaded into the same Python process can crash (their native
libraries clash), so the API / app process itself never imports either one.
"""
from __future__ import annotations

import base64
import io
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from PIL import Image

from .config import CLASS_NAMES, MODELS_DIR


def to_rgb(image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr[..., :3].astype(np.uint8)


def png_base64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _worker(framework: str, model_dir: str, size: int, conn) -> None:
    """Runs in a separate process: loads one framework + model, serves requests."""
    try:
        model_dir = Path(model_dir)
        if framework == "pytorch":
            from . import torch_model as mod
            model = mod.load(model_dir / "pytorch_model.pt")
        else:
            from . import keras_model as mod
            model = (mod.TFLitePredictor(model_dir / "keras_model.tflite") if framework == "tflite"
                     else mod.load(model_dir / "keras_model.keras"))
        conn.send(("ready", None))
    except Exception as e:  # noqa: BLE001
        conn.send(("error", f"{type(e).__name__}: {e}"))
        return
    while True:
        msg = conn.recv()
        if msg is None:
            break
        rgb, explain = msg
        try:
            prob = model.predict_one(rgb) if framework == "tflite" else mod.predict_one(model, rgb, size)
            heat = mod.gradcam(model, rgb, size) if explain and framework != "tflite" else None
            conn.send(("ok", (prob, heat)))
        except Exception as e:  # noqa: BLE001
            conn.send(("error", f"{type(e).__name__}: {e}"))


class _Worker:
    def __init__(self, framework: str, model_dir: Path, size: int):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(framework, str(model_dir), size, child), daemon=True)
        self.proc.start()
        status, err = self.conn.recv()
        if status != "ready":
            raise RuntimeError(f"Could not load {framework} model: {err}")

    def __call__(self, rgb: np.ndarray, explain: bool):
        self.conn.send((rgb, explain))
        status, payload = self.conn.recv()
        if status != "ok":
            raise RuntimeError(payload)
        return payload

    def close(self):
        try:
            self.conn.send(None)
        except (BrokenPipeError, OSError):
            pass
        self.proc.join(timeout=5)


class Inspector:
    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir) if model_dir else self._default_dir()
        self._workers: dict[str, _Worker] = {}
        self.meta: dict = {}
        for fw in ("keras", "pytorch"):
            m = self.model_dir / f"{fw}_meta.json"
            if m.exists():
                self.meta[fw] = json.loads(m.read_text())
        if (self.model_dir / "keras_model.tflite").exists() and "keras" in self.meta:
            self.meta["tflite"] = self.meta["keras"]

    @staticmethod
    def _default_dir() -> Path:
        for name in ("real", "cad_plus_10", "cad_only"):
            if (MODELS_DIR / name).is_dir():
                return MODELS_DIR / name
        dirs = sorted(p for p in MODELS_DIR.glob("*") if p.is_dir())
        return dirs[0] if dirs else MODELS_DIR / "real"

    @property
    def available(self) -> list[str]:
        return list(self.meta)

    def _get(self, fw: str) -> _Worker:
        if fw not in self._workers:
            self._workers[fw] = _Worker(fw, self.model_dir, self.meta[fw]["img_size"])
        return self._workers[fw]

    def inspect(self, image, framework: str | None = None, explain: bool = True,
                measure: bool = False) -> dict:
        if not self.available:
            raise FileNotFoundError(f"No trained models in {self.model_dir}. Train first (see README).")
        fw = framework or self.available[0]
        if fw not in self.meta:
            raise ValueError(f"Model '{fw}' not available. Available: {self.available}")
        rgb = to_rgb(image)
        meta = self.meta[fw]
        prob, heat = self._get(fw)(rgb, explain)
        thr = meta["threshold"]
        label = int(prob >= thr)
        out = {"framework": fw, "prediction": CLASS_NAMES[label], "is_defective": bool(label),
               "probability_defective": round(float(prob), 4), "threshold": thr,
               "experiment": meta.get("experiment")}
        if heat is not None:
            from .viz import overlay_heatmap
            out["heatmap"] = heat
            out["overlay"] = overlay_heatmap(rgb, heat)
        if measure:
            from .cad.measure import draw_measurement, measure_bore
            m = measure_bore(rgb)
            out["measurement"] = m
            out["measurement_overlay"] = draw_measurement(rgb, m)
        return out

    def close(self):
        for w in self._workers.values():
            w.close()
        self._workers.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
