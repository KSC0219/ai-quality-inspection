"""TensorFlow + Keras model: MobileNetV2 transfer learning.

Architecture
    image (0-255) -> augmentation (training only) -> rescale to [-1, 1]
    -> MobileNetV2 backbone (ImageNet weights, 2.2M params, built for phones)
    -> GlobalAveragePooling -> Dropout(0.3) -> Dense(1, sigmoid) = P(defective)

Two-phase training (standard transfer-learning recipe)
    Phase 1: backbone frozen, only the new head learns (fast, stable).
    Phase 2: unfreeze the top 40 backbone layers with a 10x smaller learning
             rate (fine-tuning). BatchNorm layers stay frozen so their ImageNet
             statistics are not destroyed by small batches.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import keras  # noqa: E402
from keras import layers  # noqa: E402

from .config import TrainConfig  # noqa: E402
from .data import Split  # noqa: E402

FRAMEWORK = "keras"
FINE_TUNE_LAYERS = 40
AUTOTUNE = tf.data.AUTOTUNE


# ------------------------------------------------------------------ data
def _load(path, label, size):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (size, size))
    return tf.cast(img, tf.float32), label


def make_dataset(split: Split, cfg: TrainConfig, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((split.paths, np.array(split.labels, np.float32)))
    if training:
        ds = ds.shuffle(len(split), seed=cfg.seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: _load(p, y, cfg.img_size), num_parallel_calls=AUTOTUNE)
    return ds.batch(cfg.batch_size).prefetch(AUTOTUNE)


# ------------------------------------------------------------------ model
def build_model(cfg: TrainConfig) -> keras.Model:
    s = cfg.img_size
    inputs = keras.Input((s, s, 3), name="image")
    augment = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.5, fill_mode="reflect"),   # impeller looks the same at any angle
        layers.RandomZoom(0.1, fill_mode="reflect"),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.15, value_range=(0, 255)),
    ], name="augment")
    x = augment(inputs)
    x = layers.Rescaling(1 / 127.5, offset=-1, name="preprocess")(x)
    pretrained = cfg.weights != "none"
    base = keras.applications.MobileNetV2(input_shape=(s, s, 3), include_top=False,
                                          weights="imagenet" if pretrained else None)
    # Pretrained: freeze the backbone, BatchNorm always in inference mode.
    # From scratch (--weights none): random features are useless frozen, so train everything.
    base.trainable = not pretrained
    if not pretrained:
        # Keras BatchNorm moving statistics update slowly (momentum 0.99); from
        # scratch on a small dataset they never settle, so validation breaks.
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.momentum = 0.9
    x = base(x, training=False if pretrained else None)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="defect_prob")(x)
    return keras.Model(inputs, outputs, name="keras_mobilenetv2_defect")


def _backbone(model: keras.Model) -> keras.Model:
    return next(l for l in model.layers if l.name.startswith("mobilenetv2"))


def _compile(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.Recall(name="recall"), keras.metrics.AUC(name="auc")])


def train(train_split: Split, val_split: Split, cfg: TrainConfig, log=print) -> tuple[keras.Model, dict]:
    keras.utils.set_random_seed(cfg.seed)
    tr, va = make_dataset(train_split, cfg, True), make_dataset(val_split, cfg, False)
    c = train_split.counts()
    # class weights: rarer class counts more, so the model can't win by guessing the majority
    total = c["ok"] + c["defective"]
    class_weight = {0: total / (2 * max(c["ok"], 1)), 1: total / (2 * max(c["defective"], 1))}
    model = build_model(cfg)
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience,
                                               restore_best_weights=True)]

    pretrained = cfg.weights != "none"
    log(f"[keras] phase 1: {'head only' if pretrained else 'whole network from scratch'}, {cfg.epochs_head} epochs")
    _compile(model, cfg.lr_head)
    h1 = model.fit(tr, validation_data=va, epochs=cfg.epochs_head, class_weight=class_weight,
                   callbacks=callbacks, verbose=2).history

    log(f"[keras] phase 2: fine-tune top {FINE_TUNE_LAYERS} backbone layers, {cfg.epochs_finetune} epochs")
    if pretrained:
        base = _backbone(model)
        base.trainable = True
        for layer in base.layers[:-FINE_TUNE_LAYERS]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
    _compile(model, cfg.lr_finetune)
    h2 = model.fit(tr, validation_data=va, epochs=cfg.epochs_finetune, class_weight=class_weight,
                   callbacks=callbacks, verbose=2).history if cfg.epochs_finetune > 0 else {}

    history = {k: [float(v) for v in h1.get(k, [])] + [float(v) for v in h2.get(k, [])]
               for k in ("loss", "accuracy", "val_loss", "val_accuracy")}
    history["phase2_start"] = len(h1["loss"])
    return model, history


def predict_probs(model: keras.Model, split: Split, cfg: TrainConfig) -> np.ndarray:
    return model.predict(make_dataset(split, cfg, False), verbose=0).ravel()


def preprocess_image(img_rgb: np.ndarray, size: int) -> np.ndarray:
    return tf.image.resize(img_rgb.astype(np.float32), (size, size)).numpy()


def predict_one(model, img_rgb: np.ndarray, size: int) -> float:
    return float(model(preprocess_image(img_rgb, size)[None], training=False).numpy().ravel()[0])


# ------------------------------------------------------------------ Grad-CAM
def gradcam(model: keras.Model, img_rgb: np.ndarray, size: int) -> np.ndarray:
    """Which regions pushed the 'defective' score up? Returns heatmap 0..1.

    Grad-CAM: weight each feature map of the last conv block by the gradient
    of the output w.r.t. that map (averaged), sum, keep positive values.
    """
    x = tf.convert_to_tensor(preprocess_image(img_rgb, size)[None])
    x = model.get_layer("preprocess")(x)
    base = _backbone(model)
    with tf.GradientTape() as tape:
        feats = base(x, training=False)
        tape.watch(feats)
        y = model.get_layer("defect_prob")(model.get_layer("gap")(feats))
    grads = tape.gradient(y, feats)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(feats * weights[:, None, None, :], axis=-1))[0].numpy()
    return cam / (cam.max() + 1e-8)


# ------------------------------------------------------------------ save / stats
def save(model: keras.Model, out_base: Path) -> list[Path]:
    path = out_base.with_name(out_base.name + ".keras")
    model.save(path)
    return [path]


def load(path: str | Path) -> keras.Model:
    return keras.models.load_model(path)


def count_params(model) -> int:
    return int(model.count_params())


def latency_ms(model, img_rgb: np.ndarray, size: int, runs: int = 30) -> float:
    x = preprocess_image(img_rgb, size)[None]
    model(x, training=False)  # warm-up
    t = time.perf_counter()
    for _ in range(runs):
        model(x, training=False)
    return (time.perf_counter() - t) / runs * 1000


# ------------------------------------------------------------------ TFLite
def inference_model(model: keras.Model) -> keras.Model:
    """Same trained weights, minus the augmentation block. Random-augmentation
    ops can't be converted to TFLite and are useless at inference anyway."""
    s = model.input_shape[1]
    inp = keras.Input((s, s, 3), name="image")
    x = model.get_layer("preprocess")(inp)
    x = _backbone(model)(x, training=False)
    x = model.get_layer("gap")(x)
    out = model.get_layer("defect_prob")(x)      # dropout is a no-op at inference
    return keras.Model(inp, out, name="keras_defect_inference")


def export_tflite(model: keras.Model, out_path: str | Path, quantize: bool = True) -> Path:
    """Keras -> TensorFlow Lite for phones / Raspberry Pi / edge cameras.

    quantize=True applies dynamic-range quantization: weights stored as 8-bit
    integers instead of 32-bit floats, so the model is about 4x smaller."""
    out_path = Path(out_path)
    with tempfile.TemporaryDirectory() as d:
        inference_model(model).export(d, verbose=False)   # SavedModel without augmentation ops
        conv = tf.lite.TFLiteConverter.from_saved_model(d)
        if quantize:
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
        out_path.write_bytes(conv.convert())
    return out_path


class TFLitePredictor:
    def __init__(self, path: str | Path):
        try:  # LiteRT is the new name of the TFLite runtime (pip install ai-edge-litert)
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            Interpreter = tf.lite.Interpreter
        self.interp = Interpreter(model_path=str(path))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.size = int(self.inp["shape"][1])

    def predict_one(self, img_rgb: np.ndarray) -> float:
        x = preprocess_image(img_rgb, self.size)[None].astype(np.float32)
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        return float(self.interp.get_tensor(self.out["index"]).ravel()[0])
