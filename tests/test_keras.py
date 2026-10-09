"""TensorFlow/Keras model: build, train one step, predict, Grad-CAM, TFLite."""
import numpy as np
from PIL import Image

from qi import keras_model as km
from qi.config import TrainConfig
from qi.data import list_images, train_val_split

CFG = TrainConfig(img_size=64, batch_size=8, epochs_head=1, epochs_finetune=1, weights="none", workers=0)


def _data(tiny_dataset):
    train, test = tiny_dataset
    tr, va = train_val_split(list_images(train), 0.25, 0)
    return tr, va, list_images(test)


def test_model_output_is_probability():
    m = km.build_model(CFG)
    out = m(np.random.rand(2, 64, 64, 3).astype("float32") * 255).numpy()
    assert out.shape == (2, 1) and ((out >= 0) & (out <= 1)).all()


def test_train_predict_gradcam_tflite(tiny_dataset, tmp_path):
    tr, va, te = _data(tiny_dataset)
    model, history = km.train(tr, va, CFG, log=lambda *_: None)
    assert len(history["loss"]) == 2 and history["phase2_start"] == 1
    probs = km.predict_probs(model, te, CFG)
    assert probs.shape == (len(te),) and np.isfinite(probs).all()

    img = np.array(Image.open(te.paths[0]).convert("RGB"))
    heat = km.gradcam(model, img, CFG.img_size)
    assert heat.ndim == 2 and 0 <= heat.min() and heat.max() <= 1

    path = km.save(model, tmp_path / "m")[0]
    reloaded = km.load(path)
    assert abs(km.predict_one(model, img, 64) - km.predict_one(reloaded, img, 64)) < 1e-5

    tfl = km.TFLitePredictor(km.export_tflite(model, tmp_path / "m.tflite"))
    assert abs(km.predict_one(model, img, 64) - tfl.predict_one(img)) < 0.02
    assert (tmp_path / "m.tflite").stat().st_size < path.stat().st_size
