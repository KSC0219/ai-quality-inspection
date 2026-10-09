"""PyTorch model: build, train, predict, Grad-CAM, save/load."""
import numpy as np
import torch
from PIL import Image

from qi import torch_model as tm
from qi.config import TrainConfig
from qi.data import list_images, train_val_split

CFG = TrainConfig(img_size=64, batch_size=8, epochs_head=1, epochs_finetune=1, weights="none", workers=0)


def test_model_outputs_one_logit():
    m = tm.build_model(CFG).eval()
    with torch.no_grad():
        assert m(torch.randn(2, 3, 64, 64)).shape == (2, 1)


def test_train_predict_gradcam_save(tiny_dataset, tmp_path):
    train, test = tiny_dataset
    tr, va = train_val_split(list_images(train), 0.25, 0)
    te = list_images(test)
    model, history = tm.train(tr, va, CFG, log=lambda *_: None)
    assert len(history["loss"]) >= 2
    probs = tm.predict_probs(model, te, CFG)
    assert probs.shape == (len(te),) and ((probs >= 0) & (probs <= 1)).all()

    img = np.array(Image.open(te.paths[0]).convert("RGB"))
    heat = tm.gradcam(model, img, CFG.img_size)
    assert heat.ndim == 2 and 0 <= heat.min() and heat.max() <= 1

    path = tm.save(model, tmp_path / "m")[0]
    reloaded = tm.load(path)
    assert abs(tm.predict_one(model, img, 64) - tm.predict_one(reloaded, img, 64)) < 1e-5


def test_phase1_trains_only_the_head():
    m = tm.build_model(TrainConfig(weights="none"))
    params = tm._set_trainable(m, phase=1, scratch=False)
    assert sum(p.numel() for p in params) == sum(p.numel() for p in m.fc.parameters())
