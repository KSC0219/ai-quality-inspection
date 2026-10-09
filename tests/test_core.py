"""Metrics, threshold choice and data splitting (no deep-learning framework needed)."""
import numpy as np

from qi.data import Split, list_images, merge, subsample, train_val_split
from qi.metrics import evaluate, full_report, threshold_for_recall


def test_threshold_never_above_half():
    y = [0, 0, 1, 1]
    p = [0.01, 0.02, 0.98, 0.99]           # perfectly separated
    assert threshold_for_recall(y, p, 0.98) <= 0.5


def test_threshold_lowers_to_catch_defects():
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.linspace(0, 0.3, 50), np.linspace(0.2, 0.9, 50)])
    thr = threshold_for_recall(y, p, 0.98)
    assert thr < 0.5
    assert evaluate(y, p, thr)["recall_defective"] >= 0.98


def test_evaluate_confusion_matrix():
    m = evaluate([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.3], 0.5)
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert m["missed_defects"] == 1 and m["false_alarms"] == 1
    assert m["accuracy"] == 0.5


def test_full_report_keys():
    r = full_report([0, 1, 0, 1], [0.2, 0.8, 0.1, 0.9], [0, 1], [0.3, 0.7], 0.98)
    assert {"chosen_threshold", "test_at_0.5", "test_at_chosen_threshold"} <= set(r)


def test_list_images_labels(tiny_dataset):
    train, _ = tiny_dataset
    s = list_images(train)
    assert s.counts() == {"ok": 12, "defective": 12}
    assert all(("ok_front" in p) == (l == 0) for p, l in zip(s.paths, s.labels))


def test_split_keeps_views_of_one_part_together(tiny_dataset):
    """No leakage: two photos of the same CAD part never end up in train AND val."""
    train, _ = tiny_dataset
    tr, va = train_val_split(list_images(train), 0.3, seed=0)
    part = lambda p: p.rsplit("/", 1)[-1].rsplit("_v", 1)[0]  # noqa: E731
    assert not ({part(p) for p in tr.paths} & {part(p) for p in va.paths})
    assert len(tr) + len(va) == 24


def test_subsample_and_merge():
    s = Split([f"ok{i}" for i in range(10)] + [f"d{i}" for i in range(10)], [0] * 10 + [1] * 10)
    sub = subsample(s, 0.2, seed=1)
    assert sub.counts() == {"ok": 2, "defective": 2}
    assert len(merge(s, sub)) == 24
