"""Evaluation metrics shared by both frameworks.

In quality inspection the costly mistake is a MISSED DEFECT (false negative):
a bad part reaches the customer. A false alarm only costs a manual re-check.
So besides accuracy we report recall on defects, and we choose the decision
threshold on the validation set to hit a target recall (e.g. 98%).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def threshold_for_recall(y_true, probs, target_recall: float = 0.98) -> float:
    """Decision threshold for "defective" = min(0.5, t), where t is the highest
    threshold that still catches >= target_recall of validation defects.

    The threshold only ever moves DOWN from 0.5 (flag more parts as defective).
    Never above 0.5: if validation happens to be perfectly separated, t can be
    ~0.99, which would be over-fitted to the validation set and miss defects
    on new parts.
    """
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    pos = np.sort(probs[y_true == 1])
    if pos.size == 0:
        return 0.5
    # allow at most this many defects to be missed
    k = int(np.floor((1 - target_recall) * pos.size))
    thr = pos[k] if k < pos.size else pos[-1]
    return float(min(max(thr - 1e-6, 1e-6), 0.5))


def evaluate(y_true, probs, threshold: float = 0.5) -> dict:
    y_true, probs = np.asarray(y_true).astype(int), np.asarray(probs, dtype=float)
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "threshold": round(float(threshold), 4),
        "accuracy": accuracy_score(y_true, pred),
        "precision_defective": precision_score(y_true, pred, zero_division=0),
        "recall_defective": recall_score(y_true, pred, zero_division=0),
        "f1_defective": f1_score(y_true, pred, zero_division=0),
        "specificity_ok": tn / (tn + fp) if (tn + fp) else 0.0,
        "missed_defects": int(fn), "false_alarms": int(fp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n": int(len(y_true)),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, probs)
    except ValueError:
        out["roc_auc"] = float("nan")
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def full_report(y_val, p_val, y_test, p_test, target_recall: float) -> dict:
    thr = threshold_for_recall(y_val, p_val, target_recall)
    return {
        "chosen_threshold": round(thr, 4),
        "target_recall": target_recall,
        "val_at_0.5": evaluate(y_val, p_val, 0.5),
        "test_at_0.5": evaluate(y_test, p_test, 0.5),
        "test_at_chosen_threshold": evaluate(y_test, p_test, thr),
    }
