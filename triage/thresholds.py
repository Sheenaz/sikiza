"""Cost-derived crisis threshold, tightened by a recall floor.

1. Bayes-optimal threshold on calibrated probabilities:
       escalate if p * C_fn > (1 - p) * C_fp   <=>   p > C_fp / (C_fp + C_fn)
   With C_fn = 100, C_fp = 1 this is ~0.0099, which is far below 0.5.
2. If, on the calibration split, that threshold still gives recall below the
   target (0.95), lower it to the largest threshold that meets the target.

The final threshold is min(1, 2). It is chosen on CALIB, never on TEST.
"""
from __future__ import annotations

import numpy as np


def bayes_threshold(cost_fn: float, cost_fp: float) -> float:
    return cost_fp / (cost_fp + cost_fn)


def recall_threshold(p: np.ndarray, y: np.ndarray, target: float) -> float:
    """Largest t such that recall(p >= t) >= target."""
    pos = np.sort(np.asarray(p, float)[np.asarray(y) == 1])
    if len(pos) == 0:
        return 0.0
    # Need at least ceil(target * n) positives at or above t.
    k = int(np.ceil(target * len(pos)))
    return float(pos[len(pos) - k])


def choose_threshold(p_calib: np.ndarray, y_calib: np.ndarray, cost_fn: float, cost_fp: float,
                     target_recall: float) -> dict:
    t_cost = bayes_threshold(cost_fn, cost_fp)
    t_rec = recall_threshold(p_calib, y_calib, target_recall)
    t = min(t_cost, t_rec)
    y = np.asarray(y_calib)
    pred = np.asarray(p_calib) >= t
    n_pos = int(y.sum())
    return {
        "threshold": float(t),
        "t_cost": float(t_cost),
        "t_recall_floor": float(t_rec),
        "binding": "recall_floor" if t_rec < t_cost else "cost",
        "calib_recall": float(pred[y == 1].mean()) if n_pos else float("nan"),
        "calib_escalation_rate": float(pred.mean()),
        "calib_positives": n_pos,
        "warning": (f"only {n_pos} crisis examples in calib: threshold is fragile"
                    if n_pos < 20 else None),
    }
