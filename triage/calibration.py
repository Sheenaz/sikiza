"""Temperature scaling (Guo et al., 2017) on a held-out calibration split.

Thresholds are only meaningful on true probabilities, so the models are trained
WITHOUT class weights (which shift the logits) and a single temperature per head
is fitted on the calib split. The binary crisis head optionally allows a bias
term (Platt scaling) for models where a prior shift is suspected.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, log_softmax, softmax


def _bin_nll(z: np.ndarray, y: np.ndarray) -> float:
    # numerically stable binary cross-entropy on logits
    return float(np.mean(np.logaddexp(0, z) - y * z))


@dataclass
class Calibrator:
    t_crisis: float = 1.0
    b_crisis: float = 0.0
    t_category: float = 1.0
    method: str = "temperature"

    # ---- fitting -------------------------------------------------------------
    @classmethod
    def fit(cls, crisis_logit: np.ndarray, y_crisis: np.ndarray, cat_logits: np.ndarray,
            y_cat: np.ndarray, method: str = "temperature") -> "Calibrator":
        crisis_logit = np.asarray(crisis_logit, float)
        y_crisis = np.asarray(y_crisis, float)
        if method == "platt":
            res = minimize(lambda p: _bin_nll(crisis_logit / np.exp(p[0]) + p[1], y_crisis), x0=[0.0, 0.0])
            t_c, b_c = float(np.exp(res.x[0])), float(res.x[1])
        else:
            res = minimize_scalar(lambda lt: _bin_nll(crisis_logit / np.exp(lt), y_crisis),
                                  bounds=(-3, 3), method="bounded")
            t_c, b_c = float(np.exp(res.x)), 0.0
        cat_logits = np.asarray(cat_logits, float)
        y_cat = np.asarray(y_cat, int)

        def cat_nll(lt):
            lp = log_softmax(cat_logits / np.exp(lt), axis=1)
            return float(-lp[np.arange(len(y_cat)), y_cat].mean())

        t_k = float(np.exp(minimize_scalar(cat_nll, bounds=(-3, 3), method="bounded").x))
        return cls(t_c, b_c, t_k, method)

    # ---- applying ------------------------------------------------------------
    def crisis_proba(self, crisis_logit: np.ndarray) -> np.ndarray:
        return expit(np.asarray(crisis_logit, float) / self.t_crisis + self.b_crisis)

    def category_proba(self, cat_logits: np.ndarray) -> np.ndarray:
        return softmax(np.asarray(cat_logits, float) / self.t_category, axis=1)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(**d)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)
