"""Evaluation exactly as promised in the proposal's success criteria:

* crisis recall with a bootstrap 95% CI (target >= 0.95)
* macro-F1 over the five categories + confusion matrix
* everything split by English-only vs code-mixed
* McNemar's exact test vs the baseline
* Cohen's kappa on the double-coded sample
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from .config import CATEGORIES


def _recall(y, d):
    y, d = np.asarray(y), np.asarray(d)
    return float(d[y == 1].mean()) if (y == 1).any() else float("nan")


def _precision(y, d):
    y, d = np.asarray(y), np.asarray(d)
    return float(y[d == 1].mean()) if (d == 1).any() else float("nan")


def bootstrap_ci(fn, *arrays, n: int = 2000, ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap over rows. Resamples without a valid metric are skipped."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    m = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        v = fn(*[a[idx] for a in arrays])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    a = (1 - ci) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def crisis_metrics(y, decision, n_boot: int = 2000, ci: float = 0.95) -> dict:
    y, d = np.asarray(y, int), np.asarray(decision, int)
    lo, hi = bootstrap_ci(_recall, y, d, n=n_boot, ci=ci)
    tp = int(((y == 1) & (d == 1)).sum())
    fn = int(((y == 1) & (d == 0)).sum())
    fp = int(((y == 0) & (d == 1)).sum())
    tn = int(((y == 0) & (d == 0)).sum())
    # Exact Clopper-Pearson interval: the percentile bootstrap degenerates to [1, 1]
    # when every crisis is caught, which overstates certainty on small test sets.
    if tp + fn:
        e = binomtest(tp, tp + fn).proportion_ci(confidence_level=ci, method="exact")
        exact = [float(e.low), float(e.high)]
    else:
        exact = [float("nan"), float("nan")]
    conservative_lo = np.nanmin([lo, exact[0]]) if not (np.isnan(lo) and np.isnan(exact[0])) else float("nan")
    esc = float(d.mean()) if len(d) else float("nan")
    return {
        "n": int(len(y)), "positives": int(y.sum()),
        "recall": _recall(y, d), "recall_ci": [lo, hi], "recall_exact_ci": exact,
        "precision": _precision(y, d),
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) else float("nan"),
        "escalation_rate": esc,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "meets_target_0.95": bool(conservative_lo >= 0.95) if not np.isnan(conservative_lo) else False,
        "warning": ("escalates more than half of all messages — recall is not informative"
                    if esc > 0.5 else None),
    }


def category_metrics(y_true: list[str], y_pred: list[str], n_boot: int = 2000, ci: float = 0.95) -> dict:
    labels = CATEGORIES
    yt, yp = np.asarray(y_true), np.asarray(y_pred)

    def mf1(a, b):
        return f1_score(a, b, labels=labels, average="macro", zero_division=0)

    lo, hi = bootstrap_ci(mf1, yt, yp, n=n_boot, ci=ci)
    return {
        "macro_f1": float(mf1(yt, yp)), "macro_f1_ci": [lo, hi],
        "per_class_f1": dict(zip(labels, f1_score(yt, yp, labels=labels, average=None, zero_division=0).tolist())),
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(yt, yp, labels=labels).tolist()},
    }


def mcnemar(correct_a, correct_b) -> dict:
    """Exact McNemar on paired correctness. b = A right / B wrong, c = A wrong / B right."""
    a, b_ = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return {"a_only_correct": b, "b_only_correct": c, "p_value": float(p)}


def kappa(df) -> dict | None:
    out = {}
    if {"category", "category_2"}.issubset(df.columns):
        s = df.dropna(subset=["category_2"])
        if len(s):
            out["category_kappa"] = float(cohen_kappa_score(s["category"], s["category_2"]))
            out["category_n"] = int(len(s))
    if {"crisis", "crisis_2"}.issubset(df.columns):
        s = df.dropna(subset=["crisis_2"])
        if len(s):
            out["crisis_kappa"] = float(cohen_kappa_score(s["crisis"].astype(int), s["crisis_2"].astype(int)))
            out["crisis_n"] = int(len(s))
    return out or None


def full_report(df, decision, cat_pred, n_boot: int = 2000, ci: float = 0.95) -> dict:
    """df needs crisis, category, group columns aligned with decision / cat_pred."""
    decision, cat_pred = np.asarray(decision, int), np.asarray(cat_pred)
    rep = {"overall": {"crisis": crisis_metrics(df["crisis"], decision, n_boot, ci),
                       "category": category_metrics(df["category"], cat_pred, n_boot, ci)}}
    for g in ["english_only", "code_mixed"]:
        m = (df["group"] == g).to_numpy()
        if m.sum():
            rep[g] = {"crisis": crisis_metrics(df["crisis"].to_numpy()[m], decision[m], n_boot, ci),
                      "category": category_metrics(df["category"].to_numpy()[m], cat_pred[m], n_boot, ci)}
    return rep
