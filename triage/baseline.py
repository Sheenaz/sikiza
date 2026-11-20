"""TF-IDF (word + char n-grams) + logistic regression: the floor every model must beat."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import CAT2ID, CATEGORIES, load_config
from .text import normalise


def _features(cfg: dict) -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=normalise, ngram_range=tuple(cfg["word_ngrams"]),
                                 max_features=cfg["max_features"], sublinear_tf=True, min_df=1)),
        ("char", TfidfVectorizer(preprocessor=normalise, analyzer="char_wb",
                                 ngram_range=tuple(cfg["char_ngrams"]),
                                 max_features=cfg["max_features"], sublinear_tf=True, min_df=1)),
    ])


class BaselineScorer:
    """Exposes the same interface as the transformer/ONNX scorers: raw logits."""

    kind = "tfidf_logreg"

    def __init__(self, crisis: Pipeline | None = None, category: Pipeline | None = None):
        self.crisis, self.category = crisis, category

    def fit(self, texts, y_crisis, y_category) -> "BaselineScorer":
        cfg = load_config()["baseline"]
        # No class_weight: keep logits calibratable; recall comes from the low threshold.
        self.crisis = Pipeline([("f", _features(cfg)),
                                ("lr", LogisticRegression(C=cfg["C"], max_iter=3000))])
        self.category = Pipeline([("f", _features(cfg)),
                                  ("lr", LogisticRegression(C=cfg["C"], max_iter=3000))])
        self.crisis.fit(list(texts), np.asarray(y_crisis, int))
        y_cat = np.array([CAT2ID[c] for c in y_category])
        self.category.fit(list(texts), y_cat)
        return self

    def logits(self, texts) -> tuple[np.ndarray, np.ndarray]:
        texts = list(texts)
        z = self.crisis.decision_function(texts)
        # full K-column log-probas even if a class was absent in training
        lp = np.full((len(texts), len(CATEGORIES)), -1e4)
        lp[:, self.category.classes_] = self.category.predict_log_proba(texts)
        return z, lp

    def save(self, d: str | Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump({"crisis": self.crisis, "category": self.category}, d / "baseline.joblib")

    @classmethod
    def load(cls, d: str | Path) -> "BaselineScorer":
        m = joblib.load(Path(d) / "baseline.joblib")
        return cls(m["crisis"], m["category"])

    def top_features(self, k: int = 25) -> list[tuple[str, float]]:
        """Explainability: strongest crisis n-grams (positive weights)."""
        f, lr = self.crisis.named_steps["f"], self.crisis.named_steps["lr"]
        names = f.get_feature_names_out()
        w = lr.coef_[0]
        idx = np.argsort(w)[::-1][:k]
        return [(names[i], float(w[i])) for i in idx]
