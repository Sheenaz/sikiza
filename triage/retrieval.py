"""Retrieve a counsellor-approved coping strategy. Never generates text.

Ranking within the predicted category uses char n-gram TF-IDF similarity between
the (pseudonymised) message and each strategy's `when` keywords + text. Ties
break deterministically, so the same input always gets the same, auditable answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import CATEGORIES, load_config, resolve
from .text import normalise


@dataclass
class Retrieved:
    id: str
    text: str
    language: str
    category: str
    approved: bool


class CopingLibrary:
    def __init__(self, path: str | Path | None = None, allow_draft: bool | None = None):
        cfg = load_config()["retrieval"]
        spec = yaml.safe_load(Path(resolve(path or cfg["library"])).read_text())
        self.version = str(spec["version"])
        self.allow_draft = cfg["allow_draft_entries"] if allow_draft is None else allow_draft
        self.crisis = spec["crisis_response"]
        all_entries = spec["strategies"]
        ids = [e["id"] for e in all_entries]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate strategy ids in library")
        self.entries = [e for e in all_entries if e.get("approved") or self.allow_draft]
        missing = [c for c in CATEGORIES if not any(e["category"] == c for e in self.entries)]
        if missing:
            raise RuntimeError(
                f"No usable (approved) strategies for categories {missing}. Approve entries in "
                f"the library or set retrieval.allow_draft_entries: true for development."
            )
        if not (self.crisis.get("approved") or self.allow_draft):
            raise RuntimeError("crisis_response is not approved")
        self._vec = TfidfVectorizer(preprocessor=normalise, analyzer="char_wb", ngram_range=(3, 5))
        docs = [f"{e['when']} {e['en']} {e['sw']}" for e in self.entries]
        self._m = self._vec.fit_transform(docs)

    @property
    def is_draft(self) -> bool:
        return not all(e.get("approved") for e in self.entries) or not self.crisis.get("approved")

    def retrieve(self, message: str, category: str, language: str = "en") -> Retrieved:
        lang = "sw" if language == "sw" else "en"
        idx = [i for i, e in enumerate(self.entries) if e["category"] == category]
        if not idx:
            idx = [i for i, e in enumerate(self.entries) if e["category"] == "other"]
        q = self._vec.transform([message])
        sims = (self._m[idx] @ q.T).toarray().ravel()
        best = idx[int(np.argmax(sims))]  # argmax picks first on ties -> deterministic
        e = self.entries[best]
        return Retrieved(e["id"], e[lang].strip(), lang, e["category"], bool(e.get("approved")))

    def crisis_response(self, language: str = "en") -> tuple[Retrieved, list[dict]]:
        lang = "sw" if language == "sw" else "en"
        c = self.crisis
        # never send a placeholder (e.g. "TODO") as a dialable number
        hotlines = [h for h in c.get("hotlines", [])
                    if sum(ch.isdigit() for ch in str(h.get("number", ""))) >= 3]
        return Retrieved(c["id"], c[lang].strip(), lang, "crisis", bool(c.get("approved"))), hotlines


@lru_cache(maxsize=1)
def default_library() -> CopingLibrary:
    return CopingLibrary()
