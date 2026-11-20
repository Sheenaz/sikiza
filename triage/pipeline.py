"""End-to-end triage, matching the serving sketch:

    pseudonymise -> safety valve -> classifier (calibrated) -> crisis?  (OR-gate)
        yes -> human escalation + crisis response
        no  -> retrieve approved strategy for the predicted category
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field

import numpy as np

from .calibration import Calibrator
from .config import CATEGORIES
from .pseudonymise import pseudonymise
from .retrieval import CopingLibrary
from .safety_valve import SafetyValve
from .text import language_group


@dataclass
class TriageResult:
    request_id: str
    crisis: bool
    crisis_source: list[str]            # "valve", "classifier", "model_error"
    crisis_prob: float | None
    threshold: float
    category: str
    category_prob: float | None
    valve_rules: list[str]
    distress_rules: list[str]
    response_id: str
    response_text: str
    response_language: str
    hotlines: list[dict] = field(default_factory=list)
    degraded: bool = False
    latency_ms: float = 0.0
    pseudonymised_text: str = ""        # NEVER logged; only stored in the escalation queue

    def public(self) -> dict:
        """What the student app receives."""
        return {
            "request_id": self.request_id,
            "crisis": self.crisis,
            "category": None if self.crisis else self.category,
            "response_id": self.response_id,
            "response_text": self.response_text,
            "response_language": self.response_language,
            "hotlines": self.hotlines,
        }

    def to_dict(self) -> dict:
        return asdict(self)


def decide(valve_hit: bool, p_crisis: float | None, threshold: float) -> tuple[bool, list[str]]:
    src = []
    if valve_hit:
        src.append("valve")
    if p_crisis is not None and p_crisis >= threshold:
        src.append("classifier")
    return bool(src), src


class TriagePipeline:
    def __init__(self, scorer, calibrator: Calibrator, manifest: dict, valve: SafetyValve,
                 library: CopingLibrary, escalate_on_model_error: bool = True):
        self.scorer, self.cal, self.manifest = scorer, calibrator, manifest
        self.valve, self.library = valve, library
        self.threshold = float(manifest["threshold"]["threshold"])
        self.escalate_on_model_error = escalate_on_model_error

    @property
    def versions(self) -> dict:
        return {
            "model": self.manifest.get("model_version"),
            "model_type": self.manifest.get("model_type"),
            "threshold": self.threshold,
            "lexicon": self.valve.version,
            "library": self.library.version,
        }

    def __call__(self, text: str, language_pref: str | None = None) -> TriageResult:
        t0 = time.perf_counter()
        clean = pseudonymise(text)
        valve = self.valve.check(clean)
        degraded = False
        p_crisis = p_cat = None
        cat = "other"
        try:
            z, k = self.scorer.logits([clean])
            p_crisis = float(self.cal.crisis_proba(z)[0])
            probs = self.cal.category_proba(k)[0]
            cat = CATEGORIES[int(np.argmax(probs))]
            p_cat = float(probs.max())
        except Exception:  # noqa: BLE001 — fail safe, never fail silent
            degraded = True
        crisis, src = decide(valve.crisis_hit, p_crisis, self.threshold)
        if degraded and self.escalate_on_model_error:
            crisis, src = True, src + ["model_error"]

        lang = language_pref or ("sw" if language_group(clean) == "code_mixed" else "en")
        if crisis:
            r, hotlines = self.library.crisis_response(lang)
        else:
            r, hotlines = self.library.retrieve(clean, cat, lang), []
        return TriageResult(
            request_id=uuid.uuid4().hex, crisis=crisis, crisis_source=src, crisis_prob=p_crisis,
            threshold=self.threshold, category=cat, category_prob=p_cat,
            valve_rules=valve.crisis_rules, distress_rules=valve.distress_rules,
            response_id=r.id, response_text=r.text, response_language=r.language,
            hotlines=hotlines, degraded=degraded,
            latency_ms=(time.perf_counter() - t0) * 1000, pseudonymised_text=clean,
        )


def batch_decisions(scorer, cal: Calibrator, threshold: float, valve: SafetyValve, texts) -> dict:
    """Vectorised version for evaluation. Returns classifier-only and combined decisions."""
    texts = [pseudonymise(t) for t in texts]
    z, k = scorer.logits(texts)
    p = cal.crisis_proba(z)
    cat = [CATEGORIES[i] for i in np.argmax(cal.category_proba(k), axis=1)]
    v = np.array([valve.check(t).crisis_hit for t in texts], int)
    clf = (p >= threshold).astype(int)
    return {"p_crisis": p, "valve": v, "classifier": clf, "combined": np.maximum(v, clf), "category": cat}
