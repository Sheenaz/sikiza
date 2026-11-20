"""Keyword safety valve: ANY crisis-tier hit escalates, regardless of the model.

The final decision is an OR-gate (proposal §3):
    crisis = valve.crisis_hit  OR  p_crisis >= threshold
An AND-gate would let the classifier veto the keywords and miss crises.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from .config import load_config, resolve
from .text import normalise


@dataclass
class ValveResult:
    crisis_hit: bool
    crisis_rules: list[str] = field(default_factory=list)
    distress_rules: list[str] = field(default_factory=list)


class SafetyValve:
    def __init__(self, lexicon_path: str | Path | None = None):
        path = resolve(lexicon_path or load_config()["safety"]["lexicon"])
        spec = yaml.safe_load(Path(path).read_text())
        self.version: str = str(spec["version"])
        self.reviewed: bool = bool(spec.get("reviewed_by"))
        self.rules: list[tuple[str, str, re.Pattern]] = []
        for r in spec["rules"]:
            self.rules.append((r["id"], r["tier"], re.compile(r["pattern"])))
        ids = [r[0] for r in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate rule ids in lexicon")

    def check(self, text: str) -> ValveResult:
        t = normalise(text)
        crisis, distress = [], []
        for rid, tier, pat in self.rules:
            if pat.search(t):
                (crisis if tier == "crisis" else distress).append(rid)
        return ValveResult(bool(crisis), crisis, distress)

    def export_for_mobile(self) -> dict:
        """JSON-able rules for the on-device offline valve (same regexes, JS-compatible)."""
        return {
            "version": self.version,
            "rules": [{"id": rid, "tier": tier, "pattern": pat.pattern} for rid, tier, pat in self.rules],
        }


@lru_cache(maxsize=1)
def default_valve() -> SafetyValve:
    return SafetyValve()
