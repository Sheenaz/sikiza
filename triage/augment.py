"""Training-volume augmentation: Sheng swaps, synonyms, optional back-translation.

Every variant keeps its seed's seed_id, so grouped splits keep it on the same side
as its original (see data.assemble). Variants are training data only.

Crisis-bearing words are protected: no swap is applied inside a span that matches a
crisis-tier safety-valve rule, so a crisis label is never attached to a sentence
whose crisis phrase was rewritten away.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .safety_valve import default_valve
from .text import normalise

# Bidirectional-ish swap lists (English / Swahili -> Sheng and back).
# STARTER LIST — to be extended with the student panel; synthetic Sheng may not match real usage.
SHENG_SWAPS: dict[str, list[str]] = {
    "money": ["doh", "ganji", "chapaa"],
    "pesa": ["doh", "ganji", "chapaa"],
    "friends": ["mbogi", "mabeshte"],
    "marafiki": ["mbogi", "mabeshte"],
    "friend": ["beshte", "msee wangu"],
    "house": ["keja"],
    "nyumba": ["keja"],
    "mother": ["mathe", "mum"],
    "mama": ["mathe"],
    "father": ["budaa", "fathe"],
    "baba": ["budaa", "fathe"],
    "good": ["fiti", "poa"],
    "nzuri": ["fiti", "poa"],
    "really": ["manze", "kabisa"],
    "girl": ["dem"],
    "man": ["msee"],
    "fare": ["fare", "nauli"],
    "school": ["shule", "campo"],
    "university": ["campo", "uni"],
    "food": ["dishi", "chakula"],
    "tired": ["nimechoka", "exhausted"],
    "worried": ["na wasiwasi", "stressed"],
    "very": ["sana", "mbaya"],
    "sana": ["very", "mbaya"],
}

SYNONYMS: dict[str, list[str]] = {
    "exam": ["test", "paper", "cat"],
    "exams": ["tests", "papers", "cats"],
    "sad": ["down", "low", "unhappy"],
    "scared": ["afraid", "anxious", "nervous"],
    "nervous": ["anxious", "scared", "tense"],
    "lonely": ["alone", "isolated"],
    "stressed": ["overwhelmed", "under pressure"],
    "broke": ["out of money", "sina doh"],
    "fees": ["school fees", "karo"],
    "lecturer": ["lec", "mwalimu"],
}

_WORD = re.compile(r"\w+|\W+")


def _protected_spans(text_norm: str) -> list[tuple[int, int]]:
    spans = []
    for _, tier, pat in default_valve().rules:
        if tier == "crisis":
            spans += [m.span() for m in pat.finditer(text_norm)]
    return spans


def swap_variant(text: str, rng: np.random.Generator, p: float = 0.5) -> str:
    t = normalise(text)
    spans = _protected_spans(t)
    out, pos = [], 0
    for piece in _WORD.findall(t):
        start, end = pos, pos + len(piece)
        pos = end
        inside = any(s < end and start < e for s, e in spans)
        cands = SHENG_SWAPS.get(piece) or SYNONYMS.get(piece)
        if cands and not inside and rng.random() < p:
            piece = cands[rng.integers(len(cands))]
        out.append(piece)
    return "".join(out)


FILLERS_START = ["manze ", "aki ", "honestly ", "sasa ", "yaani ", "bana ", "ok so ", "kusema ukweli "]
FILLERS_END = [" bana", " manze", " kabisa", " tu", " sana", " fr", " aki", ""]

# SMS-style shortenings students actually type
SMS = {"nimechoka": "nmechoka", "sijui": "sjui", "sana": "sna", "because": "coz", "you": "u",
       "are": "r", "please": "pls", "though": "tho", "with": "wth", "about": "abt", "people": "ppl",
       "nimekuwa": "nmekuwa", "kwa": "kwa", "really": "rly", "tomorrow": "tmrw", "kesho": "kesho"}


def noise_variant(text: str, rng: np.random.Generator) -> str:
    """Word dropout + SMS spelling + discourse fillers, never inside a crisis span."""
    t = normalise(text)
    spans = _protected_spans(t)
    out, pos = [], 0
    for piece in _WORD.findall(t):
        start, end = pos, pos + len(piece)
        pos = end
        inside = any(s < end and start < e for s, e in spans)
        if not inside and piece.strip() and piece.isalnum():
            r = rng.random()
            if r < 0.08:
                continue                       # drop word
            if r < 0.30 and piece in SMS:
                piece = SMS[piece]
        out.append(piece)
    v = " ".join("".join(out).split())
    if rng.random() < 0.5:
        v = FILLERS_START[rng.integers(len(FILLERS_START))] + v
    if rng.random() < 0.5:
        v = v + FILLERS_END[rng.integers(len(FILLERS_END))]
    return v


def augment_frame(seed: pd.DataFrame, n_per_seed: int = 15, rng: np.random.Generator | None = None,
                  back_translator=None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(0)
    rows = []
    for r in seed.itertuples(index=False):
        seen = {normalise(r.text)}
        sources = [r.text]
        if back_translator is not None:
            try:
                bt = back_translator(r.text)
                # keep back-translation only if it preserved the crisis signal
                if bt and default_valve().check(bt).crisis_hit == default_valve().check(r.text).crisis_hit:
                    sources.append(bt)
            except Exception:  # noqa: BLE001 — augmentation must never break the pipeline
                pass
        tries = 0
        while len(seen) < n_per_seed + 1 and tries < n_per_seed * 4:
            tries += 1
            base = sources[rng.integers(len(sources))]
            v = swap_variant(base, rng, p=rng.uniform(0.3, 0.8))
            method = "swap"
            if rng.random() < 0.7:
                v, method = noise_variant(v, rng), "swap+noise"
            if base is not r.text:
                method = "bt+" + method
            # label safety: a variant must keep the seed's valve status
            if default_valve().check(v).crisis_hit != default_valve().check(r.text).crisis_hit:
                continue
            if v not in seen:
                seen.add(v)
                rows.append({"seed_id": r.seed_id, "text": v, "category": r.category,
                             "crisis": int(r.crisis), "aug_method": method})
    return pd.DataFrame(rows, columns=["seed_id", "text", "category", "crisis", "aug_method"])


def marian_back_translator(device: str = "cpu"):
    """en->sw->en / sw->en->sw round trip using Helsinki-NLP MarianMT (optional, slow)."""
    from transformers import pipeline

    en_sw = pipeline("translation", model="Helsinki-NLP/opus-mt-en-swc", device=device)
    sw_en = pipeline("translation", model="Helsinki-NLP/opus-mt-swc-en", device=device)

    def bt(text: str) -> str:
        mid = en_sw(text, max_length=128)[0]["translation_text"]
        return sw_en(mid, max_length=128)[0]["translation_text"]

    return bt
