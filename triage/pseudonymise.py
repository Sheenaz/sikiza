"""Strip direct identifiers before anything else touches the message (DPA 2019).

Removes: Kenyan phone numbers, emails, URLs, @handles, M-Pesa transaction codes,
student / national ID numbers, and names (self-introductions + a gazetteer).

Limitation: names that are not introduced ("Wanjiru alinitusi") and are not in
the gazetteer will survive. data/lexicon/names.txt is to be extended from the real data,
and this is defence in depth, not a guarantee.
"""
from __future__ import annotations

import re
from functools import lru_cache

from .config import ROOT

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("[EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("[URL]", re.compile(r"\b(?:https?://|www\.)\S+", re.I)),
    # +254 7xx / 07xx / 01xx, with optional spaces or dashes
    ("[PHONE]", re.compile(r"(?<!\w)(?:\+?254|0)[\s-]?[17]\d{2}[\s-]?\d{3}[\s-]?\d{3}(?!\w)")),
    # M-Pesa codes: 10 uppercase alphanumerics with at least one digit and one letter
    ("[TXN]", re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{10}\b")),
    ("[HANDLE]", re.compile(r"(?<!\w)@\w{2,}")),
    # student / national ID style numbers (6–9 digits)
    ("[ID]", re.compile(r"(?<!\d)\d{6,9}(?!\d)")),
]

# "my name is X", "I'm X", "naitwa X", "jina langu ni X", "mimi ni X"
_INTRO = re.compile(
    r"\b(my name is|name's|i am called|naitwa|jina langu ni|ni mimi)\s+([A-Za-z][a-z'’-]+)",
    re.I,
)


@lru_cache(maxsize=1)
def _gazetteer() -> re.Pattern | None:
    path = ROOT / "data" / "lexicon" / "names.txt"
    if not path.exists():
        return None
    names = sorted(
        {ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")},
        key=len,
        reverse=True,
    )
    if not names:
        return None
    return re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\b", re.I)


def pseudonymise(text: str) -> str:
    out = text
    for token, pat in _PATTERNS:
        out = pat.sub(token, out)
    out = _INTRO.sub(lambda m: f"{m.group(1)} [NAME]", out)
    gz = _gazetteer()
    if gz is not None:
        out = gz.sub("[NAME]", out)
    return out
