"""Text normalisation for English–Swahili–Sheng messages.

Sheng is written inconsistently: "nime-hang", "nimehang", "nimehaaang" and
"nmehang" are the same word. Normalisation is applied identically in training,
the safety valve and serving so all three see the same string.
"""
from __future__ import annotations

import re
import unicodedata

_REPEAT = re.compile(r"(\w)\1{2,}")          # "sanaaaa" -> "sanaa"
_INTRAWORD_HYPHEN = re.compile(r"(?<=\w)[-_'’](?=\w)")
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")

# Common SMS / Sheng digit substitutions, only applied inside alphabetic words.
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower()
    t = _INTRAWORD_HYPHEN.sub("", t)
    t = _REPEAT.sub(r"\1\1", t)
    words = []
    for w in t.split():
        # only de-leet tokens that are mostly letters ("l1fe" -> "life", but keep "2026")
        letters = sum(c.isalpha() for c in w)
        if letters and letters >= len(w) / 2:
            w = w.translate(_LEET)
        words.append(w)
    return _WS.sub(" ", " ".join(words)).strip()


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(normalise(text))


# --- crude code-mixing detector ------------------------------------------------
# Used to split evaluation into English-only vs code-mixed (proposal §3 "Success").
# Prefer the human `language` column when it exists; this is a fallback.
_SW_SHENG_MARKERS = {
    # Swahili function words / frequent forms
    "na", "ni", "sana", "kwa", "lakini", "sijui", "sina", "niko", "nime", "nataka",
    "sitaki", "kwenye", "hii", "hiyo", "yangu", "wangu", "mimi", "wewe", "kila",
    "kitu", "leo", "jana", "kesho", "pesa", "shule", "maisha", "nimechoka", "kweli",
    "tu", "pia", "hata", "bado", "sasa", "kama", "nini", "vipi", "mbona", "juu",
    # Sheng
    "doh", "mbogi", "msee", "manze", "maze", "sare", "fiti", "poa", "budaa",
    "mathe", "dem", "beshte", "keja", "mtaa", "ganji", "chapaa", "form", "ati",
    "sijai", "kuna", "imeniweza", "stress", "kuhang", "nimehang", "nimeboeka",
}
_SW_PREFIX = re.compile(r"^(ni|u|a|tu|m|wa|si|ha|hu|ku)(me|na|ta|li|ki|nge)\w{2,}$")


def is_code_mixed(text: str) -> bool:
    toks = tokens(text)
    if not toks:
        return False
    hits = sum(t in _SW_SHENG_MARKERS or bool(_SW_PREFIX.match(t)) for t in toks)
    return hits >= 1


def language_group(text: str, language: str | None = None) -> str:
    """Return 'english_only' or 'code_mixed'."""
    if language:
        return "english_only" if language.strip().lower() in {"en", "english"} else "code_mixed"
    return "code_mixed" if is_code_mixed(text) else "english_only"
