"""Build train-only positive / neutral check-ins: data/extra/positive_neutral.csv.

Why: every Zindi message comes from a mental-health survey (depression, suicide, alcohol, drugs), so the
model never saw an ordinary or happy message. At a ~1% crisis threshold it escalated messages such as
"I am feeling happy today" (out-of-distribution input). These rows teach it what non-distress looks like.

Sources (all crisis = 0; category `positive`, or `other` for the plain neutral written check-ins):
* GoEmotions (Demszky et al., 2020; Apache 2.0): English Reddit comments with a single positive emotion
  label (joy, gratitude, excitement, optimism, pride, relief). Filtered hard: first person, a stated
  good feeling, 4-25 words, no placeholders/URLs/slurs, no distress, risk or substance words, and no
  hit from the safety-valve lexicon.
* data/extra/kenyan_checkins_written.csv: everyday student check-ins in English, Swahili and Sheng,
  written for this project (AI-drafted; see AI_USE.md), because GoEmotions is English only.

The rows join TRAINING only (scripts/_common.py), never calibration or test, so every reported
test number is still measured on the same real Zindi messages.

    python scripts/prepare_positive.py [--goemotions path/to/train.parquet]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage.config import resolve  # noqa: E402
from triage.safety_valve import SafetyValve  # noqa: E402

URL = ("https://huggingface.co/datasets/google-research-datasets/go_emotions/resolve/main/"
       "simplified/train-00000-of-00001.parquet")
# GoEmotions label ids
# "amusement" (mostly "lol" banter) and "neutral" (off-topic Reddit talk, some of it dark) were tried and
# dropped after reading samples; neutral check-ins come from the written file instead.
POSITIVE = {13: "excitement", 15: "gratitude", 17: "joy", 20: "optimism", 21: "pride", 23: "relief"}
N_POSITIVE, SEED = 300, 42

FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|i'll|i’m|i’ve)\b", re.I)
# the writer describes their own good feeling or news, like a check-in would
FEELING = re.compile(r"glad|happy|excited|grateful|thankful|proud|relieved|enjoy|feel|feeling|good day|"
                     r"great day|love (it|this|that)|can't wait|so much fun|blessed|good news", re.I)
BLOCK = re.compile(
    r"die|dead|death|kill|suicid|depress|sad|cry|hate|hurt|pain|alone|lonel|drunk|drink|drug|weed|"
    r"\bhigh\b|anxi|stress|worr|scar|fear|afraid|sick|cancer|hospital|\blost\b|\blose\b|fail|broke|"
    r"debt|fuck|shit|damn|hell|sex|gun|shoot|himself|herself|myself|rape|abus|\bcut|blood|tired|empty|"
    r"hopeless|worthless|ugly|\bfat\b|funeral|grave|miss (him|her|you)|divorce|cheat|alcohol|beer|"
    r"vodka|smok|pill|overdose|bitch|ass\b|stupid|idiot|kms|unalive|end it|panic|therap|meds\b|"
    r"\bwar\b|attack|murder|crash|accident|bury|terribl|awful|horrib|worst|disgust|angry|mad\b",
    re.I)
PLACEHOLDER = re.compile(r"\[(NAME|RELIGION)\]|http|www\.|/r/|/u/", re.I)


def keep(text: str, valve: SafetyValve) -> bool:
    n = len(text.split())
    return (4 <= n <= 25 and FIRST_PERSON.search(text) is not None and FEELING.search(text) is not None
            and not BLOCK.search(text)
            and not PLACEHOLDER.search(text) and not valve.check(text).crisis_hit)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goemotions", help="local GoEmotions simplified train.parquet (default: download)")
    args = ap.parse_args()
    if args.goemotions:
        ge = pd.read_parquet(args.goemotions)
    else:
        with urllib.request.urlopen(URL, timeout=60) as r:
            ge = pd.read_parquet(io.BytesIO(r.read()))

    valve = SafetyValve()
    ge = ge[ge["labels"].map(len) == 1].copy()
    ge["label"] = ge["labels"].map(lambda x: int(x[0]))
    ge["text"] = ge["text"].str.strip()
    ge = ge.drop_duplicates("text")
    ge = ge[[keep(t, valve) for t in ge["text"]]]
    pos = ge[ge["label"].isin(POSITIVE)].sample(N_POSITIVE, random_state=SEED)
    goemo = pos.assign(category="positive", 
        language="en", source="goemotions",
        emotion=lambda d: d["label"].map(lambda i: POSITIVE.get(i, "neutral")))

    written = pd.read_csv(resolve("data/extra/kenyan_checkins_written.csv")).assign(source="written")
    written["emotion"] = written["category"].map({"positive": "positive", "other": "neutral"})
    for t in written["text"]:
        if valve.check(t).crisis_hit:
            raise SystemExit(f"written check-in hits the safety valve: {t!r}")

    cols = ["text", "language", "source", "emotion", "category"]
    out = pd.concat([goemo[cols], written[cols]], ignore_index=True)
    out["seed_id"] = "p-" + out["text"].map(lambda t: hashlib.sha1(t.lower().encode()).hexdigest()[:10])
    out = out.drop_duplicates("seed_id")
    out["crisis"] = 0
    path = resolve("data/extra/positive_neutral.csv")
    out[["seed_id", "text", "category", "crisis", "language", "source", "emotion"]].to_csv(path, index=False)
    print(f"{len(out)} train-only non-crisis rows -> {path}")
    print(out.groupby(["source", "emotion"]).size().to_string())


if __name__ == "__main__":
    main()
