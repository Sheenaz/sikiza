"""Public datasets for stage-1 intermediate fine-tuning, mapped to ONE shared label:
3-way sentiment (0 negative, 1 neutral, 2 positive).

Why sentiment and not the five target categories? It is the only label all three sources
share without inventing a mapping; forcing GoEmotions "nervousness" onto
"social_anxiety" would be the lossy mapping the proposal warns about. Stage 1 only
teaches the encoder affect in English + Swahili; its head is discarded.

* GoEmotions (simplified): single-label rows, emotions grouped by the authors'
  sentiment taxonomy; "ambiguous" emotions dropped.
* AfriSenti, Swahili config: positive / neutral / negative tweets (loaded from the
  Hub's Parquet conversion — the original repo uses a loading script).
* tabularisai/swahili-sentiment-dataset: LLM-generated; filtered on its own quality
  score and mapped from 5-point labels.
"""
from __future__ import annotations

import pandas as pd

GOEMO_POS = {"admiration", "amusement", "approval", "caring", "desire", "excitement", "gratitude",
             "joy", "love", "optimism", "pride", "relief"}
GOEMO_NEG = {"anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment",
             "fear", "grief", "nervousness", "remorse", "sadness"}

AFRISENTI_SWA = ("https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment/resolve/"
                 "refs%2Fconvert%2Fparquet/swa/{split}/0000.parquet")


def load_goemotions(max_rows: int | None = None, seed: int = 0) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
    names = ds.features["labels"].feature.names
    rows = []
    for ex in ds:
        if len(ex["labels"]) != 1:
            continue
        n = names[ex["labels"][0]]
        lab = 2 if n in GOEMO_POS else 0 if n in GOEMO_NEG else 1 if n == "neutral" else None
        if lab is not None:
            rows.append((ex["text"], lab))
    df = pd.DataFrame(rows, columns=["text", "sentiment"]).assign(source="goemotions")
    return _cap(df, max_rows, seed)


def load_afrisenti_swahili(max_rows: int | None = None, seed: int = 0) -> pd.DataFrame:
    parts = [pd.read_parquet(AFRISENTI_SWA.format(split=s)) for s in ("train", "validation")]
    df = pd.concat(parts, ignore_index=True)
    # ClassLabel order on the Hub: 0 positive, 1 neutral, 2 negative -> ours
    df = pd.DataFrame({"text": df["tweet"], "sentiment": df["label"].map({0: 2, 1: 1, 2: 0})})
    return _cap(df.dropna().assign(source="afrisenti_swa"), max_rows, seed)


def load_tabularisai_swahili(max_rows: int | None = None, seed: int = 0, min_quality: float = 4.0) -> pd.DataFrame:
    from datasets import load_dataset

    df = load_dataset("tabularisai/swahili-sentiment-dataset", split="train").to_pandas()
    q = pd.to_numeric(df.get("Overall_Quality"), errors="coerce")
    if q.notna().any():
        df = df[q >= min_quality]
    s = df["sentiment"].astype(str).str.lower()
    lab = s.map(lambda x: 0 if "negative" in x else 2 if "positive" in x else 1 if "neutral" in x else None)
    df = pd.DataFrame({"text": df["text"], "sentiment": lab}).dropna()
    df["sentiment"] = df["sentiment"].astype(int)
    return _cap(df.assign(source="tabularisai_swa"), max_rows, seed)


def load_intermediate(max_rows_per_source: int | None = 20000, seed: int = 0) -> pd.DataFrame:
    frames = []
    for fn in (load_goemotions, load_afrisenti_swahili, load_tabularisai_swahili):
        try:
            frames.append(fn(max_rows_per_source, seed))
            print(f"[hf_data] {fn.__name__}: {len(frames[-1])} rows")
        except Exception as e:  # noqa: BLE001
            print(f"[hf_data] WARNING {fn.__name__} failed: {e}")
    if not frames:
        raise RuntimeError("no intermediate data could be loaded")
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed)


def _cap(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n and len(df) > n:
        # stratified cap keeps the sentiment balance of the source
        df = df.groupby("sentiment", group_keys=False).apply(
            lambda g: g.sample(min(len(g), max(1, int(n * len(g) / len(df)))), random_state=seed))
    return df.reset_index(drop=True)
