"""Seed data schema, validation and leakage-safe grouped splits.

Seed CSV schema (one row per REAL, consented, de-identified check-in):
    seed_id      unique id of the original message (augmented variants reuse it)
    text         the message
    category     one of config labels.categories
    crisis       0/1
    language     optional: en | sw | sheng | mixed   (used for the English-only vs code-mixed split)
    annotator    optional: who labelled it
    category_2, crisis_2   optional: second coder's labels on the double-coded sample (Cohen's kappa)

Leakage rules enforced here:
  * splits are made over seed_ids, never rows;
  * augmented rows only ever join TRAIN, and only for seed_ids already in train;
  * calib and test contain real seed messages only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CATEGORIES, load_config, resolve
from .pseudonymise import pseudonymise
from .text import language_group

REQUIRED = ["seed_id", "text", "category", "crisis"]


def validate(df: pd.DataFrame, name: str = "data") -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    df = df.copy()
    df["seed_id"] = df["seed_id"].astype(str)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    bad = set(df["category"]) - set(CATEGORIES)
    if bad:
        raise ValueError(f"{name}: unknown categories {bad}; expected {CATEGORIES}")
    df["crisis"] = df["crisis"].astype(int)
    if not set(df["crisis"]).issubset({0, 1}):
        raise ValueError(f"{name}: crisis must be 0/1")
    return df


def load_seed(path: str | Path | None = None, allow_demo: bool = False) -> tuple[pd.DataFrame, bool]:
    """Load real seed data; fall back to the synthetic demo set only if allowed.

    Returns (df, is_demo). Every downstream report prints a banner when is_demo is True.
    """
    cfg = load_config()["data"]
    p = resolve(path or cfg["seed_csv"])
    is_demo = False
    if not p.exists():
        if not allow_demo:
            raise FileNotFoundError(
                f"{p} not found. Put the consented Strathmore check-ins there, "
                "or pass --demo to run on synthetic data (results meaningless)."
            )
        p, is_demo = resolve(cfg["demo_csv"]), True
    df = validate(pd.read_csv(p), str(p))
    if df["seed_id"].duplicated().any():
        raise ValueError("seed_id must be unique in the seed file")
    # Pseudonymise on load, so nothing downstream ever sees identifiers.
    df["text"] = df["text"].map(pseudonymise)
    if "language" not in df.columns:
        df["language"] = None
    df["group"] = [language_group(t, l if isinstance(l, str) else None) for t, l in zip(df["text"], df["language"])]
    return df, is_demo


def make_splits(seed: pd.DataFrame, fractions: dict | None = None, random_state: int | None = None) -> dict[str, list[str]]:
    cfg = load_config()
    fr = fractions or cfg["data"]["split"]
    rs = cfg["project"]["seed"] if random_state is None else random_state
    ids = seed["seed_id"].to_numpy()
    # Stratify by crisis x category when every stratum is big enough, else by crisis only.
    strata = seed["crisis"].astype(str) + "_" + seed["category"]
    if strata.value_counts().min() < 4:
        strata = seed["crisis"].astype(str)
    test_frac = fr["test"]
    calib_frac = fr["calib"] / (1 - test_frac)
    rest, test = train_test_split(ids, test_size=test_frac, stratify=strata, random_state=rs)
    rest_strata = strata[seed["seed_id"].isin(rest)].to_numpy()
    rest = seed.loc[seed["seed_id"].isin(rest), "seed_id"].to_numpy()
    train, calib = train_test_split(rest, test_size=calib_frac, stratify=rest_strata, random_state=rs)
    splits = {"train": sorted(train), "calib": sorted(calib), "test": sorted(test)}
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["calib"]) & set(splits["test"]))
    return splits


def save_splits(splits: dict, out_dir: str | Path | None = None) -> Path:
    d = resolve(out_dir or load_config()["data"]["splits_dir"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "splits.json").write_text(json.dumps(splits, indent=1))
    return d / "splits.json"


def load_splits(path: str | Path | None = None) -> dict[str, list[str]]:
    p = resolve(path or Path(load_config()["data"]["splits_dir"]) / "splits.json")
    return json.loads(Path(p).read_text())


def assemble(seed: pd.DataFrame, splits: dict, augmented: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Return {'train','calib','test'} frames. Augmented rows join train only."""
    out = {k: seed[seed["seed_id"].isin(set(v))].copy() for k, v in splits.items()}
    for k in out:
        out[k]["is_augmented"] = False
    if augmented is not None and len(augmented):
        aug = validate(augmented, "augmented")
        held_out = set(splits["calib"]) | set(splits["test"])
        leaked = aug["seed_id"].isin(held_out)
        aug = aug[~leaked & aug["seed_id"].isin(set(splits["train"]))].copy()
        aug["is_augmented"] = True
        aug["group"] = aug["seed_id"].map(seed.set_index("seed_id")["group"])
        out["train"] = pd.concat([out["train"], aug], ignore_index=True)
        out["train"] = out["train"].drop_duplicates(subset=["text"])
    return out


def data_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for sid, t, c, k in df[["seed_id", "text", "category", "crisis"]].itertuples(index=False):
        h.update(f"{sid}\x1f{t}\x1f{c}\x1f{k}\n".encode())
    return h.hexdigest()[:16]


def class_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "crisis_rate": float(df["crisis"].mean()) if len(df) else 0.0,
        "crisis_n": int(df["crisis"].sum()),
        "categories": df["category"].value_counts().to_dict(),
        "groups": df["group"].value_counts().to_dict() if "group" in df else {},
    }


def rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(load_config()["project"]["seed"] if seed is None else seed)
