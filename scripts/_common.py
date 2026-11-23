"""Shared helpers for CLI scripts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from triage.config import load_config, resolve  # noqa: E402
from triage.data import assemble, load_seed, load_splits, make_splits, save_splits, validate  # noqa: E402
from triage.pseudonymise import pseudonymise  # noqa: E402
from triage.text import language_group, tokens  # noqa: E402

DEMO_BANNER = ("\n" + "!" * 78 + "\n!! DEMO DATA: synthetic template rows. Every number below is meaningless.\n"
               "!! Put real data in data/raw/strathmore_checkins.csv and re-run without --demo.\n" + "!" * 78 + "\n")


def load_frames(demo: bool, use_augmented: bool = True):
    cfg = load_config()
    if demo:   # --demo always means the synthetic set, even when real data is present
        seed, is_demo = load_seed(cfg["data"]["demo_csv"])[0], True
    else:
        seed, is_demo = load_seed(allow_demo=False)
    if is_demo:
        print(DEMO_BANNER)
    split_path = resolve(cfg["data"]["splits_dir"]) / ("splits_demo.json" if is_demo else "splits.json")
    if split_path.exists():
        splits = json.loads(split_path.read_text())
        known = set(seed["seed_id"])
        if not all(s in known for v in splits.values() for s in v):
            raise RuntimeError(f"{split_path} does not match the seed file; delete it to re-split")
    else:
        splits = make_splits(seed)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(splits, indent=1))
        print(f"created grouped splits -> {split_path}")
    aug = None
    aug_path = resolve(cfg["data"]["augmented_csv"]).with_name(
        "augmented_demo.csv" if is_demo else "augmented.csv")
    if use_augmented and aug_path.exists():
        aug = pd.read_csv(aug_path)
    frames = assemble(seed, splits, aug)
    # Positive/neutral non-crisis rows (never test): only when training (use_augmented) on real data.
    extra_cfg = cfg["data"].get("extra_train_csv")
    if use_augmented and not is_demo and extra_cfg and resolve(extra_cfg).exists():
        extra = validate(pd.read_csv(resolve(extra_cfg)), "extra_train")
        extra["text"] = extra["text"].map(pseudonymise)
        extra["group"] = [language_group(t, lang) for t, lang in zip(extra["text"], extra["language"])]
        extra["is_augmented"] = False
        assert not set(extra["seed_id"]) & set(seed["seed_id"])
        # a fixed 20% slice (by id hash) joins CALIBRATION: real check-in traffic includes ordinary and
        # happy messages, so the crisis threshold should be tuned with some in view. Test stays Zindi-only.
        to_calib = extra["seed_id"].map(lambda s: int(s[2:], 16) % 5 == 0)
        reps = int(cfg["data"].get("extra_train_repeat", 1))
        cols = frames["train"].columns.intersection(extra.columns)
        frames["train"] = pd.concat([frames["train"]] + [extra.loc[~to_calib, cols]] * reps, ignore_index=True)
        frames["calib"] = pd.concat([frames["calib"], extra.loc[to_calib, frames["calib"].columns.intersection(cols)]],
                                    ignore_index=True)
        print(f"+ positive/neutral rows from {extra_cfg}: {int((~to_calib).sum())} train (x{reps}), "
              f"{int(to_calib.sum())} calib")
    return frames, seed, is_demo


def save_vocab(texts, d: Path) -> None:
    vocab = sorted({t for x in texts for t in tokens(x)})
    (Path(d) / "vocab_tokens.json").write_text(json.dumps(vocab))


__all__ = ["ROOT", "DEMO_BANNER", "load_frames", "save_vocab", "load_config", "resolve",
           "load_splits", "save_splits"]
