"""Create the augmented TRAINING set (~15 variants per train seed).

Only seeds in the train split are augmented, so no variant of a calib/test message
can exist anywhere. Usage:
    python scripts/augment.py [--demo] [--n 15] [--back-translate]
"""
from __future__ import annotations

import argparse

from _common import load_frames, resolve, load_config  # noqa: F401

from triage.augment import augment_frame, marian_back_translator
from triage.data import rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--back-translate", action="store_true", help="slow; downloads MarianMT en<->sw models")
    args = ap.parse_args()

    frames, seed, is_demo = load_frames(args.demo, use_augmented=False)
    train_seed = frames["train"]
    bt = marian_back_translator() if args.back_translate else None
    aug = augment_frame(train_seed, n_per_seed=args.n, rng=rng(), back_translator=bt)
    out = resolve(load_config()["data"]["augmented_csv"]).with_name(
        "augmented_demo.csv" if is_demo else "augmented.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    aug.to_csv(out, index=False)
    print(f"{len(train_seed)} train seeds -> {len(aug)} augmented rows -> {out}")
    print(f"crisis rows in augmented: {int(aug.crisis.sum())}")


if __name__ == "__main__":
    main()
