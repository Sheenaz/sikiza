"""Monthly drift report: new Sheng terms + embedding novelty vs the training data.

    python scripts/drift_report.py --new data/raw/checkins_2026-12.csv [--bundle artifacts/afroxlmr_torch] [--demo]

--new: consented, de-identified check-ins collected this month (column `text`).
Without --bundle a TF-IDF/SVD embedding is used (fast, no torch).
"""
from __future__ import annotations

import argparse
import json
import re
import time

import pandas as pd
import yaml

from _common import load_config, load_frames, resolve

from triage.drift import drift_report, tfidf_embedder, torch_embedder
from triage.pseudonymise import pseudonymise


def lexicon_terms() -> set[str]:
    spec = yaml.safe_load(resolve(load_config()["safety"]["lexicon"]).read_text())
    return {w for r in spec["rules"] for w in re.findall(r"[a-z]{3,}", r["pattern"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    frames, seed, _ = load_frames(args.demo, use_augmented=False)
    train = frames["train"]["text"].tolist()
    new = [pseudonymise(t) for t in pd.read_csv(args.new)["text"].astype(str)]
    emb = torch_embedder(resolve(args.bundle)) if args.bundle else tfidf_embedder(train)
    rep = drift_report(train, new, lexicon_terms(), emb)
    out = resolve("artifacts/reports")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"drift_{time.strftime('%Y-%m')}.json"
    path.write_text(json.dumps(rep, indent=2))
    print(f"novel share {rep['novel_share']:.1%} (≈5% expected without drift) | "
          f"centroid shift {rep['centroid_cosine_shift']:.4f} | OOV rate {rep['oov_token_rate']:.1%}")
    print("top OOV tokens:", rep["top_oov_tokens"][:10])
    for c in rep["clusters"]:
        if c["candidate_new_terms"]:
            print(f"  cluster {c['cluster']} (n={c['size']}): {c['candidate_new_terms'][:6]}")
    if rep["alert"]:
        print("ALERT: >10% of messages are far from anything seen in training — review with the student panel.")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
