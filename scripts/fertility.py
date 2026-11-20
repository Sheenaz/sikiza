"""Tokenizer fertility: how badly does each tokenizer fragment Sheng? (proposal: "measure first")

fertility = subword tokens per whitespace word (lower is better; ~1.3 is good for English).
Also: share of words split into >= 3 pieces, and the split of lexicon Sheng/Swahili terms.

    python scripts/fertility.py [--demo]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import yaml

from _common import load_config, load_frames, resolve

from triage.text import normalise

PROBE = ["nimehang", "imeniweza", "nimeboeka", "kujitoa", "kujinyonga", "mbogi", "doh", "ganji",
         "chapaa", "keja", "budaa", "mathe", "manze", "sijai", "fiti", "nitajihang", "nimechoka"]


def stats(tok, texts):
    per_word = []
    for t in texts:
        for w in normalise(t).split():
            per_word.append(len(tok.tokenize(w)))
    a = np.array(per_word)
    return {"fertility": float(a.mean()), "share_ge3_pieces": float((a >= 3).mean()), "words": int(len(a))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    frames, seed, _ = load_frames(args.demo, use_augmented=False)
    cands = load_config()["transformer"]["candidates"]
    res = {}
    for key, c in cands.items():
        tok = AutoTokenizer.from_pretrained(c["name"])
        res[key] = {
            "model": c["name"],
            "english_only": stats(tok, seed.loc[seed.group == "english_only", "text"]),
            "code_mixed": stats(tok, seed.loc[seed.group == "code_mixed", "text"]),
            "probe_terms": {w: tok.tokenize(w) for w in PROBE},
        }
        print(f"{key:10s} EN fert={res[key]['english_only']['fertility']:.2f}  "
              f"MIXED fert={res[key]['code_mixed']['fertility']:.2f}  "
              f"MIXED >=3 pieces={res[key]['code_mixed']['share_ge3_pieces']:.1%}")
        print("           ", {w: len(p) for w, p in list(res[key]["probe_terms"].items())[:8]})
    out = resolve("artifacts/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "fertility.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"-> {out / 'fertility.json'}")


if __name__ == "__main__":
    main()
