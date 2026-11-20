"""Zero-shot NLI check (week 9): how far does an off-the-shelf multilingual NLI model get
with no Strathmore training at all? Reported on the same TEST split.

    python scripts/zero_shot.py [--demo]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from _common import load_config, load_frames, resolve

from triage.config import CATEGORIES
from triage.evaluate import full_report
from triage.pseudonymise import pseudonymise
from triage.thresholds import recall_threshold

CAT_HYP = {
    "academic_stress": "stress about exams, studies or grades",
    "financial_strain": "problems with money, fees or rent",
    "social_anxiety": "anxiety about people, friends or speaking in public",
    "low_mood": "sadness, emptiness or low mood",
    "other": "an ordinary update or something else",
}
CRISIS_HYP = "thoughts of suicide, self-harm or wanting to die"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    from transformers import pipeline

    cfg = load_config()
    clf = pipeline("zero-shot-classification", model=cfg["zero_shot"]["model"])
    frames, _, is_demo = load_frames(args.demo, use_augmented=False)

    def score(df):
        texts = [pseudonymise(t) for t in df["text"]]
        crisis = np.array([clf(t, [CRISIS_HYP], multi_label=True)["scores"][0] for t in texts])
        cats = []
        for t in texts:
            r = clf(t, list(CAT_HYP.values()))
            cats.append(CATEGORIES[list(CAT_HYP.values()).index(r["labels"][0])])
        return crisis, cats

    pc, _ = score(frames["calib"])
    thr = recall_threshold(pc, frames["calib"]["crisis"].to_numpy(), cfg["safety"]["target_recall"])
    pt, cats = score(frames["test"])
    rep = full_report(frames["test"], (pt >= thr).astype(int), cats,
                      cfg["evaluation"]["bootstrap_iterations"], cfg["evaluation"]["ci"])
    rep["threshold"] = thr
    rep["demo_data"] = is_demo
    out = resolve("artifacts/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "zero_shot.json").write_text(json.dumps(rep, indent=2, default=float))
    o = rep["overall"]
    print(f"zero-shot  crisis recall={o['crisis']['recall']:.3f} exact CI={o['crisis']['recall_exact_ci']} "
          f"escalation={o['crisis']['escalation_rate']:.3f}  macro-F1={o['category']['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
