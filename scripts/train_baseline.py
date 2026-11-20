"""Train the TF-IDF + logistic-regression baseline, calibrate, pick the threshold.

    python scripts/train_baseline.py [--demo] [--out artifacts/baseline]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from _common import load_config, load_frames, resolve, save_vocab

from triage.baseline import BaselineScorer
from triage.bundle import write_manifest
from triage.calibration import Calibrator, expected_calibration_error
from triage.config import CAT2ID
from triage.data import class_summary, data_fingerprint
from triage.safety_valve import default_valve
from triage.thresholds import choose_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config()
    out = resolve(args.out or cfg["serving"]["baseline_dir"])

    frames, seed, is_demo = load_frames(args.demo)
    tr, ca = frames["train"], frames["calib"]
    print("train:", class_summary(tr))
    print("calib:", class_summary(ca))

    model = BaselineScorer().fit(tr["text"], tr["crisis"], tr["category"])
    z, k = model.logits(ca["text"])
    cal = Calibrator.fit(z, ca["crisis"], k, [CAT2ID[c] for c in ca["category"]])
    p = cal.crisis_proba(z)
    s = cfg["safety"]
    thr = choose_threshold(p, ca["crisis"].to_numpy(), s["cost_false_negative"], s["cost_false_positive"],
                           s["target_recall"])
    model.save(out)
    save_vocab(tr["text"], out)
    man = write_manifest(
        out, model_type="tfidf_logreg", model_version=f"baseline-{data_fingerprint(tr)[:8]}",
        base_model=None, calibrator=cal.to_dict(), threshold=thr,
        lexicon_version=default_valve().version, data_fingerprint=data_fingerprint(seed),
        demo_data=is_demo, calib_ece_crisis=expected_calibration_error(p, ca["crisis"].to_numpy()),
        train_rows=int(len(tr)), calib_rows=int(len(ca)),
        top_crisis_features=model.top_features(15),
    )
    print(json.dumps({k: man[k] for k in ("model_version", "calibrator", "threshold", "calib_ece_crisis")},
                     indent=2, default=float))
    print(f"saved -> {out}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
