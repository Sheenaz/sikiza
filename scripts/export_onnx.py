"""Export a trained torch bundle to ONNX int8 for serving and the mobile app.

    python scripts/export_onnx.py --src artifacts/afroxlmr_torch --out artifacts/model [--demo]
"""
from __future__ import annotations

import argparse
import json

from _common import load_config, load_frames, resolve

from triage.export import export_onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="artifacts/model")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--quantization", choices=["int8", "none"], default="none",
                    help="none (default): serve fp32. int8 is 4x smaller but shifted near-threshold crisis "
                         "probabilities enough to escalate ordinary happy messages")
    args = ap.parse_args()
    frames, _, _ = load_frames(args.demo, use_augmented=False)
    # calib split: re-calibrates the int8 model and measures parity (never the test split)
    man = export_onnx(resolve(args.src), resolve(args.out), frames["calib"],
                      safety_cfg=load_config()["safety"], quantization=args.quantization)
    print(json.dumps({k: man.get(k) for k in ("model_version", "size_mb", "parity", "threshold")},
                     indent=2, default=float))
    p = man.get("parity") or {}
    if p and p["crisis_decision_agreement"] < 1.0:
        print("NOTE: int8 changes some crisis decisions vs torch; the int8 model has its own "
              "calibration + threshold (in manifest). Evaluate artifacts/model, not the torch bundle.")


if __name__ == "__main__":
    main()
