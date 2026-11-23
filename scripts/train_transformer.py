"""Fine-tune a multilingual encoder with LoRA (two stages), calibrate, pick threshold.

    python scripts/train_transformer.py --model afroxlmr [--intermediate] [--demo]
    python scripts/train_transformer.py --model distilbert          # comparison model

Stage 1 (--intermediate) downloads GoEmotions, AfriSenti-swa and tabularisai Swahili.
"""
from __future__ import annotations

import argparse
import json

from _common import load_config, load_frames, resolve, save_vocab

from triage import transformer as T
from triage.bundle import write_manifest
from triage.calibration import Calibrator, expected_calibration_error
from triage.config import CAT2ID
from triage.data import class_summary, data_fingerprint
from triage.safety_valve import default_valve
from triage.thresholds import choose_threshold


def main():
    cfg = load_config()
    tc = cfg["transformer"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=tc["default"], choices=list(tc["candidates"]))
    ap.add_argument("--intermediate", action="store_true", help="run stage-1 sentiment fine-tuning first")
    ap.add_argument("--intermediate-rows", type=int, default=tc["intermediate"]["max_rows_per_source"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.epochs:
        tc["finetune"]["epochs"] = args.epochs

    cand = tc["candidates"][args.model]
    out = resolve(args.out or f"artifacts/{args.model}_torch")
    device = T.torch.device("cpu") if args.cpu else T.pick_device()
    print(f"model={cand['name']} device={device}")

    frames, seed, is_demo = load_frames(args.demo)
    tr, ca = frames["train"], frames["calib"]
    print("train:", class_summary(tr))

    T.torch.manual_seed(cfg["project"]["seed"])   # reproducible LoRA/head initialisation
    model, tok = T.build(cand["name"])
    stage1 = None
    if args.intermediate:
        from triage.hf_data import load_intermediate

        inter = load_intermediate(args.intermediate_rows, cfg["project"]["seed"])
        stage1 = {"rows": int(len(inter)), "sources": inter["source"].value_counts().to_dict()}
        model = T.train_intermediate(model, tok, inter, tc, cand["lora_targets"], device)

    model, history = T.train_finetune(model, tok, tr, tc, cand["lora_targets"], device, calib_df=ca)
    model.to(device)
    z, k = T.predict_logits(model, tok, ca["text"].tolist(), tc["max_length"], device)
    cal = Calibrator.fit(z, ca["crisis"], k, [CAT2ID[c] for c in ca["category"]])
    p = cal.crisis_proba(z)
    s = cfg["safety"]
    thr = choose_threshold(p, ca["crisis"].to_numpy(), s["cost_false_negative"], s["cost_false_positive"],
                           s["target_recall"])

    T.save(model.cpu(), tok, out, cand["name"], tc["max_length"])
    save_vocab(tr["text"], out)
    man = write_manifest(
        out, model_type="torch", model_version=f"{args.model}-{data_fingerprint(tr)[:8]}",
        base_model=cand["name"], lora=tc["lora"], stage1=stage1, history=history,
        calibrator=cal.to_dict(), threshold=thr, lexicon_version=default_valve().version,
        data_fingerprint=data_fingerprint(seed), demo_data=is_demo, max_length=tc["max_length"],
        calib_ece_crisis=expected_calibration_error(p, ca["crisis"].to_numpy()),
        train_rows=int(len(tr)), calib_rows=int(len(ca)),
    )
    print(json.dumps({k: man[k] for k in ("model_version", "calibrator", "threshold")}, indent=2, default=float))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
