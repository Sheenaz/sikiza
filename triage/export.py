"""Export a torch bundle to ONNX and quantise to int8 (dynamic, weights-only).

Output bundle (model_type "onnx") contains model.int8.onnx, tokenizer.json and a
manifest copied from the torch bundle with the SAME calibrator and threshold —
then checks parity, because quantisation can shift probabilities near threshold.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .bundle import OnnxScorer, read_manifest, write_manifest
from .calibration import Calibrator
from .transformer import TorchScorer


class _ExportWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids, attention_mask)


def export_onnx(torch_dir: str | Path, out_dir: str | Path, calib_df=None, opset: int = 17,
                safety_cfg: dict | None = None) -> dict:
    """calib_df (text, crisis, category): used to RE-CALIBRATE the int8 model and re-derive
    its threshold, because int8 shifts logits (measured up to ~2 logits on a weak model) and
    the deployed artefact is the int8 one. Parity vs torch is reported on the same rows."""
    parity_texts = calib_df["text"].tolist() if calib_df is not None else None
    torch_dir, out_dir = Path(torch_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer = TorchScorer.load(torch_dir)
    model = _ExportWrapper(scorer.model.cpu().eval())
    enc = scorer.tok(["habari, niko sawa", "exams zimeniweza sana this week"], padding=True,
                     return_tensors="pt")
    args = (enc["input_ids"], enc["attention_mask"])
    fp32 = out_dir / "model.onnx"
    kw = dict(input_names=["input_ids", "attention_mask"], output_names=["crisis_logit", "category_logits"],
              opset_version=opset)
    try:
        torch.onnx.export(model, args, str(fp32), dynamo=False,
                          dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                                        "attention_mask": {0: "batch", 1: "seq"},
                                        "crisis_logit": {0: "batch"}, "category_logits": {0: "batch"}}, **kw)
    except Exception as e:  # noqa: BLE001 — newer torch may drop the legacy exporter
        print(f"[export] legacy exporter failed ({e}); trying dynamo exporter")
        batch, seq = torch.export.Dim("batch"), torch.export.Dim("seq")
        torch.onnx.export(model, args, str(fp32), dynamo=True, external_data=False,
                          dynamic_shapes={"input_ids": {0: batch, 1: seq},
                                          "attention_mask": {0: batch, 1: seq}}, **kw)

    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    int8 = out_dir / "model.int8.onnx"
    pre = out_dir / "model.pre.onnx"
    quant_pre_process(str(fp32), str(pre))
    # per-channel int8 everywhere (incl. the 250k-token embedding: keeps the file ~280 MB
    # for phones). MatMul-only quantisation was tested: 3x larger for little accuracy gain.
    quantize_dynamic(str(pre), str(int8), weight_type=QuantType.QInt8, per_channel=True)
    pre.unlink()

    # tokenizer.json for the torch-free runtime
    scorer.tok.save_pretrained(out_dir / "_tok")
    shutil.copy(out_dir / "_tok" / "tokenizer.json", out_dir / "tokenizer.json")
    shutil.rmtree(out_dir / "_tok")
    for extra in ("vocab_tokens.json",):
        if (torch_dir / extra).exists():
            shutil.copy(torch_dir / extra, out_dir / extra)

    man = read_manifest(torch_dir)
    man.update(model_type="onnx", source_torch_bundle=Path(torch_dir).name, quantization="dynamic_int8_per_channel",
               model_version=f"{man.get('model_version', 'unknown')}-onnx-int8",
               max_length=scorer.max_length, pad_id=scorer.tok.pad_token_id, pad_token=scorer.tok.pad_token,
               size_mb={"fp32": round(fp32.stat().st_size / 1e6, 1), "int8": round(int8.stat().st_size / 1e6, 1)})
    man.pop("created_at", None)

    if parity_texts:
        from .config import CAT2ID
        from .thresholds import choose_threshold

        cal_t = Calibrator.from_dict(man["calibrator"])
        thr_t = man["threshold"]["threshold"]
        zt, kt = scorer.logits(parity_texts)
        onnx = OnnxScorer(out_dir, scorer.max_length, pad_id=man["pad_id"], pad_token=man["pad_token"])
        zo, ko = onnx.logits(parity_texts)
        # re-calibrate + re-threshold on the int8 model's own calib outputs
        y = calib_df["crisis"].to_numpy()
        cal_o = Calibrator.fit(zo, y, ko, [CAT2ID[c] for c in calib_df["category"]])
        s = safety_cfg or {"cost_false_negative": 100.0, "cost_false_positive": 1.0, "target_recall": 0.95}
        thr = choose_threshold(cal_o.crisis_proba(zo), y, s["cost_false_negative"], s["cost_false_positive"],
                               s["target_recall"])
        man["torch_calibrator"], man["torch_threshold"] = man["calibrator"], man["threshold"]
        man["calibrator"], man["threshold"] = cal_o.to_dict(), thr
        pt, po = cal_t.crisis_proba(zt), cal_o.crisis_proba(zo)
        parity = {
            "n": len(parity_texts),
            "max_abs_crisis_logit_diff": float(np.max(np.abs(zt - zo))),
            "max_abs_crisis_prob_diff_after_recal": float(np.max(np.abs(pt - po))),
            "crisis_decision_agreement": float(np.mean((pt >= thr_t) == (po >= thr["threshold"]))),
            "category_agreement": float(np.mean(kt.argmax(1) == ko.argmax(1))),
        }
        t0 = time.perf_counter()
        for t in parity_texts[:50]:
            onnx.logits([t])
        parity["cpu_ms_per_message_int8"] = (time.perf_counter() - t0) * 1000 / min(50, len(parity_texts))
        man["parity"] = parity
    fp32.unlink()  # ship int8 only
    write_manifest(out_dir, **man)
    return man
