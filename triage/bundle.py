"""Model bundles: a directory holding a scorer + manifest.json.

manifest.json pins everything a prediction depends on:
    model_type, model_version, base_model, calibrator, threshold,
    lexicon_version, data_fingerprint, created_at, (optional) metrics
and serving logs those versions with every prediction.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from .calibration import Calibrator


class Scorer(Protocol):
    kind: str

    def logits(self, texts) -> tuple[np.ndarray, np.ndarray]: ...


def write_manifest(d: str | Path, **fields) -> dict:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    fields.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (d / "manifest.json").write_text(json.dumps(fields, indent=2, default=str))
    return fields


def read_manifest(d: str | Path) -> dict:
    return json.loads((Path(d) / "manifest.json").read_text())


class OnnxScorer:
    """Torch-free inference: onnxruntime + HF `tokenizers`. Used in Docker/serving."""

    kind = "onnx"

    def __init__(self, d: str | Path, max_length: int = 128, threads: int | None = None,
                 pad_id: int = 0, pad_token: str = "[PAD]"):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(d)
        model = d / "model.int8.onnx"
        if not model.exists():
            model = d / "model.onnx"
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_length)
        self.tok.enable_padding(pad_id=pad_id, pad_token=pad_token)  # XLM-R pads with id 1, not 0

    def logits(self, texts):
        enc = self.tok.encode_batch(list(texts))
        feeds = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
        }
        feeds = {k: v for k, v in feeds.items() if k in self.input_names}
        crisis, cat = self.session.run(["crisis_logit", "category_logits"], feeds)
        return crisis.reshape(-1), cat


def load_bundle(d: str | Path) -> tuple[Scorer, Calibrator, dict]:
    d = Path(d)
    man = read_manifest(d)
    t = man["model_type"]
    if t == "tfidf_logreg":
        from .baseline import BaselineScorer
        scorer = BaselineScorer.load(d)
    elif t == "onnx":
        scorer = OnnxScorer(d, man.get("max_length", 128), pad_id=man.get("pad_id", 0),
                            pad_token=man.get("pad_token", "[PAD]"))
    elif t == "torch":
        from .transformer import TorchScorer
        scorer = TorchScorer.load(d)
    else:
        raise ValueError(f"unknown model_type {t}")
    return scorer, Calibrator.from_dict(man["calibrator"]), man
