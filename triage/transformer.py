"""Multilingual encoder + LoRA with two heads (crisis logit, 5-way category).

Stage 1 (optional): LoRA + sentiment head on public data -> merge LoRA into encoder.
Stage 2: fresh LoRA + crisis/category heads on Strathmore train rows (seed + augmented).
Loss  = crisis_loss_weight * BCE(crisis) + CE(category). No class weights (calibration).
Saved bundle: merged encoder + heads + tokenizer + manifest (model_type "torch").
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import CAT2ID, CATEGORIES


def pick_device() -> torch.device:
    import os
    import platform

    if os.environ.get("TRIAGE_DEVICE"):
        return torch.device(os.environ["TRIAGE_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Recent torch builds call Metal APIs that only exist on macOS >= 15; on older macOS
    # training on MPS aborts with "-[MPSNDArray arrayViewWithShape:strides:]: unrecognized selector".
    mac = platform.mac_ver()[0]
    if torch.backends.mps.is_available() and mac and int(mac.split(".")[0]) >= 15:
        return torch.device("mps")
    return torch.device("cpu")


class MultiTaskModel(nn.Module):
    def __init__(self, encoder: nn.Module, hidden: int, n_cat: int = len(CATEGORIES)):
        super().__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(0.1)
        self.crisis_head = nn.Linear(hidden, 1)
        self.cat_head = nn.Linear(hidden, n_cat)
        self.sent_head = nn.Linear(hidden, 3)   # stage 1 only

    def pooled(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)   # mean pooling

    def forward(self, input_ids, attention_mask):
        h = self.drop(self.pooled(input_ids, attention_mask))
        return self.crisis_head(h).squeeze(-1), self.cat_head(h)

    def sentiment(self, input_ids, attention_mask):
        return self.sent_head(self.drop(self.pooled(input_ids, attention_mask)))


def _add_lora(model: MultiTaskModel, targets: list[str], r: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets, bias="none")
    model.encoder = get_peft_model(model.encoder, cfg)
    return model


def _merge_lora(model: MultiTaskModel):
    if hasattr(model.encoder, "merge_and_unload"):
        model.encoder = model.encoder.merge_and_unload()
    return model


def build(base_model: str):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    enc = AutoModel.from_pretrained(base_model)
    return MultiTaskModel(enc, enc.config.hidden_size), tok


def _batches(n: int, bs: int, rng: np.random.Generator | None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def _encode(tok, texts, max_length, device):
    enc = tok(list(texts), truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def _optimizer(model, lr, steps):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    warm = max(1, int(0.06 * steps))
    def lr_lambda(s: int) -> float:  # linear warmup, then linear decay to 0
        if s < warm:
            return (s + 1) / warm
        return max(0.0, (steps - s) / max(1, steps - warm))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched


def _freeze_encoder_base(model: MultiTaskModel):
    # after get_peft_model only LoRA params are trainable in the encoder; heads stay trainable
    for p in model.crisis_head.parameters():
        p.requires_grad = True
    for p in model.cat_head.parameters():
        p.requires_grad = True
    for p in model.sent_head.parameters():
        p.requires_grad = True


def train_intermediate(model, tok, df, cfg_t: dict, targets, device, seed=0, log=print):
    lc, ic = cfg_t["lora"], cfg_t["intermediate"]
    model = _add_lora(model, targets, lc["r"], lc["alpha"], lc["dropout"])
    _freeze_encoder_base(model)
    model.to(device).train()
    texts, y = df["text"].tolist(), torch.tensor(df["sentiment"].to_numpy(), dtype=torch.long)
    steps = ic["epochs"] * math.ceil(len(texts) / ic["batch_size"])
    opt, sched = _optimizer(model, ic["lr"], steps)
    rng = np.random.default_rng(seed)
    step = 0
    for ep in range(ic["epochs"]):
        for b in _batches(len(texts), ic["batch_size"], rng):
            ids, mask = _encode(tok, [texts[i] for i in b], cfg_t["max_length"], device)
            loss = nn.functional.cross_entropy(model.sentiment(ids, mask), y[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
            if step % 100 == 0:
                log(f"[stage1] epoch {ep} step {step}/{steps} loss {loss.item():.4f}")
    return _merge_lora(model.cpu())


def train_finetune(model, tok, train_df, cfg_t: dict, targets, device, calib_df=None, seed=0, log=print):
    lc, fc = cfg_t["lora"], cfg_t["finetune"]
    model = _add_lora(model, targets, lc["r"], lc["alpha"], lc["dropout"])
    _freeze_encoder_base(model)
    model.to(device).train()
    texts = train_df["text"].tolist()
    yc = torch.tensor(train_df["crisis"].to_numpy(), dtype=torch.float32)
    yk = torch.tensor([CAT2ID[c] for c in train_df["category"]], dtype=torch.long)
    steps = fc["epochs"] * math.ceil(len(texts) / fc["batch_size"])
    opt, sched = _optimizer(model, fc["lr"], steps)
    rng = np.random.default_rng(seed)
    history = []
    for ep in range(fc["epochs"]):
        model.train()
        tot = 0.0
        for b in _batches(len(texts), fc["batch_size"], rng):
            ids, mask = _encode(tok, [texts[i] for i in b], cfg_t["max_length"], device)
            zc, zk = model(ids, mask)
            loss = (fc["crisis_loss_weight"] * nn.functional.binary_cross_entropy_with_logits(zc, yc[b].to(device))
                    + nn.functional.cross_entropy(zk, yk[b].to(device)))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(b)
        rec = {"epoch": ep, "train_loss": tot / len(texts)}
        if calib_df is not None and len(calib_df):
            # informational only — no epoch selection on calib (it is reserved for calibration)
            z, k = predict_logits(model, tok, calib_df["text"].tolist(), cfg_t["max_length"], device)
            rec["calib_crisis_bce"] = float(np.mean(np.logaddexp(0, z) - calib_df["crisis"].to_numpy() * z))
        history.append(rec)
        log(f"[stage2] {rec}")
    return _merge_lora(model.cpu()), history


@torch.no_grad()
def predict_logits(model, tok, texts, max_length=128, device=None, bs=64):
    device = device or next(model.parameters()).device
    model.eval()
    zs, ks = [], []
    for i in range(0, len(texts), bs):
        ids, mask = _encode(tok, texts[i:i + bs], max_length, device)
        z, k = model(ids, mask)
        zs.append(z.float().cpu().numpy()); ks.append(k.float().cpu().numpy())
    return np.concatenate(zs), np.concatenate(ks)


def save(model: MultiTaskModel, tok, d: str | Path, base_model: str, max_length: int) -> None:
    d = Path(d)
    (d / "encoder").mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(d / "encoder")
    tok.save_pretrained(d)
    torch.save({"crisis_head": model.crisis_head.state_dict(), "cat_head": model.cat_head.state_dict()},
               d / "heads.pt")
    (d / "torch_config.json").write_text(json.dumps({"base_model": base_model, "max_length": max_length,
                                                     "categories": CATEGORIES}))


class TorchScorer:
    kind = "torch"

    def __init__(self, model, tok, max_length=128, device=None):
        self.model, self.tok, self.max_length = model, tok, max_length
        self.device = device or torch.device("cpu")
        self.model.to(self.device).eval()

    @classmethod
    def load(cls, d: str | Path, device=None) -> "TorchScorer":
        from transformers import AutoModel, AutoTokenizer

        d = Path(d)
        cfg = json.loads((d / "torch_config.json").read_text())
        enc = AutoModel.from_pretrained(d / "encoder")
        model = MultiTaskModel(enc, enc.config.hidden_size, len(cfg["categories"]))
        heads = torch.load(d / "heads.pt", map_location="cpu")
        model.crisis_head.load_state_dict(heads["crisis_head"])
        model.cat_head.load_state_dict(heads["cat_head"])
        return cls(model, AutoTokenizer.from_pretrained(d), cfg["max_length"], device)

    def logits(self, texts):
        return predict_logits(self.model, self.tok, list(texts), self.max_length, self.device)
