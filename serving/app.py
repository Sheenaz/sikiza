"""FastAPI service: the 'API gateway' box of the serving sketch.

    POST /v1/checkin          student check-in -> crisis flag + approved response
    GET  /v1/lexicon          safety-valve rules for the app's offline fallback
    GET  /v1/version          pinned model / threshold / lexicon / library versions
    GET  /health
    GET  /v1/escalations      counsellor queue      (X-Counsellor-Key required)
    POST /v1/escalations/{id}/resolve                (X-Counsellor-Key required)

Env:
    TRIAGE_MODEL_DIR       bundle to serve (default: config serving.model_dir, else baseline_dir)
    COUNSELLOR_API_KEY     required for escalation endpoints
    ALLOW_DRAFT_LIBRARY=1  serve unapproved strategies (development only)
    ESCALATION_WEBHOOK     optional URL notified (no message text) on each escalation
    TRIAGE_AUDIT_LOG, TRIAGE_ESCALATION_DB   override storage paths (Docker: /app/state volume)
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import sys
import threading
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage.audit import AuditLog, EscalationStore  # noqa: E402
from triage.bundle import load_bundle  # noqa: E402
from triage.config import load_config, resolve  # noqa: E402
from triage.pipeline import TriagePipeline  # noqa: E402
from triage.retrieval import CopingLibrary  # noqa: E402
from triage.safety_valve import SafetyValve  # noqa: E402

log = logging.getLogger("triage")
STATE: dict = {}


def _model_dir() -> Path:
    s = load_config()["serving"]
    if os.environ.get("TRIAGE_MODEL_DIR"):
        return resolve(os.environ["TRIAGE_MODEL_DIR"])
    primary = resolve(s["model_dir"])
    return primary if (primary / "manifest.json").exists() else resolve(s["baseline_dir"])


def build_state() -> dict:
    s = load_config()["serving"]
    d = _model_dir()
    scorer, cal, man = load_bundle(d)
    lib = CopingLibrary(allow_draft=os.environ.get("ALLOW_DRAFT_LIBRARY") == "1" or None)
    valve = SafetyValve()
    vocab_path = d / "vocab_tokens.json"
    return {
        "pipeline": TriagePipeline(scorer, cal, man, valve, lib),
        "audit": AuditLog(resolve(os.environ.get("TRIAGE_AUDIT_LOG", s["audit_log"]))),
        "escalations": EscalationStore(resolve(os.environ.get("TRIAGE_ESCALATION_DB", s["escalation_db"]))),
        "vocab": set(json.loads(vocab_path.read_text())) if vocab_path.exists() else None,
        "valve": valve,
        "model_dir": str(d),
        "warnings": [w for w in [
            "library contains UNAPPROVED strategies" if lib.is_draft else None,
            "lexicon has not been reviewed" if not valve.reviewed else None,
            "model was trained on DEMO data" if man.get("demo_data") else None,
        ] if w],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.update(build_state())
    for w in STATE["warnings"]:
        log.warning("STARTUP WARNING: %s", w)
    yield
    STATE.clear()


app = FastAPI(title="Sikiza: Sheng-aware distress triage", version="0.1.0", lifespan=lifespan)


class CheckIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    student_ref: str | None = Field(None, max_length=64, description="pseudonymous id issued by the app")
    language: str | None = Field(None, pattern="^(en|sw)$")


class Resolve(BaseModel):
    note: str = Field(..., max_length=500)


def _notify(esc_id: str) -> None:
    url = os.environ.get("ESCALATION_WEBHOOK")
    if not url:
        return
    body = json.dumps({"escalation_id": esc_id}).encode()   # never the message

    def go():
        try:
            urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}),
                                   timeout=5)
        except Exception as e:  # noqa: BLE001
            log.error("escalation webhook failed: %s", e)

    threading.Thread(target=go, daemon=True).start()


def _require_counsellor(key: str | None) -> None:
    expected = os.environ.get("COUNSELLOR_API_KEY")
    if not expected or not key or not hmac.compare_digest(key, expected):
        raise HTTPException(401, "counsellor key required")


@app.post("/v1/checkin")
def checkin(body: CheckIn):
    p: TriagePipeline = STATE["pipeline"]
    r = p(body.text, body.language)
    if r.crisis:
        STATE["escalations"].add(r, body.student_ref)
        _notify(r.request_id)
    STATE["audit"].write(r, p.versions)
    if STATE["vocab"] is not None:
        STATE["escalations"].count_oov(r.pseudonymised_text, STATE["vocab"])
    return {**r.public(), "versions": p.versions}


@app.get("/v1/lexicon")
def lexicon():
    return STATE["valve"].export_for_mobile()


@app.get("/v1/version")
def version():
    return {**STATE["pipeline"].versions, "model_dir": STATE["model_dir"], "warnings": STATE["warnings"]}


@app.get("/health")
def health():
    return {"ok": True, "warnings": STATE.get("warnings", [])}


@app.get("/v1/escalations")
def escalations(status: str = "open", x_counsellor_key: str | None = Header(None)):
    _require_counsellor(x_counsellor_key)
    return STATE["escalations"].list(status)


@app.post("/v1/escalations/{esc_id}/resolve")
def resolve_escalation(esc_id: str, body: Resolve, x_counsellor_key: str | None = Header(None)):
    _require_counsellor(x_counsellor_key)
    if not STATE["escalations"].resolve(esc_id, body.note):
        raise HTTPException(404, "no open escalation with that id")
    return {"ok": True}


@app.get("/v1/drift/oov")
def oov(x_counsellor_key: str | None = Header(None)):
    _require_counsellor(x_counsellor_key)
    return STATE["escalations"].oov_report(load_config()["serving"]["oov_min_requests"])
