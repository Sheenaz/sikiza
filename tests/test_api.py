"""API tests against the baseline bundle (run scripts/train_baseline.py --demo first)."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "artifacts" / "baseline" / "manifest.json"

pytestmark = pytest.mark.skipif(not BASELINE.exists(), reason="train the baseline first")


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("TRIAGE_MODEL_DIR", str(BASELINE.parent))
    monkeypatch.setenv("ALLOW_DRAFT_LIBRARY", "1")
    monkeypatch.setenv("COUNSELLOR_API_KEY", "test-key")
    from triage.config import load_config
    cfg = load_config()
    monkeypatch.setitem(cfg["serving"], "audit_log", str(tmp_path / "log.jsonl"))
    monkeypatch.setitem(cfg["serving"], "escalation_db", str(tmp_path / "esc.sqlite"))
    from serving.app import app
    with TestClient(app) as c:
        yield c, tmp_path


def test_crisis_message_escalates_and_logs_no_text(client):
    c, tmp = client
    msg = "nataka kujiua, call me 0712345678"
    r = c.post("/v1/checkin", json={"text": msg, "student_ref": "anon-42"}).json()
    assert r["crisis"] is True and r["hotlines"]
    log = (tmp / "log.jsonl").read_text()
    assert "kujiua" not in log and "0712345678" not in log      # never the message
    rec = json.loads(log.strip().splitlines()[-1])
    assert "valve" in rec["crisis_source"] and rec["versions"]["lexicon"]

    assert c.get("/v1/escalations").status_code == 401
    q = c.get("/v1/escalations", headers={"X-Counsellor-Key": "test-key"}).json()
    assert q and q[0]["student_ref"] == "anon-42"
    assert "0712345678" not in q[0]["pseudonymised_text"]       # pseudonymised before storage
    ok = c.post(f"/v1/escalations/{q[0]['id']}/resolve", json={"note": "called, safe"},
                headers={"X-Counsellor-Key": "test-key"})
    assert ok.status_code == 200


def test_non_crisis_gets_library_response(client):
    c, _ = client
    r = c.post("/v1/checkin", json={"text": "niko poa leo, nimechill tu"}).json()
    assert r["response_id"] and r["response_text"]
    assert r["versions"]["library"]


def test_lexicon_endpoint(client):
    c, _ = client
    lex = c.get("/v1/lexicon").json()
    assert lex["version"] and any(r["tier"] == "crisis" for r in lex["rules"])


def test_chat_page_is_served(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "/v1/checkin" in r.text and "1199" in r.text    # posts to the API; shows the hotline
