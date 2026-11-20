"""Prediction audit log, human-escalation queue, and privacy-preserving OOV counter.

* Audit log (JSONL): versions + decision + scores. NEVER the raw or pseudonymised text.
* Escalation queue (SQLite): what a counsellor needs to act — the pseudonymised text,
  an app-provided pseudonymous student_ref, scores and triggering rules. Access is
  restricted by API key; retention must follow the ethics approval.
* OOV counter: tokens unseen in training, counted by number of distinct requests;
  only surfaced to the drift report once seen in >= k requests (k-anonymity floor).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from .text import tokens


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, result, versions: dict) -> None:
        rec = {
            "ts": time.time(), "request_id": result.request_id, "versions": versions,
            "crisis": result.crisis, "crisis_source": result.crisis_source,
            "crisis_prob": result.crisis_prob, "category": result.category,
            "category_prob": result.category_prob, "valve_rules": result.valve_rules,
            "distress_rules": result.distress_rules, "response_id": result.response_id,
            "degraded": result.degraded, "latency_ms": round(result.latency_ms, 2),
        }
        with self._lock, open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")


class EscalationStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS escalations (
                id TEXT PRIMARY KEY, created_at REAL, student_ref TEXT, crisis_prob REAL,
                sources TEXT, valve_rules TEXT, pseudonymised_text TEXT,
                status TEXT DEFAULT 'open', resolved_at REAL, resolution_note TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS oov (
                token TEXT PRIMARY KEY, requests INTEGER, first_seen REAL, last_seen REAL)""")

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10)

    def add(self, result, student_ref: str | None) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO escalations (id, created_at, student_ref, crisis_prob, sources, "
                      "valve_rules, pseudonymised_text) VALUES (?,?,?,?,?,?,?)",
                      (result.request_id, time.time(), student_ref, result.crisis_prob,
                       json.dumps(result.crisis_source), json.dumps(result.valve_rules),
                       result.pseudonymised_text))

    def list(self, status: str = "open") -> list[dict]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            rows = c.execute("SELECT * FROM escalations WHERE status=? ORDER BY created_at",
                             (status,)).fetchall()
        return [dict(r) for r in rows]

    def resolve(self, esc_id: str, note: str) -> bool:
        with self._conn() as c:
            cur = c.execute("UPDATE escalations SET status='resolved', resolved_at=?, resolution_note=? "
                            "WHERE id=? AND status='open'", (time.time(), note, esc_id))
            return cur.rowcount == 1

    # ---- OOV counter ---------------------------------------------------------
    def count_oov(self, text: str, vocab: set[str]) -> None:
        new = {t for t in tokens(text) if t not in vocab and t.isalpha() and len(t) > 2}
        if not new:
            return
        now = time.time()
        with self._conn() as c:
            for t in new:
                c.execute("INSERT INTO oov VALUES (?,1,?,?) ON CONFLICT(token) DO UPDATE SET "
                          "requests=requests+1, last_seen=excluded.last_seen", (t, now, now))

    def oov_report(self, min_requests: int) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT token, requests, first_seen, last_seen FROM oov WHERE requests>=? "
                             "ORDER BY requests DESC", (min_requests,)).fetchall()
        return [dict(zip(["token", "requests", "first_seen", "last_seen"], r)) for r in rows]
