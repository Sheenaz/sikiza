"""Latency: model-only (in-process) and full HTTP round trip.

    python scripts/benchmark_latency.py --bundle artifacts/model            # in-process
    python scripts/benchmark_latency.py --url http://localhost:8000          # HTTP round trip

The proposal's < 400 ms budget is a round trip from a mid-range Android phone; measure
that on a device with the mobile app's built-in timer (it shows ms per check-in).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

from _common import resolve

MSGS = ["exams zimeniweza manze sijai soma", "I can't pay my fees this sem", "niko down sana leo",
        "I have a presentation tomorrow and I'm panicking", "niko poa leo, nimechill tu"]


def summarise(ms):
    ms = sorted(ms)
    return {"n": len(ms), "p50": statistics.median(ms), "p95": ms[int(0.95 * (len(ms) - 1))], "max": ms[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle")
    ap.add_argument("--url")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    ms = []
    if args.bundle:
        from triage.bundle import load_bundle

        scorer, _, man = load_bundle(resolve(args.bundle))
        scorer.logits(MSGS[:1])  # warm-up
        for i in range(args.n):
            t = time.perf_counter(); scorer.logits([MSGS[i % len(MSGS)]]); ms.append((time.perf_counter() - t) * 1000)
        print(man["model_type"], "model-only ms:", json.dumps(summarise(ms)))
    if args.url:
        ms = []
        for i in range(args.n):
            body = json.dumps({"text": MSGS[i % len(MSGS)]}).encode()
            req = urllib.request.Request(args.url.rstrip("/") + "/v1/checkin", body,
                                         {"Content-Type": "application/json"})
            t = time.perf_counter(); urllib.request.urlopen(req).read(); ms.append((time.perf_counter() - t) * 1000)
        print("HTTP round-trip ms:", json.dumps(summarise(ms)))


if __name__ == "__main__":
    main()
