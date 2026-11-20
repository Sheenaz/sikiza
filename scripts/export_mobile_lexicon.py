"""Write the safety-valve lexicon + crisis response as JSON for the app's offline fallback.

The YAML files stay the single source of truth; the app bundles these JSON files and
refreshes the lexicon from GET /v1/lexicon when online.

    python scripts/export_mobile_lexicon.py
"""
from __future__ import annotations

import json

import yaml

from _common import ROOT, load_config, resolve

from triage.safety_valve import default_valve


def main():
    out = ROOT / "mobile" / "src" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    (out / "lexicon.json").write_text(json.dumps(default_valve().export_for_mobile(), indent=1))

    lib = yaml.safe_load(resolve(load_config()["retrieval"]["library"]).read_text())
    c = lib["crisis_response"]
    offline = {
        "library_version": lib["version"],
        "approved": bool(c.get("approved")),
        "crisis": {"id": c["id"], "en": c["en"].strip(), "sw": c["sw"].strip(),
                   "hotlines": [h for h in c.get("hotlines", []) if h["number"] != "TODO"]},
        # shown when offline and no crisis keyword matched: grounding exercise from the library
        "fallback": next({"id": s["id"], "en": s["en"].strip(), "sw": s["sw"].strip()}
                         for s in lib["strategies"] if s["id"] == "soc_grounding"),
    }
    (out / "offline.json").write_text(json.dumps(offline, indent=1, ensure_ascii=False))
    print(f"-> {out}/lexicon.json, offline.json")


if __name__ == "__main__":
    main()
