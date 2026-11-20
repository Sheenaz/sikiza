"""Evaluate one or more bundles on the held-out TEST split (real seed messages only).

    python scripts/evaluate.py --bundles artifacts/baseline artifacts/afroxlmr_torch [--demo]

The FIRST bundle is the reference for McNemar's test (normally the baseline).
Writes artifacts/reports/eval.json, eval.md and confusion-matrix PNGs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import DEMO_BANNER, load_config, load_frames, resolve

from triage.bundle import load_bundle
from triage.calibration import expected_calibration_error
from triage.evaluate import full_report, kappa, mcnemar
from triage.pipeline import batch_decisions
from triage.safety_valve import default_valve


def plot_cm(cm: dict, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.array(cm["matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(m, cmap="Greens")
    ax.set_xticks(range(len(cm["labels"])), cm["labels"], rotation=35, ha="right")
    ax.set_yticks(range(len(cm["labels"])), cm["labels"])
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, int(m[i, j]), ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fmt_ci(x, ci):
    return f"{x:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", nargs="+", required=True)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default="artifacts/reports")
    args = ap.parse_args()
    cfg = load_config()["evaluation"]
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames, seed, is_demo = load_frames(args.demo, use_augmented=False)
    test = frames["test"].reset_index(drop=True)
    assert not test.get("is_augmented", False).any(), "test must contain real seed messages only"
    valve = default_valve()

    results, correct = {}, {}
    for b in args.bundles:
        scorer, cal, man = load_bundle(resolve(b))
        thr = man["threshold"]["threshold"]
        d = batch_decisions(scorer, cal, thr, valve, test["text"].tolist())
        name = man["model_version"]
        rep = {
            "manifest": {k: man.get(k) for k in ("model_type", "model_version", "base_model", "threshold",
                                                 "calibrator", "lexicon_version")},
            "valve_only": full_report(test, d["valve"], d["category"], cfg["bootstrap_iterations"], cfg["ci"])["overall"]["crisis"],
            "classifier_only": full_report(test, d["classifier"], d["category"], cfg["bootstrap_iterations"], cfg["ci"]),
            "combined_or_gate": full_report(test, d["combined"], d["category"], cfg["bootstrap_iterations"], cfg["ci"]),
            "test_ece_crisis": expected_calibration_error(d["p_crisis"], test["crisis"].to_numpy()),
        }
        results[name] = rep
        correct[name] = {"crisis": d["combined"] == test["crisis"].to_numpy(),
                         "category": np.array(d["category"]) == test["category"].to_numpy()}
        plot_cm(rep["combined_or_gate"]["overall"]["category"]["confusion_matrix"], name,
                out / f"confusion_{name}.png")

    names = list(results)
    tests = {}
    for other in names[1:]:
        tests[f"{names[0]} vs {other}"] = {
            "crisis_decision": mcnemar(correct[names[0]]["crisis"], correct[other]["crisis"]),
            "category": mcnemar(correct[names[0]]["category"], correct[other]["category"]),
        }
    report = {"demo_data": is_demo, "test_rows": int(len(test)), "test_crisis": int(test["crisis"].sum()),
              "groups": test["group"].value_counts().to_dict(), "kappa": kappa(seed),
              "models": results, "mcnemar": tests}
    (out / "eval.json").write_text(json.dumps(report, indent=2, default=float))

    # ---- markdown summary -------------------------------------------------------
    L = []
    if is_demo:
        L.append("> **DEMO DATA — synthetic rows; numbers are meaningless.**\n")
    L.append(f"Test set: {len(test)} real seed messages, {int(test['crisis'].sum())} crisis; "
             f"groups {report['groups']}. Cohen's κ: {report['kappa']}\n")
    L.append("| model | gate | crisis recall | bootstrap 95% CI | exact 95% CI | precision | escalation rate | macro-F1 [95% CI] | recall ≥ 0.95 (lower bound) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    warns = []
    for n, r in results.items():
        rows = [(g, r[g]["overall"]["crisis"], r[g]["overall"]["category"])
                for g in ("classifier_only", "combined_or_gate")] + [("valve_only", r["valve_only"], None)]
        for gate, c, k in rows:
            ex = c["recall_exact_ci"]
            L.append(f"| {n} | {gate} | {c['recall']:.3f} | [{c['recall_ci'][0]:.3f}, {c['recall_ci'][1]:.3f}] | "
                     f"[{ex[0]:.3f}, {ex[1]:.3f}] | {c['precision']:.3f} | {c['escalation_rate']:.3f} | "
                     f"{fmt_ci(k['macro_f1'], k['macro_f1_ci']) if k else '–'} | "
                     f"{'yes' if c['meets_target_0.95'] else 'no'} |")
            if c.get("warning"):
                warns.append(f"- ⚠ {n} / {gate}: {c['warning']}")
    L += [""] + warns
    L.append("\n**English-only vs code-mixed (combined OR-gate)**\n")
    L.append("| model | group | n | crisis recall [95% CI] | macro-F1 |")
    L.append("|---|---|---|---|---|")
    for n, r in results.items():
        for g in ("english_only", "code_mixed"):
            if g in r["combined_or_gate"]:
                c = r["combined_or_gate"][g]["crisis"]; k = r["combined_or_gate"][g]["category"]
                L.append(f"| {n} | {g} | {c['n']} | {fmt_ci(c['recall'], c['recall_ci'])} | {k['macro_f1']:.3f} |")
    if tests:
        L.append("\n**McNemar (exact) vs reference**\n")
        for k_, v in tests.items():
            L.append(f"- {k_}: crisis p={v['crisis_decision']['p_value']:.4f} "
                     f"({v['crisis_decision']['a_only_correct']}/{v['crisis_decision']['b_only_correct']}), "
                     f"category p={v['category']['p_value']:.4f} "
                     f"({v['category']['a_only_correct']}/{v['category']['b_only_correct']})")
    (out / "eval.md").write_text("\n".join(L) + "\n")
    if is_demo:
        print(DEMO_BANNER)
    print("\n".join(L))
    print(f"\nreports -> {out}")


if __name__ == "__main__":
    main()
