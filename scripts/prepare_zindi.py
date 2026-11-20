"""Build the seed dataset from the Zindi BNBR Kenya Tech4MentalHealth data.

Inputs (committed):
    data/zindi/Train.csv, data/zindi/Test.csv    original competition files (ID, text[, label])
    data/labels/zindi_labels.csv                 Sikiza labels per ID (see data/labels/LABELLING_GUIDE.md)
Outputs:
    data/zindi/zindi_seed.csv                    seed_id, text, category, crisis, language, ... (config data.seed_csv)
    artifacts/reports/zindi_data.json            dataset profile + label-noise analysis vs the original Zindi labels

Real-data issues handled here:
* Duplicates: 925 rows hold 884 distinct texts after normalisation, and 13 texts appear in BOTH
  Zindi's train and test files. Identical texts are merged into one seed (crisis if any copy is a
  crisis, else the most frequent category), so a message can never sit on both sides of a split.
* Zindi's Test.csv has no labels; both files are labelled from scratch with one guide.
* The original Zindi topic label is kept (zindi_label) to measure how much it disagrees with
  text-based crisis labels.

    python scripts/prepare_zindi.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage.config import CATEGORIES, load_config, resolve  # noqa: E402
from triage.data import validate  # noqa: E402
from triage.text import normalise  # noqa: E402


def cohen_kappa(a, b) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a, b))


def main() -> None:
    zdir = resolve("data/zindi")
    tr = pd.read_csv(zdir / "Train.csv").assign(zindi_file="train")
    te = pd.read_csv(zdir / "Test.csv").assign(zindi_file="test", label=None)
    raw = pd.concat([tr, te], ignore_index=True).rename(columns={"label": "zindi_label"})
    raw["text"] = raw["text"].astype(str).str.strip()

    labels = pd.read_csv(resolve("data/labels/zindi_labels.csv"))
    missing = set(raw["ID"]) - set(labels["ID"])
    if missing:
        raise SystemExit(f"{len(missing)} messages have no label, e.g. {sorted(missing)[:5]}")
    labels = labels.drop(columns=["text", "zindi_label"], errors="ignore")
    bad = set(labels["category"]) - set(CATEGORIES)
    if bad:
        raise SystemExit(f"unknown categories {bad}; expected {CATEGORIES}")
    df = raw.merge(labels, on="ID", how="left")
    df["norm"] = df["text"].map(normalise)
    df = df[df["norm"].str.len() > 0]

    # ---- merge duplicate texts into one seed -------------------------------------------------
    groups = df.groupby("norm", sort=False)
    conflicts = int((groups["crisis"].nunique() > 1).sum() + (groups["category"].nunique() > 1).sum())
    seeds = groups.agg(
        text=("text", "first"),
        crisis=("crisis", "max"),
        category=("category", lambda s: s.mode().iloc[0]),
        n_copies=("ID", "size"),
        zindi_ids=("ID", lambda s: " ".join(s)),
        zindi_label=("zindi_label", lambda s: s.dropna().mode().iloc[0] if s.notna().any() else ""),
        zindi_file=("zindi_file", lambda s: "+".join(sorted(set(s)))),
        reviewed=("reviewed", "min"),
        crisis_ai=("crisis_ai", "max"),
        labelled_by=("labelled_by", "first"),
        **({"category_2": ("category_2", "first"), "crisis_2": ("crisis_2", "max")}
           if {"category_2", "crisis_2"}.issubset(df.columns) else {}),
    ).reset_index()
    seeds["seed_id"] = "z-" + seeds["norm"].map(lambda t: hashlib.sha1(t.encode()).hexdigest()[:10])
    seeds["language"] = "en"
    cols = ["seed_id", "text", "category", "crisis", "language", "zindi_label", "zindi_file",
            "n_copies", "zindi_ids", "labelled_by", "reviewed", "crisis_ai"]
    # optional second, independent coder (a sample of rows) -> Cohen's kappa in the evaluation
    cols += [c for c in ("category_2", "crisis_2") if c in seeds.columns]
    seeds = validate(seeds[cols], "zindi_seed")

    out = resolve(load_config()["data"]["seed_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    seeds.to_csv(out, index=False)

    # ---- profile + label noise vs the original Zindi topic labels ----------------------------
    lab = df[df["zindi_label"].notna()]
    z_crisis = (lab["zindi_label"] == "Suicide").astype(int)
    prof = {
        "rows_raw": int(len(raw)), "rows_labelled_by_zindi": int(raw["zindi_label"].notna().sum()),
        "distinct_texts": int(len(seeds)), "duplicate_rows_merged": int(len(df) - len(seeds)),
        "texts_in_both_zindi_train_and_test": int((seeds["zindi_file"] == "test+train").sum()),
        "label_conflicts_within_duplicates": conflicts,
        "seed_crisis": int(seeds["crisis"].sum()), "seed_crisis_rate": float(seeds["crisis"].mean()),
        "seed_categories": seeds["category"].value_counts().to_dict(),
        "reviewed_rows": int(seeds["reviewed"].sum()),
        # the reviewer saw the AI suggestions, so this is review agreement, not independent inter-annotator kappa
        "ai_first_pass_vs_human_review_crisis": {
            "kappa": cohen_kappa(df["crisis_ai"], df["crisis"]),
            "ai_0_human_1": int(((df["crisis_ai"] == 0) & (df["crisis"] == 1)).sum()),
            "ai_1_human_0": int(((df["crisis_ai"] == 1) & (df["crisis"] == 0)).sum()),
            "rows": int(len(df)),
        } if "crisis_ai" in df.columns else None,
        "zindi_label_vs_text_crisis": {
            "kappa_zindi_suicide_vs_crisis": cohen_kappa(z_crisis, lab["crisis"]),
            "zindi_suicide_rows": int(z_crisis.sum()),
            "zindi_suicide_but_not_crisis": int(((z_crisis == 1) & (lab["crisis"] == 0)).sum()),
            "crisis_but_not_zindi_suicide": int(((z_crisis == 0) & (lab["crisis"] == 1)).sum()),
            "crisis_by_zindi_label": lab.groupby("zindi_label")["crisis"].sum().astype(int).to_dict(),
        },
        "zindi_label_x_category": pd.crosstab(lab["zindi_label"], lab["category"]).to_dict(orient="index"),
    }
    rep = resolve("artifacts/reports")
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "zindi_data.json").write_text(json.dumps(prof, indent=2))
    print(json.dumps(prof, indent=2))
    print(f"\nseed -> {out} ({len(seeds)} messages)")
    if prof["reviewed_rows"] < len(seeds):
        print(f"!! {len(seeds) - prof['reviewed_rows']} labels are still unreviewed AI suggestions "
              f"(data/labels/zindi_labels.csv, column reviewed).")


if __name__ == "__main__":
    main()
