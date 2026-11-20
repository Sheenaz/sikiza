"""Generate a SYNTHETIC demo seed set so the pipeline can run before real data exists.

These rows are template-built, NOT real student messages, and any metric computed on
them is meaningless. They exist to smoke-test code paths only. Replace with
data/raw/strathmore_checkins.csv (same schema) for real work.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from triage.config import resolve  # noqa: E402

OPEN = {
    "en": ["", "Honestly ", "Not gonna lie, ", "Today ", "Lately "],
    "sw": ["", "Kusema kweli ", "Siku hizi ", "Leo ", "Manze "],
}

CORE = {
    "academic_stress": {
        "en": ["I have three CATs this week and I'm so behind on revision",
               "I failed my stats exam and I'm scared of being discontinued",
               "the assignments are too much, I can't keep up with deadlines",
               "my GPA dropped and I don't know how to fix it before finals"],
        "sw": ["nina CAT tatu wiki hii na sijasoma chochote",
               "nimefeli mtihani wa stats na naogopa nitafukuzwa",
               "assignments ni mingi sana siwezi maliza kwa deadline",
               "nimehang na project yangu, lecturer hajibu emails"],
        "sheng": ["exams zimeniweza manze, sijai soma kitu",
                  "hii sem stress imeniweza, CATs ni mob",
                  "nimehang na hii assignment ya coding, deadline ni kesho",
                  "supp ya last sem bado inanikula akili"],
    },
    "financial_strain": {
        "en": ["I can't pay my fee balance and they won't give me an exam card",
               "my HELB hasn't come and rent is due on Friday",
               "I'm skipping meals so I can afford fare to campus",
               "my parents lost their jobs and I don't know how I'll stay in school"],
        "sw": ["sina pesa ya karo na exam card imezuiliwa",
               "HELB haijaingia na kodi ya nyumba inadaiwa",
               "nakosa chakula ili nipate nauli ya kuja shule",
               "wazazi wamepoteza kazi na sijui nitamaliza shule aje"],
        "sheng": ["sina doh ya fare hii wiki, nitakam campo aje",
                  "keja rent iko due na ganji imeisha",
                  "fees balance inanisumbua, sina chapaa kabisa",
                  "budaa hana doh na mathe anasema nitafute side hustle"],
    },
    "social_anxiety": {
        "en": ["I have a presentation tomorrow and my heart is racing just thinking about it",
               "I feel like everyone in class is judging me when I speak",
               "I don't have friends here and I'm too shy to join any group",
               "I panic in crowded places like the cafeteria"],
        "sw": ["nina presentation kesho na moyo unadunda sana",
               "nahisi kila mtu darasani ananihukumu nikiongea",
               "sina marafiki hapa na naogopa kujiunga na kikundi",
               "naogopa sana kuongea mbele ya watu"],
        "sheng": ["kesho niko na presentation na niko na wasiwasi mob",
                  "sina mbogi hapa campo, kila mtu ako na group yake",
                  "nikiongea class nafeel kila msee ananicheka",
                  "nashtuka nikiwa kwa crowd, siwezi hata kukula cafeteria"],
    },
    "low_mood": {
        "en": ["I feel empty and I've been crying every night this week",
               "nothing feels enjoyable anymore, I just stay in bed",
               "I feel so alone even when I'm with people",
               "I have no energy to do anything and I don't know why"],
        "sw": ["najisikia mtupu na nimekuwa nikilia kila usiku",
               "hakuna kitu kinanifurahisha tena, nalala tu",
               "najihisi mpweke hata nikiwa na watu",
               "sina nguvu ya kufanya chochote"],
        "sheng": ["niko down sana hii wiki, sitaki kuongea na mtu",
                  "nimeboeka na kila kitu, nakaa tu kwa keja",
                  "niko low manze, hata food sitaki",
                  "life imekuwa boring mbaya, sina morale"],
    },
    "other": {
        "en": ["just checking in, today was okay",
               "my roommate and I had a fight about cleaning",
               "I broke up with my girlfriend last week and it's weird",
               "I'm fine, just wanted to log my mood"],
        "sw": ["nimeingia tu kuripoti, leo ilikuwa sawa",
               "mimi na roommate tumegombana juu ya usafi",
               "niko sawa, nataka tu kuandika hisia zangu",
               "familia yangu inanipigia simu kila saa"],
        "sheng": ["niko poa leo, nimechill tu",
                  "mimi na dem tumeachana, ni sawa tu",
                  "leo ilikuwa fiti, nimeenda gym",
                  "roommate wangu ni noma, anaongea usiku mzima"],
    },
}

# Crisis rows: (text, category, language). Some contain valve keywords, some are
# IMPLICIT warning signs that only the classifier can catch — that is why the
# decision is an OR-gate.
CRISIS = [
    ("I want to kill myself, I can't do this anymore", "low_mood", "en"),
    ("I've been thinking about suicide every day", "low_mood", "en"),
    ("everyone would be better off without me", "low_mood", "en"),
    ("I failed again, I don't want to live anymore", "academic_stress", "en"),
    ("I've written letters to my family and given away my laptop, I won't need it", "low_mood", "en"),
    ("I've been saving my pills, tonight is the night", "low_mood", "en"),
    ("there's no way out of this debt, I'm going to end it all", "financial_strain", "en"),
    ("I keep cutting myself to feel something", "low_mood", "en"),
    ("nataka kujiua, maisha haina maana", "low_mood", "sw"),
    ("sitaki kuishi tena, nimechoka na maisha", "low_mood", "sw"),
    ("nimefikiria kujinyonga usiku huu", "low_mood", "sw"),
    ("nimeandika barua za kuaga kwa familia yangu", "low_mood", "sw"),
    ("nimekusanya vidonge vingi, nitavimeza vyote leo", "low_mood", "sw"),
    ("deni ni nyingi sana, nataka kufa tu", "financial_strain", "sw"),
    ("nimefeli tena, ni bora nijiue", "academic_stress", "sw"),
    ("niko done na life manze, nitajihang", "low_mood", "sheng"),
    ("hii stress imeniweza, nataka kujitoa", "academic_stress", "sheng"),
    ("sioni sababu ya kuamka kesho, nimewapa mbogi vitu zangu zote", "low_mood", "sheng"),
    ("sina doh, fees haijalipwa, nitakunywa sumu", "financial_strain", "sheng"),
    ("life iishe tu, siwezi tena", "low_mood", "sheng"),
    ("I just want to sleep and never wake up", "low_mood", "en"),
    ("nimeacha kula siku tatu, sitaki kuendelea kuwa hapa", "low_mood", "sw"),
    ("tonight I'll go to the rooftop and that's it", "low_mood", "en"),
    ("sitaki kuwa burden kwa mathe tena, ni bora niondoke milele", "financial_strain", "sheng"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo/demo_seed.csv")
    ap.add_argument("--per-category", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = []
    for cat, by_lang in CORE.items():
        combos = [(lang, core, op) for lang, cores in by_lang.items() for core in cores
                  for op in OPEN["en" if lang == "en" else "sw"]]
        pick = rng.choice(len(combos), size=min(args.per_category, len(combos)), replace=False)
        for i in pick:
            lang, core, op = combos[i]
            text = (op + core).strip()
            rows.append({"text": text[0].upper() + text[1:], "category": cat, "crisis": 0,
                         "language": lang})
    for text, cat, lang in CRISIS:
        rows.append({"text": text, "category": cat, "crisis": 1, "language": lang})

    df = pd.DataFrame(rows).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df.insert(0, "seed_id", [f"demo-{i:04d}" for i in range(len(df))])
    df["annotator"] = "synthetic"
    df["source"] = "synthetic_demo"

    # simulate a double-coded 30% sample with ~12% category disagreement (for kappa)
    m = rng.random(len(df)) < 0.30
    df["category_2"] = None
    df["crisis_2"] = None
    cats = list(CORE)
    for i in np.where(m)[0]:
        c = df.at[i, "category"]
        df.at[i, "category_2"] = rng.choice([x for x in cats if x != c]) if rng.random() < 0.12 else c
        df.at[i, "crisis_2"] = df.at[i, "crisis"] if rng.random() > 0.03 else 1 - df.at[i, "crisis"]

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} SYNTHETIC rows ({df.crisis.sum()} crisis) -> {out}")


if __name__ == "__main__":
    main()
