"""Dump valve decisions on demo data + edge cases, for the mobile parity check."""
import json

import pandas as pd
from _common import resolve

from triage.safety_valve import default_valve
from triage.text import normalise

EXTRA = ["Nime-Hang SANAAAA", "I wanna d1e", "don't want to live", "nataka KUJIUA!!!", "kujitoa uhai",
         "niko done na life", "l1fe iishe", "café stress", "I'm tired of life fr", "Sitaki kuamka kesho"]
v = default_valve()
texts = pd.read_csv(resolve("data/demo/demo_seed.csv"))["text"].tolist() + EXTRA
print(json.dumps([{"text": t, "norm": normalise(t), "rules": v.check(t).crisis_rules} for t in texts]))
