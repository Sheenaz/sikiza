import numpy as np
import pandas as pd
import pytest

from triage.augment import augment_frame
from triage.calibration import Calibrator
from triage.data import assemble, make_splits
from triage.pipeline import decide
from triage.pseudonymise import pseudonymise
from triage.retrieval import CopingLibrary
from triage.safety_valve import SafetyValve
from triage.text import normalise
from triage.thresholds import bayes_threshold, choose_threshold, recall_threshold

valve = SafetyValve()


@pytest.mark.parametrize("text", [
    "I want to kill myself", "nataka kujiua", "Nitajiua leo", "nimefikiria kujinyonga",
    "niko done na life", "nitajihang", "sitaki kuishi tena", "thinking about SUICIDE",
    "kujitoa", "nitakunywa sumu", "I just want to sleep and never wake up",
    "I don't want to live anymore", "sitaki kujiua",   # negation still escalates (by design)
])
def test_valve_catches_explicit_crisis(text):
    assert valve.check(text).crisis_hit


@pytest.mark.parametrize("text", [
    "exams zimeniweza manze", "sina doh ya fare", "niko poa leo", "I have a presentation tomorrow",
    "nimehang na assignment",   # ambiguous Sheng -> distress tier only
])
def test_valve_no_escalation_on_ordinary_distress(text):
    assert not valve.check(text).crisis_hit


def test_hang_is_distress_signal():
    r = valve.check("nime-hang na hii assignment")
    assert "d_hang" in r.distress_rules and not r.crisis_hit


def test_normalise_sheng_spellings():
    assert normalise("Nime-Hang SANAAAA") == "nimehang sanaa"
    assert normalise("don't") == "dont"


def test_or_gate_valve_overrides_low_score():
    crisis, src = decide(valve_hit=True, p_crisis=0.0001, threshold=0.01)
    assert crisis and src == ["valve"]
    crisis, src = decide(valve_hit=False, p_crisis=0.5, threshold=0.01)
    assert crisis and src == ["classifier"]
    assert decide(False, 0.001, 0.01) == (False, [])


def test_pseudonymise_removes_identifiers():
    out = pseudonymise("My name is Wanjiru, 0712 345 678, a@b.co, adm 123456, code QJK7ABC123")
    for leaked in ["Wanjiru", "0712", "a@b.co", "123456", "QJK7ABC123"]:
        assert leaked not in out


def test_pseudonymise_keeps_distress_words():
    # names that are also words must not be redacted (gazetteer excludes them)
    assert "hope" in pseudonymise("I have no hope and no faith left").lower()


def test_threshold_cost_and_recall_floor():
    assert bayes_threshold(100, 1) == pytest.approx(1 / 101)
    p = np.array([0.001, 0.002, 0.5, 0.9, 0.003, 0.004])
    y = np.array([1, 0, 1, 1, 0, 0])
    t = recall_threshold(p, y, 0.95)
    assert t == pytest.approx(0.001)          # must include the lowest-scored crisis
    res = choose_threshold(p, y, 100, 1, 0.95)
    assert res["threshold"] <= res["t_cost"] and res["calib_recall"] == 1.0


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    z = (y * 2 - 1) * 1.0 + rng.normal(0, 1, 500)
    cal = Calibrator.fit(z * 5, y, np.zeros((500, 5)), np.zeros(500, int))
    assert cal.t_crisis > 2.5   # recovers the x5 over-confidence


def _seed(n=80):
    cats = ["academic_stress", "financial_strain", "social_anxiety", "low_mood", "other"]
    return pd.DataFrame({
        "seed_id": [f"s{i}" for i in range(n)],
        "text": [f"message number {i} about exams na pesa" for i in range(n)],
        "category": [cats[i % 5] for i in range(n)],
        "crisis": [1 if i % 10 == 0 else 0 for i in range(n)],
        "group": ["code_mixed"] * n,
    })


def test_grouped_split_has_no_augmentation_leakage():
    seed = _seed()
    splits = make_splits(seed)
    aug = augment_frame(seed, n_per_seed=3, rng=np.random.default_rng(0))  # augment EVERYTHING on purpose
    frames = assemble(seed, splits, aug)
    held_out = set(splits["calib"]) | set(splits["test"])
    assert not frames["train"]["seed_id"].isin(held_out).any()
    assert not frames["test"]["is_augmented"].any() and not frames["calib"]["is_augmented"].any()
    assert set(frames["test"]["seed_id"]) == set(splits["test"])


def test_augmentation_preserves_crisis_phrases():
    seed = pd.DataFrame({"seed_id": ["a"], "text": ["nataka kujiua, sina pesa"], "category": ["low_mood"],
                         "crisis": [1]})
    aug = augment_frame(seed, n_per_seed=10, rng=np.random.default_rng(1))
    assert len(aug) > 0
    assert all(valve.check(t).crisis_hit for t in aug["text"])


def test_library_refuses_unapproved_entries():
    with pytest.raises(RuntimeError):
        CopingLibrary(allow_draft=False)


def test_library_retrieval_is_deterministic_and_in_category():
    lib = CopingLibrary(allow_draft=True)
    a = lib.retrieve("sina doh ya fees", "financial_strain")
    b = lib.retrieve("sina doh ya fees", "financial_strain")
    assert a.id == b.id and a.category == "financial_strain"
    r, hotlines = lib.crisis_response("sw")
    assert r.language == "sw" and hotlines


def test_drift_surfaces_new_slang():
    from triage.drift import drift_report

    train = [f"exams zimeniweza sana leo number {i}" for i in range(40)] + \
            [f"sina pesa ya fees hii sem {i}" for i in range(40)]
    new = train[:50] + ["wueh mafala wamenibana na fees"] * 5 + ["wueh nimedungwa na mafala"] * 5
    rep = drift_report(train, new, set(), min_token_count=3)
    terms = {t for c in rep["clusters"] for t, _ in c["candidate_new_terms"]}
    assert {"wueh", "mafala"} <= terms and rep["novel_share"] >= 0.1


def test_placeholder_hotlines_never_sent():
    _, hotlines = CopingLibrary(allow_draft=True).crisis_response("en")
    assert hotlines and all(any(c.isdigit() for c in h["number"]) for h in hotlines)
