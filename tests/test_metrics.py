from extract_eval.metrics import aggregate, check_regression, oov_predictions, score_note
from extract_eval.schema import NoteExtraction

GOLD = NoteExtraction(
    visit_date="2026-03-14",
    visit_type="follow_up",
    patient_age=9,
    conditions=["autism spectrum disorder", "asthma"],
    medications=["sertraline"],
    allergies=[],
    referrals=["speech therapy"],
    follow_up_weeks=6,
)


def test_perfect_prediction_scores_one():
    m = aggregate([score_note(GOLD, GOLD)])
    assert m["overall"]["f1"] == 1.0
    assert m["exact_match_rate"] == 1.0


def test_aliases_brands_and_doses_normalize():
    pred = GOLD.model_copy(update={
        "conditions": ["ASD", "Asthma"],
        "medications": ["Zoloft 50 mg daily"],
        "referrals": ["SLP"],
    })
    m = aggregate([score_note(pred, GOLD)])
    assert m["overall"]["f1"] == 1.0


def test_false_positive_and_false_negative_counted():
    pred = GOLD.model_copy(update={"conditions": ["autism spectrum disorder", "epilepsy"]})
    s = score_note(pred, GOLD)["conditions"]
    assert (s.tp, s.fp, s.fn) == (1, 1, 1)


def test_wrong_scalar_is_one_fp_and_one_fn():
    pred = GOLD.model_copy(update={"follow_up_weeks": 4})
    s = score_note(pred, GOLD)["follow_up_weeks"]
    assert (s.tp, s.fp, s.fn) == (0, 1, 1)


def test_failed_output_counts_everything_as_missed():
    s = score_note(None, GOLD)
    assert all(c.tp == 0 and c.fp == 0 for c in s.values())
    assert s["conditions"].fn == 2


def test_regression_check_flags_drops_only():
    ref = {"fields": {"conditions": {"f1": 0.95}, "medications": {"f1": 0.90}}}
    new = {"fields": {"conditions": {"f1": 0.94}, "medications": {"f1": 0.80}}}
    problems = check_regression(new, ref, max_drop=0.02)
    assert len(problems) == 1 and problems[0].startswith("medications")


def test_out_of_vocab_values_are_flagged_but_still_strict_false_positives():
    pred = GOLD.model_copy(update={"conditions": ["ASD", "asthma", "eczema"]})
    assert oov_predictions(pred, GOLD)["conditions"] == ["eczema"]
    assert oov_predictions(GOLD, GOLD) == {f: [] for f in ["conditions", "medications", "allergies", "referrals"]}
    m = aggregate([score_note(pred, GOLD)], oov=[oov_predictions(pred, GOLD)])
    assert m["fields"]["conditions"]["fp"] == 1  # strict: unchanged
    assert m["fields"]["conditions"]["precision"] == 2 / 3
    assert m["oov"]["fields"]["conditions"] == {"oov_count": 1, "precision_if_oov_excluded": 1.0}
    assert m["oov"]["total"] == 1 and m["oov"]["notes_with_oov"] == 1


def test_known_wrong_value_is_a_fp_not_oov():
    pred = GOLD.model_copy(update={"conditions": ["autism spectrum disorder", "epilepsy"]})
    assert oov_predictions(pred, GOLD)["conditions"] == []
