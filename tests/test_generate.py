"""Label integrity: if these fail, the dataset is wrong and every score built on it is wrong too."""

import re

from extract_eval.generate import generate
from extract_eval.vocab import load_vocab

VOCAB = load_vocab()
RECORDS = generate(n=200, seed=123)


def _mentioned(text: str, field: str, canonical: str) -> bool:
    return any(
        re.search(rf"(?<![\w-]){re.escape(form)}(?![\w-])", text, re.IGNORECASE)
        for form in VOCAB.surface_forms(field, canonical)
    )


def test_same_seed_same_data():
    assert [r.text for r in generate(5, seed=7)] == [r.text for r in generate(5, seed=7)]


def test_every_gold_item_appears_in_the_text():
    for r in RECORDS:
        for field in ["conditions", "medications", "allergies", "referrals"]:
            for item in getattr(r.gold, field):
                assert _mentioned(r.text, field, item), f"{r.id}: {field}={item!r} not in text"


def test_gold_lists_have_no_duplicates():
    for r in RECORDS:
        for field in ["conditions", "medications", "allergies", "referrals"]:
            values = getattr(r.gold, field)
            assert len(values) == len(set(values)), r.id


def test_every_edge_case_shows_up():
    tags = {t for r in RECORDS for t in r.tags}
    for t in ["negation", "family_history", "rule_out", "discontinued_med", "declined_referral", "brand_name"]:
        assert t in tags
