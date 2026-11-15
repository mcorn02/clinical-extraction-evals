"""Field-level scoring.

Every field is treated as a set of values: list fields as-is, scalar fields as a set of 0 or 1
values. That gives one consistent definition of precision and recall for every field:

    TP = predicted and correct    FP = predicted but wrong    FN = in the gold but missed

Scores are micro-averaged across notes (counts are summed, then divided).
"""

from collections import defaultdict
from dataclasses import dataclass

from .schema import ALL_FIELDS, LIST_FIELDS, NoteExtraction
from .vocab import Vocab, load_vocab

# Which field each edge case is designed to trip up
TAG_FIELDS = {
    "negation": ["conditions"],
    "family_history": ["conditions"],
    "rule_out": ["conditions"],
    "alias": ["conditions", "allergies", "referrals"],
    "brand_name": ["medications"],
    "discontinued_med": ["medications"],
    "declined_referral": ["referrals"],
    "no_allergies": ["allergies"],
    "number_words": ["follow_up_weeks"],
    # hand-written adversarial notes (notes_v2)
    "resolved_condition": ["conditions"],
    "hx_but_active": ["conditions"],
    "suspected_dx": ["conditions"],
    "ruled_out_result": ["conditions"],
    "negation_contrast": ["conditions"],
    "negation_list": ["conditions"],
    "new_diagnosis": ["conditions"],
    "inferred_condition_trap": ["conditions"],
    "evaluate_for": ["conditions", "referrals"],
    "family_medication": ["medications"],
    "planned_med": ["medications"],
    "dose_change": ["medications"],
    "restarted_med": ["medications"],
    "self_discontinued": ["medications"],
    "switched_med": ["medications"],
    "list_with_status": ["medications"],
    "duplicate_surface": ["medications"],
    "prn_med": ["medications"],
    "nkda_food_allergy": ["allergies"],
    "allergy_refuted": ["allergies"],
    "allergy_negation": ["allergies"],
    "family_allergy": ["allergies"],
    "correction": ["allergies"],
    "conditional_referral": ["referrals"],
    "existing_referral": ["referrals"],
    "prior_referral": ["referrals"],
    "no_referral_needed": ["referrals"],
    "abbreviation_collision": ["referrals"],
    "date_distractor": ["visit_date"],
    "age_distractor": ["patient_age"],
    "unit_conversion": ["follow_up_weeks"],
    "followup_hedge": ["follow_up_weeks"],
}


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float | None:
        d = self.tp + self.fp
        return self.tp / d if d else None

    @property
    def recall(self) -> float | None:
        d = self.tp + self.fn
        return self.tp / d if d else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None if (p is None or r is None) else 0.0
        return 2 * p * r / (p + r)

    def as_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": self.precision, "recall": self.recall, "f1": self.f1}


def field_values(field: str, obj: NoteExtraction | None, vocab: Vocab) -> set:
    if obj is None:
        return set()
    value = getattr(obj, field)
    if field in LIST_FIELDS:
        return {vocab.normalize(field, v) for v in value}
    return set() if value is None else {value}


def score_note(pred: NoteExtraction | None, gold: NoteExtraction, vocab: Vocab | None = None) -> dict[str, Counts]:
    vocab = vocab or load_vocab()
    out = {}
    for field in ALL_FIELDS:
        p, g = field_values(field, pred, vocab), field_values(field, gold, vocab)
        out[field] = Counts(tp=len(p & g), fp=len(p - g), fn=len(g - p))
    return out


def oov_predictions(pred: NoteExtraction | None, gold: NoteExtraction, vocab: Vocab | None = None) -> dict[str, list[str]]:
    """Predicted list values the vocab can't map to a canonical name (and the gold doesn't contain).

    These are still scored as false positives in `score_note`, so the headline numbers stay strict.
    This just flags them so a human can grade each one: a hallucination, or a correct extraction
    the vocab doesn't cover yet?
    """
    vocab = vocab or load_vocab()
    out = {}
    for field in LIST_FIELDS:
        canonical = set(vocab.canonical_names(field))
        gold_vals = field_values(field, gold, vocab)
        out[field] = sorted(v for v in field_values(field, pred, vocab) if v not in canonical and v not in gold_vals)
    return out


def aggregate(
    per_note: list[dict[str, Counts]],
    tags: list[list[str]] | None = None,
    oov: list[dict[str, list[str]]] | None = None,
) -> dict:
    by_field = {f: Counts() for f in ALL_FIELDS}
    overall = Counts()
    exact = 0
    for scores in per_note:
        for f, c in scores.items():
            by_field[f].add(c)
            overall.add(c)
        if all(c.fp == 0 and c.fn == 0 for c in scores.values()):
            exact += 1

    result = {
        "n_notes": len(per_note),
        "exact_match_rate": exact / len(per_note) if per_note else None,
        "overall": overall.as_dict(),
        "fields": {f: c.as_dict() for f, c in by_field.items()},
    }

    if oov is not None:
        # Strict scoring counts every OOV value as a FP. Show what precision would be without them.
        by_oov = {}
        for f in LIST_FIELDS:
            n = sum(len(o[f]) for o in oov)
            c = by_field[f]
            d = c.tp + c.fp - n
            by_oov[f] = {"oov_count": n, "precision_if_oov_excluded": c.tp / d if d else None}
        total = sum(v["oov_count"] for v in by_oov.values())
        d = overall.tp + overall.fp - total
        result["oov"] = {
            "total": total,
            "notes_with_oov": sum(1 for o in oov if any(o.values())),
            "overall_precision_if_oov_excluded": overall.tp / d if d else None,
            "fields": by_oov,
        }

    # On notes containing each edge case, how does the field it targets hold up?
    if tags is not None:
        by_tag: dict[str, Counts] = defaultdict(Counts)
        n_by_tag: dict[str, int] = defaultdict(int)
        for scores, note_tags in zip(per_note, tags):
            for t in note_tags:
                n_by_tag[t] += 1
                for f in TAG_FIELDS.get(t, LIST_FIELDS):
                    by_tag[t].add(scores[f])
        result["by_edge_case"] = {
            t: {"n_notes": n_by_tag[t], "fields": TAG_FIELDS.get(t, LIST_FIELDS), **by_tag[t].as_dict()}
            for t in sorted(by_tag)
        }
    return result


def check_regression(new: dict, reference: dict, max_drop: float = 0.02) -> list[str]:
    """Return a message for every field whose F1 fell more than max_drop below the reference run."""
    problems = []
    for f, ref in reference["fields"].items():
        ref_f1, new_f1 = ref.get("f1"), new["fields"].get(f, {}).get("f1")
        if ref_f1 is None:
            continue
        if new_f1 is None or new_f1 < ref_f1 - max_drop:
            problems.append(f"{f}: F1 {ref_f1:.3f} -> {new_f1 if new_f1 is None else round(new_f1, 3)}")
    return problems
