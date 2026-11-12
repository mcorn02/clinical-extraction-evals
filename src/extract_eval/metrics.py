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


def aggregate(per_note: list[dict[str, Counts]], tags: list[list[str]] | None = None) -> dict:
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
