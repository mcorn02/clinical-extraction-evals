"""Load the vocabulary and map surface forms (aliases, brand names, doses) to canonical names."""

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VOCAB_PATH = ROOT / "data" / "vocab.json"

_DOSE_TAIL = re.compile(r"\s+\d[\d.,]*\s*(mg|mcg|g|ml|puffs?|units?)\b.*$", re.IGNORECASE)


@dataclass
class Vocab:
    raw: dict
    # field name -> {lowercase surface form -> canonical lowercase name}
    alias_map: dict[str, dict[str, str]] = field(default_factory=dict)

    def canonical_names(self, field_name: str) -> list[str]:
        return sorted(set(self.alias_map[field_name].values()))

    def surface_forms(self, field_name: str, canonical: str) -> list[str]:
        """Every way a canonical name can appear in text, as written in vocab.json."""
        canonical = canonical.lower()
        section = self.raw[field_name]
        for key, entry in section.items():
            if key.lower() != canonical:
                continue
            if field_name == "conditions":
                return [key, *entry["aliases"]]
            if field_name == "medications":
                return [key, *entry["brands"]]
            if field_name == "allergies":
                return [key, *entry]
            if field_name == "referrals":
                return [key, *entry["aliases"]]
        raise KeyError(f"{canonical!r} not in {field_name}")

    def normalize(self, field_name: str, value: str) -> str:
        """Map a predicted string to its canonical name. Unknown strings come back lowercased,
        so they still count as false positives instead of disappearing."""
        v = " ".join(str(value).strip().lower().split())
        table = self.alias_map[field_name]
        if v in table:
            return table[v]
        stripped = _DOSE_TAIL.sub("", v).strip(" ,;")
        if stripped in table:
            return table[stripped]
        # "sertraline (zoloft)" -> try each piece
        for piece in re.split(r"[()/,]", v):
            piece = piece.strip()
            if piece in table:
                return table[piece]
        return v


def _build_alias_map(raw: dict) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for field_name, section in raw.items():
        if field_name.startswith("_"):
            continue
        m: dict[str, str] = {}
        for key, entry in section.items():
            canonical = key.lower()
            if field_name == "conditions":
                forms = [key, *entry["aliases"]]
            elif field_name == "medications":
                forms = [key, *entry["brands"]]
            elif field_name == "allergies":
                forms = [key, *entry]
            else:  # referrals
                forms = [key, *entry["aliases"]]
            for f in forms:
                m[f.lower()] = canonical
        out[field_name] = m
    return out


@lru_cache(maxsize=4)
def load_vocab(path: str | Path = VOCAB_PATH) -> Vocab:
    raw = json.loads(Path(path).read_text())
    return Vocab(raw=raw, alias_map=_build_alias_map(raw))
