"""Extractors: a keyword baseline (offline, free) and an LLM extractor (any LiteLLM model)."""

import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from pydantic import ValidationError

from .schema import LIST_FIELDS, NoteExtraction
from .vocab import Vocab, load_vocab


@dataclass
class ExtractionResult:
    prediction: NoteExtraction | None
    raw: str | None = None
    error: str | None = None
    latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    attempts: int = 1


# --------------------------------------------------------------------------------------
# Baseline: keyword matching. It finds every mention but ignores context, so negations,
# family history, rule-outs, stopped meds and declined referrals all become false positives.
# That makes it a useful floor and a check that the eval actually catches those mistakes.
# --------------------------------------------------------------------------------------

_WORD_NUMS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "ten": 10, "twelve": 12}
_DATE_PATTERNS = [
    (re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"), "%Y-%m-%d"),
    (re.compile(r"\b(\d{2}/\d{2}/\d{4})\b"), "%m/%d/%Y"),
    (re.compile(r"\b([A-Z][a-z]+ \d{1,2}, \d{4})\b"), "%B %d, %Y"),
]


class BaselineExtractor:
    name = "baseline"

    def __init__(self, vocab: Vocab | None = None):
        self.vocab = vocab or load_vocab()
        self.patterns: dict[str, list[tuple[re.Pattern, str]]] = {}
        for field in LIST_FIELDS:
            pats = []
            for canonical in self.vocab.canonical_names(field):
                for form in self.vocab.surface_forms(field, canonical):
                    # Short all-caps abbreviations (PT, OT, GAD) match case-sensitively so "Pt" isn't physical therapy
                    flags = 0 if (form.isupper() and len(form) <= 5) else re.IGNORECASE
                    pats.append((re.compile(rf"(?<![\w-]){re.escape(form)}(?![\w-])", flags), canonical))
            self.patterns[field] = pats

    def _date(self, text: str) -> date | None:
        for pat, fmt in _DATE_PATTERNS:
            m = pat.search(text)
            if m:
                try:
                    return datetime.strptime(m.group(1), fmt).date()
                except ValueError:
                    continue
        return None

    def __call__(self, text: str) -> ExtractionResult:
        start = time.perf_counter()
        lowered = text.lower()
        if re.search(r"telehealth|virtual", lowered):
            visit_type = "telehealth"
        elif re.search(r"initial|new patient", lowered):
            visit_type = "initial_assessment"
        elif re.search(r"follow-up visit|f/u visit|return visit", lowered):
            visit_type = "follow_up"
        else:
            visit_type = None

        age_m = re.search(r"(\d{1,3})\s*(?:y/o|yo\b|-year-old)", text)
        fu_m = re.search(r"(?:rtc|follow up|return)\s+in\s+(\d+|[a-z]+)\s*(?:wks?|weeks?)\b", lowered)
        follow_up = None
        if fu_m:
            tok = fu_m.group(1)
            follow_up = int(tok) if tok.isdigit() else _WORD_NUMS.get(tok)

        lists = {}
        for field, pats in self.patterns.items():
            found = []
            for pat, canonical in pats:
                if pat.search(text) and canonical not in found:
                    found.append(canonical)
            lists[field] = found

        pred = NoteExtraction(
            visit_date=self._date(text),
            visit_type=visit_type,
            patient_age=int(age_m.group(1)) if age_m else None,
            follow_up_weeks=follow_up,
            **lists,
        )
        return ExtractionResult(prediction=pred, latency_s=time.perf_counter() - start, cost_usd=0.0)


# --------------------------------------------------------------------------------------
# LLM extractor
# --------------------------------------------------------------------------------------

def _vocab_block(vocab: Vocab) -> str:
    lines = []
    for field in LIST_FIELDS:
        lines.append(f"- {field}: " + "; ".join(vocab.canonical_names(field)))
    return "\n".join(lines)


def build_system_prompt(prompt_path: str | Path, vocab: Vocab) -> str:
    template = Path(prompt_path).read_text()
    schema = json.dumps(NoteExtraction.model_json_schema(), indent=1)
    return template.replace("{{VOCAB}}", _vocab_block(vocab)).replace("{{SCHEMA}}", schema)


def parse_json_object(text: str) -> dict:
    """Pull the JSON object out of a model reply, tolerating code fences or stray prose."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start : end + 1])


class LLMExtractor:
    def __init__(self, model: str, prompt_path: str | Path, vocab: Vocab | None = None, max_retries: int = 1):
        from .llm import call_llm  # imported here so the baseline works without litellm installed

        self._call = call_llm
        self.model = model
        self.prompt_path = Path(prompt_path)
        self.name = f"{model}__{self.prompt_path.stem}"
        self.system_prompt = build_system_prompt(prompt_path, vocab or load_vocab())
        self.max_retries = max_retries

    def __call__(self, text: str) -> ExtractionResult:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]
        result = ExtractionResult(prediction=None, latency_s=0.0, input_tokens=0, output_tokens=0, cost_usd=0.0, attempts=0)
        for attempt in range(self.max_retries + 1):
            result.attempts = attempt + 1
            try:
                resp = self._call(self.model, messages)
            except Exception as e:  # network, auth, rate limit
                result.error = f"api_error: {type(e).__name__}: {e}"[:500]
                return result
            result.raw = resp.text
            result.latency_s += resp.latency_s
            result.input_tokens = (result.input_tokens or 0) + (resp.input_tokens or 0)
            result.output_tokens = (result.output_tokens or 0) + (resp.output_tokens or 0)
            result.cost_usd = None if (result.cost_usd is None or resp.cost_usd is None) else result.cost_usd + resp.cost_usd
            try:
                result.prediction = NoteExtraction.model_validate(parse_json_object(resp.text))
                result.error = None
                return result
            except (ValueError, ValidationError) as e:
                result.error = f"invalid_output: {e}"[:500]
                # Retry once, telling the model what was wrong
                messages = messages + [
                    {"role": "assistant", "content": resp.text},
                    {"role": "user", "content": f"That output failed validation: {e}\nReturn only the corrected JSON object."},
                ]
        return result
