import json

from extract_eval.extract import BaselineExtractor, LLMExtractor, parse_json_object
from extract_eval.generate import generate
from extract_eval.llm import LLMResponse
from extract_eval.metrics import aggregate, score_note

GOOD = {
    "visit_date": "2026-03-14", "visit_type": "follow_up", "patient_age": 9,
    "conditions": ["asthma"], "medications": ["albuterol"], "allergies": [],
    "referrals": [], "follow_up_weeks": 6,
}


def test_baseline_finds_every_mention():
    """Keyword matching should never miss a mention. Its errors should all be false positives."""
    records = generate(50, seed=1)
    m = aggregate([score_note(BaselineExtractor()(r.text).prediction, r.gold) for r in records])
    assert m["overall"]["recall"] == 1.0
    assert m["fields"]["conditions"]["fp"] > 0  # it falls for negations and family history


def test_parse_json_object_handles_code_fences():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def _fake_llm(replies):
    calls = []

    def fake(model, messages, **kw):
        calls.append(messages)
        reply = replies[len(calls) - 1]
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(text=reply, latency_s=0.5, input_tokens=100, output_tokens=50, cost_usd=0.001)

    return fake, calls


def _extractor(fake):
    e = LLMExtractor("fake-model", "prompts/extract_v1.md")
    e._call = fake
    return e


def test_llm_extractor_valid_output():
    fake, calls = _fake_llm([json.dumps(GOOD)])
    res = _extractor(fake)("note text")
    assert res.prediction is not None and res.prediction.conditions == ["asthma"]
    assert res.attempts == 1 and res.cost_usd == 0.001


def test_llm_extractor_retries_once_on_invalid_output():
    bad = json.dumps({**GOOD, "visit_type": "phone call"})
    fake, calls = _fake_llm([bad, json.dumps(GOOD)])
    res = _extractor(fake)("note text")
    assert res.prediction is not None and res.attempts == 2
    assert "failed validation" in calls[1][-1]["content"]
    assert res.input_tokens == 200


def test_llm_extractor_gives_up_after_retry():
    fake, _ = _fake_llm(["not json", "still not json"])
    res = _extractor(fake)("note text")
    assert res.prediction is None and res.error.startswith("invalid_output")


def test_llm_extractor_api_error():
    fake, _ = _fake_llm([RuntimeError("rate limited")])
    res = _extractor(fake)("note text")
    assert res.prediction is None and "rate limited" in res.error


def test_prompt_placeholders_filled():
    e = LLMExtractor("fake-model", "prompts/extract_v1.md")
    assert "{{" not in e.system_prompt and "autism spectrum disorder" in e.system_prompt
