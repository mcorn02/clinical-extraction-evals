"""Error classification, failed-call timing, per-attempt traces and run stats. No network."""

import json
import sys

import litellm
import pytest

from extract_eval import llm, run_eval
from extract_eval.extract import ExtractionResult, LLMExtractor
from extract_eval.llm import LLMResponse, call_llm, classify_error, describe_error
from extract_eval.run_eval import render_report, run_stats

GOOD = {
    "visit_date": "2026-03-14", "visit_type": "follow_up", "patient_age": 9,
    "conditions": ["asthma"], "medications": ["albuterol"], "allergies": [],
    "referrals": [], "follow_up_weeks": 6,
}


class ProviderError(Exception):
    status_code = 429


@pytest.mark.parametrize("exc, expected", [
    (litellm.RateLimitError("slow down", "openai", "m"), "rate_limit"),
    (litellm.AuthenticationError("bad key", "openai", "m"), "auth"),
    (litellm.Timeout("timed out", "m", "openai"), "timeout"),  # subclass of APIConnectionError
    (litellm.APIConnectionError("conn reset", "openai", "m"), "connection"),
    (litellm.ContextWindowExceededError("too long", "m", "openai"), "context_window"),  # subclass of BadRequest
    (litellm.BadRequestError("bad param", "m", "openai"), "bad_request"),
    (litellm.ServiceUnavailableError("down", "openai", "m"), "server_error"),
    (litellm.InternalServerError("oops", "openai", "m"), "server_error"),
    (litellm.NotFoundError("no such model", "m", "openai"), "not_found"),
    (RuntimeError("rate limited"), "api_error_other"),
    (ProviderError("raw SDK error"), "rate_limit"),  # unknown class, falls back to status_code
])
def test_classify_error(exc, expected):
    assert classify_error(exc) == expected


def test_describe_error_includes_type_class_status_and_redacts():
    e = litellm.RateLimitError("limit hit for key sk-abcdefghijklmnop", "openai", "m")
    msg = describe_error(e)
    assert msg.startswith("rate_limit: RateLimitError (status 429): ")
    assert "sk-abcdefghijklmnop" not in msg and "[REDACTED]" in msg


def test_call_llm_times_failed_calls(monkeypatch):
    def boom(**kw):
        raise litellm.Timeout("timed out", "m", "openai")

    monkeypatch.setattr(litellm, "completion", boom)
    with pytest.raises(litellm.Timeout) as info:
        call_llm("m", [{"role": "user", "content": "hi"}])
    assert isinstance(info.value.latency_s, float) and info.value.latency_s >= 0


def _extractor(replies, trace=None):
    calls = []

    def fake(model, messages, **kw):
        calls.append(messages)
        reply = replies[len(calls) - 1]
        if isinstance(reply, Exception):
            reply.latency_s = 2.5  # what call_llm attaches on failure
            raise reply
        return LLMResponse(text=reply, latency_s=0.5, input_tokens=100, output_tokens=50, cost_usd=0.001)

    e = LLMExtractor("fake-model", "prompts/extract_v1.md", trace=trace)
    e._call = fake
    return e


def test_api_error_sets_type_and_records_latency():
    res = _extractor([litellm.RateLimitError("slow down", "openai", "m")])("note")
    assert res.prediction is None and res.error_type == "rate_limit"
    assert res.error.startswith("rate_limit: RateLimitError (status 429)")
    assert res.latency_s == 2.5 and res.api_failed


def test_plain_exception_is_api_error_other_and_timed_without_attribute():
    def fake(model, messages, **kw):
        raise RuntimeError("rate limited")

    e = LLMExtractor("fake-model", "prompts/extract_v1.md")
    e._call = fake
    res = e("note")
    assert res.error_type == "api_error_other" and "rate limited" in res.error
    assert res.latency_s >= 0


def test_invalid_output_type_and_success_type():
    assert _extractor(["nope", "still nope"])("note").error_type == "invalid_output"
    assert _extractor([json.dumps(GOOD)])("note").error_type is None


def test_trace_gets_one_record_per_attempt_including_retry():
    records = []
    bad = json.dumps({**GOOD, "visit_type": "phone call"})
    res = _extractor([bad, json.dumps(GOOD)], trace=records.append)("note", note_id="n-7")
    assert res.prediction is not None and res.attempts == 2
    assert [r["attempt"] for r in records] == [1, 2]
    assert [r["status"] for r in records] == ["invalid_output", "ok"]
    first, second = records
    assert first["note_id"] == "n-7" and first["model"] == "fake-model"
    assert first["prompt_file"].endswith("extract_v1.md") and len(first["system_prompt_sha256"]) == 64
    assert "failed validation" in first["retry_feedback"] and second["retry_feedback"] is None
    assert first["raw"] == bad and first["input_tokens"] == 100 and first["cost_usd"] == 0.001
    assert first["started_at"] <= first["ended_at"] and first["thread"]
    assert "system_prompt" not in first  # logged once per run, not per line


def test_trace_records_api_failure_and_broken_sink_is_harmless():
    records = []
    _extractor([litellm.AuthenticationError("bad key", "openai", "m")], trace=records.append)("note", note_id="n-1")
    (r,) = records
    assert r["status"] == "auth" and r["latency_s"] == 2.5 and r["cost_usd"] is None and r["raw"] is None

    def broken(_):
        raise OSError("disk full")

    assert _extractor([json.dumps(GOOD)], trace=broken)("note").prediction is not None


def _ok(cost):
    return ExtractionResult(prediction=object(), latency_s=1.0, cost_usd=cost)


def _fail(err_type, cost=0.0, attempts=1):
    return ExtractionResult(prediction=None, latency_s=1.0, cost_usd=cost, error_type=err_type, attempts=attempts)


def test_run_stats_failures_by_type_and_cost_excludes_api_failures():
    results = [_ok(0.002), _ok(0.002), _fail("invalid_output", cost=0.004, attempts=2),
               _fail("rate_limit"), _fail("rate_limit"), _fail("timeout")]
    s = run_stats(results, wall_clock_s=2.0)
    assert s["failures"] == 4
    assert s["failures_by_type"] == {"invalid_output": 1, "rate_limit": 2, "timeout": 1}
    assert s["api_failed_notes"] == 3 and s["cost_notes"] == 3
    assert s["cost_per_1k_notes_usd"] == pytest.approx(0.008 / 3 * 1000)  # not / 6
    assert s["sum_call_latency_s"] == 6.0 and s["parallelism"] == 3.0


def test_report_shows_failure_breakdown_and_cost_note():
    stats = run_stats([_ok(0.002), _fail("rate_limit")], wall_clock_s=1.0)
    metrics = run_eval.aggregate([])  # empty but well-formed
    metrics["n_notes"] = 2
    report = render_report({"extractor": "x", "data": "d", "run_id": "r"}, metrics, stats)
    assert "Failures by type: rate_limit: 1" in report
    assert "excludes 1 note(s) that ended on an API error" in report


def test_main_llm_path_offline(tmp_path, monkeypatch):
    """End to end through run_eval.main with call_llm faked: every run file is written."""
    from extract_eval.generate import generate

    data = tmp_path / "notes.jsonl"
    data.write_text("".join(r.model_dump_json() + "\n" for r in generate(4, seed=3)))
    replies = iter(["not json", json.dumps(GOOD)] + [json.dumps(GOOD)] * 2)

    def fake(model, messages, **kw):
        try:
            return LLMResponse(text=next(replies), latency_s=0.1, input_tokens=10, output_tokens=5, cost_usd=0.0001)
        except StopIteration:
            raise litellm.RateLimitError("slow down", "openai", model)

    monkeypatch.setattr(llm, "call_llm", fake)
    monkeypatch.setattr(sys, "argv", ["run_eval", "--extractor", "llm", "--model", "fake/model", "--data", str(data),
                                      "--workers", "1", "--out-dir", str(tmp_path / "out")])
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    run_eval.main()

    (run_dir,) = (tmp_path / "out").iterdir()
    names = {p.name for p in run_dir.iterdir()}
    assert {"llm_calls.jsonl", "system_prompt.txt", "run.log", "predictions.jsonl", "metrics.json", "report.md"} <= names
    calls = [json.loads(line) for line in (run_dir / "llm_calls.jsonl").read_text().splitlines()]
    assert len(calls) == 5  # 4 notes + 1 retry
    assert {c["run_id"] for c in calls} == {run_dir.name}
    assert sorted(c["status"] for c in calls).count("rate_limit") == 1
    preds = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
    assert all("error_type" in p and "attempts" in p for p in preds)
    stats = json.loads((run_dir / "metrics.json").read_text())["stats"]
    assert stats["failures_by_type"] == {"rate_limit": 1} and stats["wall_clock_s"] > 0
    log_text = (run_dir / "run.log").read_text()
    assert "extractor=llm" in log_text and "API error rate_limit" in log_text and "summary:" in log_text
