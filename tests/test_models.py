import json
import sys

import pytest

from extract_eval import run_eval
from extract_eval.models import ModelEntry, load_models, resolve_models

ENTRIES = [
    ModelEntry("a/cheap", "A_KEY", True),
    ModelEntry("b/cheap", "B_KEY", True),
    ModelEntry("a/pricey", "A_KEY", False),
]


def test_all_runs_enabled_models_with_keys_and_reports_the_rest():
    run, skipped = resolve_models("all", ENTRIES, env={"A_KEY": "x"})
    assert run == ["a/cheap"]  # b/cheap has no key; a/pricey is disabled so it is not even a candidate
    assert skipped == [("b/cheap", "B_KEY is not set")]


def test_explicit_list_runs_disabled_models_and_unregistered_names():
    run, skipped = resolve_models("a/pricey, other/model ,a/pricey", ENTRIES, env={"A_KEY": "x"})
    assert run == ["a/pricey", "other/model"] and skipped == []


def test_explicit_registered_model_without_key_is_skipped():
    run, skipped = resolve_models("b/cheap", ENTRIES, env={})
    assert run == [] and skipped[0][0] == "b/cheap"


def test_shipped_models_json_is_valid():
    entries = load_models()
    assert entries and all(e.name for e in entries)
    assert len({e.name for e in entries}) == len(entries)


def _fake_result(model, f1, cost=0.5, n=10, failures=0):
    o = {"precision": f1, "recall": f1, "f1": f1}
    fields = {f: {"f1": f1} for f in run_eval.ALL_FIELDS}
    return {"model": model, "run_dir": f"results/x_{model}", "regressed": False,
            "metrics": {"n_notes": n, "overall": o, "exact_match_rate": f1, "fields": fields, "oov": {"total": 2}},
            "stats": {"failures": failures, "failures_by_type": {"not_found": failures} if failures else {},
                      "mean_latency_s": 1.0, "cost_per_1k_notes_usd": cost}}


def test_comparison_table_sorts_best_first_and_lists_failures():
    md = run_eval.render_comparison(
        [_fake_result("weak", 0.5), _fake_result("strong", 0.9)],
        failed=[("broken", "NotFoundError: no such model")], skipped=[("nokey", "X is not set")],
        meta={"data": "d.jsonl", "prompt": "p.md", "stamp": "s"},
    )
    assert md.index("| strong") < md.index("| weak")
    assert "broken: NotFoundError" in md and "nokey: X is not set" in md


def test_models_all_runs_each_model_and_survives_one_failure(tmp_path, monkeypatch):
    calls = []

    def fake_run_once(args, model):
        calls.append(model)
        if model == "b/cheap":
            raise RuntimeError("boom")
        return _fake_result(model, 0.8)

    monkeypatch.setattr(run_eval, "run_once", fake_run_once)
    monkeypatch.setattr(run_eval, "load_models", lambda: ENTRIES)
    monkeypatch.setenv("A_KEY", "x")
    monkeypatch.setenv("B_KEY", "y")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["run_eval", "--models", "all", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        run_eval.main()
    assert e.value.code == 1  # a model failed
    assert calls == ["a/cheap", "b/cheap"]  # the failure didn't stop the loop
    summary = next(tmp_path.glob("compare_*.json"))
    data = json.loads(summary.read_text())
    assert [r["model"] for r in data["runs"]] == ["a/cheap"] and data["failed"][0][0] == "b/cheap"
    assert next(tmp_path.glob("compare_*.md")).exists()


def test_model_and_models_together_is_an_error(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_eval", "--model", "x", "--models", "all"])
    with pytest.raises(SystemExit) as e:
        run_eval.main()
    assert e.value.code == 2


def test_all_failed_detects_a_run_where_every_note_errored():
    assert run_eval.all_failed(_fake_result("ok", 0.9)) is None
    assert run_eval.all_failed(_fake_result("some", 0.9, failures=3)) is None  # partial: still has scores
    why = run_eval.all_failed(_fake_result("dead", None, n=32, failures=32))
    assert "all 32 notes failed" in why and "not_found: 32" in why


def test_models_loop_reports_a_model_whose_every_note_failed(tmp_path, monkeypatch):
    def fake_run_once(args, model):
        return _fake_result(model, None, failures=10) if model == "b/cheap" else _fake_result(model, 0.8)

    monkeypatch.setattr(run_eval, "run_once", fake_run_once)
    monkeypatch.setattr(run_eval, "load_models", lambda: ENTRIES)
    monkeypatch.setenv("A_KEY", "x")
    monkeypatch.setenv("B_KEY", "y")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["run_eval", "--models", "all", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        run_eval.main()
    assert e.value.code == 1
    md = next(tmp_path.glob("compare_*.md")).read_text()
    assert "## Failed to run" in md and "b/cheap: all 10 notes failed" in md
    assert "| b/cheap |" not in md  # not a score row


def test_single_model_run_with_every_note_failed_exits_nonzero(monkeypatch):
    monkeypatch.setattr(run_eval, "run_once", lambda a, m: _fake_result(m, None, failures=10))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["run_eval", "--extractor", "llm", "--model", "x/y"])
    with pytest.raises(SystemExit) as e:
        run_eval.main()
    assert e.value.code == 1
