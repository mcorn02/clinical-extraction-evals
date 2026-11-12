"""CLI safety: never overwrite a dataset in place by accident, and always record provenance."""

import hashlib
import json

from extract_eval import generate as gen

META_KEYS = {
    "n", "seed", "rewrite_model", "source", "created_at", "vocab_sha256",
    "git_commit", "edge_case_counts", "jsonl_sha256",
}


def test_refuses_existing_file_and_leaves_it_unchanged(tmp_path, capsys):
    out = tmp_path / "notes_v1.jsonl"
    out.write_text("original\n")
    (tmp_path / "notes_v3.jsonl").write_text("x\n")

    assert gen.main(["--n", "2", "--out", str(out)]) != 0

    assert out.read_text() == "original\n"
    assert not gen.sidecar_path(out).exists()
    err = capsys.readouterr().err
    assert str(out) in err
    assert "notes_v4.jsonl" in err
    assert "--force" in err


def test_refuses_existing_sidecar(tmp_path):
    out = tmp_path / "notes_v1.jsonl"
    gen.sidecar_path(out).write_text("{}")
    assert gen.main(["--n", "2", "--out", str(out)]) != 0
    assert not out.exists()


def test_force_overwrites(tmp_path):
    out = tmp_path / "notes_v1.jsonl"
    out.write_text("original\n")
    gen.sidecar_path(out).write_text("{}")

    assert gen.main(["--n", "3", "--seed", "1", "--out", str(out), "--force"]) == 0

    lines = out.read_text().splitlines()
    assert lines == [r.model_dump_json() for r in gen.generate(3, seed=1)]
    assert json.loads(gen.sidecar_path(out).read_text())["n"] == 3


def test_sidecar_written_with_expected_keys(tmp_path):
    out = tmp_path / "notes_v2.jsonl"
    assert gen.main(["--n", "4", "--seed", "5", "--out", str(out)]) == 0

    meta_path = tmp_path / "notes_v2.meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert set(meta) == META_KEYS
    assert meta["n"] == 4 and meta["seed"] == 5
    assert meta["rewrite_model"] is None
    assert meta["source"] == "template_v1"
    assert meta["jsonl_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert meta["vocab_sha256"] == hashlib.sha256(gen.VOCAB_PATH.read_bytes()).hexdigest()
    assert meta["edge_case_counts"] == gen.tag_counts(gen.generate(4, seed=5))


def test_next_free_name(tmp_path):
    assert gen.next_free_name(tmp_path) == "notes_v1.jsonl"
    assert gen.next_free_name(tmp_path / "missing") == "notes_v1.jsonl"
    for name in ["notes_v1.jsonl", "notes_v2.jsonl", "notes_v10.jsonl", "notes_v11.meta.json", "notes_vX.jsonl"]:
        (tmp_path / name).write_text("")
    assert gen.next_free_name(tmp_path) == "notes_v11.jsonl"


def test_rewrite_logs_one_line_per_note(tmp_path, monkeypatch, caplog):
    from extract_eval import llm

    monkeypatch.setattr(
        llm, "call_llm",
        lambda model, messages, **kw: llm.LLMResponse(
            text="rewritten", latency_s=0.5, input_tokens=10, output_tokens=5, cost_usd=0.001
        ),
    )
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    out = tmp_path / "notes_v1.jsonl"
    with caplog.at_level("INFO", logger="extract_eval.generate"):
        assert gen.main(["--n", "2", "--out", str(out), "--rewrite-model", "fake-model"]) == 0

    lines = [m for m in caplog.messages if m.startswith("rewrote ")]
    assert len(lines) == 2
    assert "note_0_0000" in lines[0] and "fake-model" in lines[0] and "tokens_in=10" in lines[0]
    meta = json.loads(gen.sidecar_path(out).read_text())
    assert meta["rewrite_model"] == "fake-model"
    assert meta["source"] == "template_v1+rewrite:fake-model"
