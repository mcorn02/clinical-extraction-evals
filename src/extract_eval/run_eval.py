"""Run an extractor over a dataset and score it.

    # Offline keyword baseline (no API key needed)
    python -m extract_eval.run_eval --extractor baseline

    # An LLM, with a versioned prompt
    python -m extract_eval.run_eval --extractor llm --model gpt-5-mini --prompt prompts/extract_v1.md

    # Fail (exit 1) if any field's F1 drops more than 2 points below a reference run
    python -m extract_eval.run_eval --extractor llm --model gpt-5-mini --prompt prompts/extract_v1.md \\
        --compare results/reference/metrics.json --max-drop 0.02
"""

import argparse
import json
import logging
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from .extract import BaselineExtractor, ExtractionResult, LLMExtractor
from .metrics import aggregate, check_regression, score_note
from .schema import ALL_FIELDS, NoteRecord
from .vocab import load_vocab

log = logging.getLogger("extract_eval")


class JsonlTraceWriter:
    """Thread-safe trace sink: appends one JSON line per LLM attempt, flushed immediately."""

    def __init__(self, path: str | Path, run_id: str):
        self.run_id = run_id
        self._lock = threading.Lock()
        self._f = Path(path).open("a")

    def __call__(self, record: dict) -> None:
        line = json.dumps({"run_id": self.run_id, **record}, default=str)
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()

    def close(self) -> None:
        with self._lock:
            self._f.close()


def _setup_logging(log_path: Path) -> list[logging.Handler]:
    """INFO to the console and DEBUG+ to run.log, on the package logger only (not root, so
    third-party HTTP client logs never reach the file)."""
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s [%(threadName)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    log.addHandler(fh)
    log.addHandler(ch)
    return [fh, ch]


def _teardown_logging(handlers: list[logging.Handler]) -> None:
    for h in handlers:
        log.removeHandler(h)
        h.close()


def _git_commit() -> str | None:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent).stdout.strip()
        if not sha:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5,
                               cwd=Path(__file__).resolve().parent).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def load_dataset(path: str | Path, limit: int | None = None) -> list[NoteRecord]:
    rows = [NoteRecord.model_validate_json(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def _fmt(x: float | None, pct: bool = True) -> str:
    if x is None:
        return "–"
    return f"{x * 100:.1f}" if pct else f"{x:.4f}"


def run_stats(results: list[ExtractionResult], wall_clock_s: float | None = None) -> dict:
    lat = sorted(r.latency_s for r in results)
    n = len(results)
    # Notes that ended on an API error have unknown (and probably partial) cost, so they are left
    # out of the cost figures rather than counted as free.
    completed = [r for r in results if not r.api_failed]
    costs = [r.cost_usd for r in completed]
    known_cost = all(c is not None for c in costs)
    total_cost = sum(costs) if known_cost else None
    failures_by_type = Counter(r.error_type or "unknown" for r in results if r.prediction is None)
    sum_latency = sum(lat)
    return {
        "n": n,
        "failures": sum(r.prediction is None for r in results),
        "failures_by_type": dict(sorted(failures_by_type.items())),
        "api_failed_notes": n - len(completed),
        "retries": sum(max(r.attempts - 1, 0) for r in results),
        "attempts": sum(r.attempts for r in results),
        "mean_latency_s": statistics.mean(lat) if lat else None,
        "p95_latency_s": lat[min(n - 1, int(0.95 * n))] if lat else None,
        "sum_call_latency_s": sum_latency,
        "wall_clock_s": wall_clock_s,
        # >1 means calls overlapped; close to --workers means the pool was kept busy
        "parallelism": (sum_latency / wall_clock_s) if wall_clock_s else None,
        "input_tokens": sum(r.input_tokens or 0 for r in results),
        "output_tokens": sum(r.output_tokens or 0 for r in results),
        "total_cost_usd": total_cost,
        "cost_notes": len(completed),
        "cost_per_1k_notes_usd": (total_cost / len(completed) * 1000) if (known_cost and completed) else None,
    }


def render_report(meta: dict, metrics: dict, stats: dict) -> str:
    lines = [
        f"# {meta['extractor']}",
        "",
        f"Data: `{meta['data']}` ({metrics['n_notes']} notes) · Run: {meta['run_id']}",
        "",
        "| Field | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for f in ALL_FIELDS:
        m = metrics["fields"][f]
        lines.append(f"| {f} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} | {m['tp']} | {m['fp']} | {m['fn']} |")
    o = metrics["overall"]
    lines.append(f"| **overall** | **{_fmt(o['precision'])}** | **{_fmt(o['recall'])}** | **{_fmt(o['f1'])}** | {o['tp']} | {o['fp']} | {o['fn']} |")
    lines += [
        "",
        f"- Notes with every field correct: {_fmt(metrics['exact_match_rate'])}%",
        f"- Failed or invalid outputs: {stats['failures']} (retries used: {stats['retries']})",
    ]
    if stats.get("failures_by_type"):
        lines.append("- Failures by type: " + ", ".join(f"{t}: {c}" for t, c in stats["failures_by_type"].items()))
    lines += [
        f"- Latency per note: mean {stats['mean_latency_s']:.2f}s, p95 {stats['p95_latency_s']:.2f}s",
    ]
    if stats.get("wall_clock_s") is not None:
        par = stats.get("parallelism")
        lines.append(f"- Wall clock: {stats['wall_clock_s']:.2f}s; sum of call latency {stats['sum_call_latency_s']:.2f}s"
                     + (f" (parallelism {par:.1f}x)" if par is not None else ""))
    cost = stats["cost_per_1k_notes_usd"]
    cost_line = "- Cost per 1,000 notes: " + (f"${cost:.2f}" if cost is not None else "unknown")
    if stats.get("api_failed_notes"):
        cost_line += (f" (computed over the {stats['cost_notes']} notes whose calls completed; "
                      f"excludes {stats['api_failed_notes']} note(s) that ended on an API error, whose cost is unknown)")
    lines += [
        f"- Tokens: {stats['input_tokens']:,} in / {stats['output_tokens']:,} out",
        cost_line,
    ]
    if metrics.get("by_edge_case"):
        lines += ["", "## By edge case (scored on the field each one targets)", "",
                  "| Edge case | Field(s) | Notes | Precision | Recall | FP | FN |", "|---|---|---:|---:|---:|---:|---:|"]
        for t, m in metrics["by_edge_case"].items():
            lines.append(f"| {t} | {', '.join(m['fields'])} | {m['n_notes']} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | {m['fp']} | {m['fn']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/synthetic/notes_v1.jsonl")
    p.add_argument("--extractor", choices=["baseline", "llm"], default="baseline")
    p.add_argument("--model", help="LiteLLM model string, e.g. gpt-5-mini or anthropic/claude-haiku-5-5")
    p.add_argument("--prompt", default="prompts/extract_v1.md")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--compare", help="metrics.json from a reference run")
    p.add_argument("--max-drop", type=float, default=0.02)
    args = p.parse_args()

    vocab = load_vocab()
    if args.extractor == "baseline":
        extractor = BaselineExtractor(vocab)
    else:
        if not args.model:
            p.error("--model is required with --extractor llm")
        from dotenv import load_dotenv

        load_dotenv()
        extractor = LLMExtractor(args.model, args.prompt, vocab)

    records = load_dataset(args.data, args.limit)
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{extractor.name.replace('/', '-')}"
    meta = {"run_id": run_id, "extractor": extractor.name, "data": args.data,
            "model": args.model, "prompt": args.prompt if args.extractor == "llm" else None,
            "git_commit": _git_commit(), "workers": args.workers}

    # Create the run dir first so run.log and llm_calls.jsonl stream while notes are processed
    out = Path(args.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    handlers = _setup_logging(out / "run.log")
    trace = None
    try:
        if isinstance(extractor, LLMExtractor):
            (out / "system_prompt.txt").write_text(extractor.system_prompt)
            trace = JsonlTraceWriter(out / "llm_calls.jsonl", run_id)
            extractor.trace = trace
        log.info("run %s: extractor=%s model=%s prompt=%s data=%s notes=%d workers=%d git=%s",
                 run_id, args.extractor, args.model, meta["prompt"], args.data, len(records), args.workers,
                 meta["git_commit"])
        if trace is not None:
            log.info("system prompt sha256=%s (saved to system_prompt.txt)", extractor.system_prompt_sha256)

        def run_one(rec: NoteRecord) -> ExtractionResult:
            res = extractor(rec.text, note_id=rec.id)
            cost = f"${res.cost_usd:.5f}" if (res.cost_usd is not None and not res.api_failed) else "unknown"
            level = logging.INFO if res.error_type is None else logging.WARNING
            log.log(level, "note %s done: status=%s attempts=%d latency=%.2fs cost=%s",
                    rec.id, res.error_type or "ok", res.attempts, res.latency_s, cost)
            return res

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="worker") as pool:
            results = list(pool.map(run_one, records))
        wall = time.perf_counter() - t0

        per_note = [score_note(res.prediction, rec.gold, vocab) for res, rec in zip(results, records)]
        metrics = aggregate(per_note, [r.tags for r in records])
        stats = run_stats(results, wall_clock_s=wall)

        with (out / "predictions.jsonl").open("w") as f:
            for rec, res, sc in zip(records, results, per_note):
                f.write(json.dumps({
                    "id": rec.id,
                    "tags": rec.tags,
                    "prediction": res.prediction.model_dump(mode="json") if res.prediction else None,
                    "gold": rec.gold.model_dump(mode="json"),
                    "errors": {k: {"fp": v.fp, "fn": v.fn} for k, v in sc.items() if v.fp or v.fn},
                    "raw": res.raw,
                    "error": res.error,
                    "error_type": res.error_type,
                    "attempts": res.attempts,
                    "latency_s": round(res.latency_s, 4),
                    "cost_usd": res.cost_usd,
                }) + "\n")
        (out / "metrics.json").write_text(json.dumps({"meta": meta, "stats": stats, **metrics}, indent=2, default=str))
        report = render_report(meta, metrics, stats)
        (out / "report.md").write_text(report)

        cost = stats["cost_per_1k_notes_usd"]
        log.info("summary: %d notes in %.2fs wall clock (sum of call latency %.2fs, parallelism %s); "
                 "failures=%d%s retries=%d; overall F1=%s; cost/1k notes=%s",
                 stats["n"], wall, stats["sum_call_latency_s"],
                 f"{stats['parallelism']:.1f}x" if stats["parallelism"] is not None else "n/a",
                 stats["failures"], f" {stats['failures_by_type']}" if stats["failures_by_type"] else "", stats["retries"],
                 _fmt(metrics["overall"]["f1"]), f"${cost:.2f}" if cost is not None else "unknown")
    except Exception:
        log.exception("run %s failed", run_id)
        raise
    finally:
        if trace is not None:
            trace.close()
        _teardown_logging(handlers)

    print(report)
    print(f"Saved to {out}/")

    if args.compare:
        reference = json.loads(Path(args.compare).read_text())
        problems = check_regression(metrics, reference, args.max_drop)
        if problems:
            print("REGRESSION vs reference:\n  " + "\n  ".join(problems))
            sys.exit(1)
        print(f"No field dropped more than {args.max_drop:.0%} vs reference.")


if __name__ == "__main__":
    main()
