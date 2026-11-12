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
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from .extract import BaselineExtractor, ExtractionResult, LLMExtractor
from .metrics import aggregate, check_regression, score_note
from .schema import ALL_FIELDS, NoteRecord
from .vocab import load_vocab


def load_dataset(path: str | Path, limit: int | None = None) -> list[NoteRecord]:
    rows = [NoteRecord.model_validate_json(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def _fmt(x: float | None, pct: bool = True) -> str:
    if x is None:
        return "–"
    return f"{x * 100:.1f}" if pct else f"{x:.4f}"


def run_stats(results: list[ExtractionResult]) -> dict:
    lat = sorted(r.latency_s for r in results)
    costs = [r.cost_usd for r in results]
    known_cost = all(c is not None for c in costs)
    total_cost = sum(costs) if known_cost else None
    n = len(results)
    return {
        "n": n,
        "failures": sum(r.prediction is None for r in results),
        "retries": sum(max(r.attempts - 1, 0) for r in results),
        "mean_latency_s": statistics.mean(lat) if lat else None,
        "p95_latency_s": lat[min(n - 1, int(0.95 * n))] if lat else None,
        "input_tokens": sum(r.input_tokens or 0 for r in results),
        "output_tokens": sum(r.output_tokens or 0 for r in results),
        "total_cost_usd": total_cost,
        "cost_per_1k_notes_usd": (total_cost / n * 1000) if (known_cost and n) else None,
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
        f"- Latency: mean {stats['mean_latency_s']:.2f}s, p95 {stats['p95_latency_s']:.2f}s",
        f"- Tokens: {stats['input_tokens']:,} in / {stats['output_tokens']:,} out",
        "- Cost per 1,000 notes: "
        + (f"${stats['cost_per_1k_notes_usd']:.2f}" if stats["cost_per_1k_notes_usd"] is not None else "unknown"),
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
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda r: extractor(r.text), records))

    per_note = [score_note(res.prediction, rec.gold, vocab) for res, rec in zip(results, records)]
    metrics = aggregate(per_note, [r.tags for r in records])
    stats = run_stats(results)

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{extractor.name.replace('/', '-')}"
    meta = {"run_id": run_id, "extractor": extractor.name, "data": args.data,
            "model": args.model, "prompt": args.prompt if args.extractor == "llm" else None}

    out = Path(args.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
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
                "latency_s": round(res.latency_s, 4),
                "cost_usd": res.cost_usd,
            }) + "\n")
    (out / "metrics.json").write_text(json.dumps({"meta": meta, "stats": stats, **metrics}, indent=2, default=str))
    report = render_report(meta, metrics, stats)
    (out / "report.md").write_text(report)
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
