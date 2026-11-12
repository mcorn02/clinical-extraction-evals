# Clinical Extraction Evals

An open kit for measuring how well LLMs pull structured facts out of messy clinical-style notes: diagnoses, medications, allergies, referrals and follow-up timing. It reports how accurate each field is, where models fail, and what each model costs per 1,000 notes.

**All data here is synthetic.** No real patient notes, ever.

## Why

Before a clinic trusts a model to read its notes, it needs answers to three questions:

1. How often is each field wrong, and in which direction: missing things, or inventing them?
2. Which kinds of notes break it? Common culprits are denied conditions, family history, rule-outs, stopped medicines and declined referrals.
3. What does it cost and how fast is it, compared with the alternatives?

This repo answers those questions for any model LiteLLM supports.

## How it works

1. **Generate notes with known answers.** `generate.py` samples the structured record first, then writes the note from it, so the labels are exact. Each note also includes *distractors*: things mentioned in the note that must not be extracted.
2. **Extract.** Either a keyword baseline (offline, free) or an LLM with a versioned prompt from `prompts/`.
3. **Score every field.** Precision, recall and F1 per field, plus a breakdown by edge case, cost per 1,000 notes and latency.
4. **Gate regressions.** `--compare` fails the run if any field's F1 drops below a reference run.

| Edge case | Example | Should extract |
|---|---|---|
| negation | "Denies history of asthma." | nothing |
| family_history | "FHx: mother with hypothyroidism." | nothing |
| rule_out | "R/O ADHD; screening forms sent home." | nothing |
| discontinued_med | "Discontinued Prozac last month." | nothing |
| declined_referral | "Offered referral to OT; family declined." | nothing |
| brand_name / alias | "Zoloft 50 mg", "HTN", "SLP" | sertraline, hypertension, speech therapy |
| number_words | "Return in four weeks." | 4 |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # add the API keys you have

python -m extract_eval.generate --n 20 --seed 0          # refuses: notes_v1.jsonl is committed (use --out notes_v2.jsonl for new data)
python -m extract_eval.run_eval --extractor baseline     # offline, no key needed
python -m extract_eval.run_eval --extractor llm --model gpt-5-mini --prompt prompts/extract_v1.md
pytest
```

Each run writes `results/<run_id>/` with `report.md`, `metrics.json` and `predictions.jsonl`. The predictions file lists every false positive and false negative for each note.

## Results

20 synthetic notes (`notes_v1`, seed 0).

| Extractor | Prompt | Overall F1 | Conditions precision | Notes fully correct | Cost / 1k notes |
|---|---|---:|---:|---:|---:|
| Keyword baseline | – | 90.1 | 64.1 | 5% | $0 |
| _model A_ | v1 | | | | |
| _model B_ | v1 | | | | |

The baseline finds every mention (100% recall) but has no sense of context, so its precision collapses on negations, rule-outs and family history. That's the gap an LLM has to close.

## Layout

```
data/vocab.json              canonical names, aliases, brands, ICD-10 codes
data/synthetic/              generated datasets (committed, so runs are reproducible)
prompts/extract_v*.md        versioned prompts. Add a new file; don't edit an old one
src/extract_eval/
  schema.py                  Pydantic model: the extraction target
  generate.py                synthetic notes with known labels
  extract.py                 baseline + LLM extractors (one retry on invalid JSON)
  metrics.py                 field-level precision/recall, edge-case breakdown, regression check
  run_eval.py                CLI
tests/                       metric math, label integrity, extractor behavior (no network)
```

## Roadmap

- [x] Schema, generator, baseline, field-level metrics
- [ ] First LLM run on 20 notes
- [ ] 3–4 models compared on accuracy, cost per 1,000 notes and latency
- [ ] Prompt v0 vs v1 comparison
- [ ] Scale to 75–100 notes; optional LLM rewrite pass (`--rewrite-model`), with 20 rewritten notes checked by hand
- [ ] Langfuse tracing
- [ ] Reference run committed to `results/reference/` and used as the regression gate
- [ ] Write-up: *How I'd evaluate this for a clinic*
