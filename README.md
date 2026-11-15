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
python -m extract_eval.run_eval --models all --prompt prompts/extract_v1.md   # every model in models.json with a key set
pytest
```

Each run writes `results/<run_id>/` with `report.md`, `metrics.json` and `predictions.jsonl`. The predictions file lists every false positive and false negative for each note.

## Results

### Round 1: `notes_v1` (20 generated notes, seed 0, prompt v1)

| Extractor | Overall F1 | Conditions precision | Notes fully correct | Cost / 1k notes |
|---|---:|---:|---:|---:|
| Keyword baseline | 90.1 | 64.1 | 5% | $0 |
| Claude Haiku 5.5 | 100.0 | 100.0 | 100% | $0.29 |

The baseline finds every mention (100% recall) but has no sense of context, so its precision collapses on negations, rule-outs and family history. Haiku 5.5 closes that gap completely, which says more about the dataset than the model: the template phrasings are too regular to separate models. Full run: [`results/reference/`](results/reference/).

### Round 2: `notes_v3` (24 hand-written hard notes, prompt v2)

Hard cases include checkbox forms, corrections, resolved or family conditions, planned or restarted meds, refuted allergies, conditional referrals and date or age distractors. Six models, same prompt:

| Model | F1 | Exact notes | Mean latency | Cost / 1k notes |
|---|---:|---:|---:|---:|
| Claude Haiku 5.5 | 99.7 | 95.8% | 1.65s | $0.29 |
| GPT-6 Luna | 99.7 | 95.8% | 1.97s | $0.09 |
| Gemini 3.8 Flash | 99.7 | 95.8% | 3.42s | $2.13 |
| Gemini 4 Hydrogen | 99.7 | 95.8% | 2.51s | $5.49 |
| Claude Sonnet 5.5 | 99.7 | 95.8% | 1.61s | $4.51 |
| GPT-5.4 Nano | 98.9 | 91.7% | 1.22s | $0.34 |

Takeaways so far:
- Accuracy does not separate the top five: all make the same single conditions error (one out-of-vocabulary term). Cost does: Luna and Haiku match Sonnet and Gemini 4 Hydrogen at roughly 2-60x lower cost.
- Nano is the cheapest-and-fastest tier but misses on `patient_age` and `follow_up_weeks`, so it's the only model that differs.
- With 24 notes, one note is about 4 points of exact-match, so differences this small are not significant. Scaling the dataset is the next step.
- Per-run folders under `results/` are gitignored; the numbers above come from `results/compare_20261114-173407.md` (local).

## Layout

```
data/vocab.json              canonical names, aliases, brands, ICD-10 codes
data/synthetic/              datasets (committed, so runs are reproducible). notes_v1: generated;
                             notes_v2, notes_v3: hand-written hard cases (resolved or family conditions, planned or
                             restarted meds, refuted allergies, conditional referrals, date traps)
models.json                  models that `--models all` runs (LiteLLM names, API key env var, enabled flag)
prompts/extract_v*.md        versioned prompts. Add a new file; don't edit an old one
src/extract_eval/
  schema.py                  Pydantic model: the extraction target
  generate.py                synthetic notes with known labels
  extract.py                 baseline + LLM extractors (one retry on invalid JSON)
  metrics.py                 field-level precision/recall, edge-case breakdown, out-of-vocab flagging, regression check
  models.py                  reads models.json and resolves `--models all` / comma lists
  run_eval.py                CLI: one model, or several with --models (writes a side-by-side comparison)
tests/                       metric math, label integrity, extractor behavior (no network)
```

## Roadmap

- [x] Schema, generator, baseline, field-level metrics
- [x] First LLM run on 20 notes (Claude Haiku 5.5, prompt v1)
- [x] 6 models compared on accuracy, cost per 1,000 notes and latency (`--models all`)
- [x] Harder hand-written datasets (`notes_v2`, `notes_v3`) and prompt v2
- [ ] Prompt v0 vs v1 comparison
- [ ] Scale to 75–100 notes; optional LLM rewrite pass (`--rewrite-model`), with 20 rewritten notes checked by hand
  - **Fix before scaling:** `notes_v3` note `note_adv_v2_0009` ("diabetes educator call in 1 week") has gold `referrals: []`, but "diabetes education" was graded an acceptable extraction (verdict `ok`) and is flagged out-of-vocabulary, so it counts as a strict false positive. Decide on the label, add it to `vocab.json` if kept, and fix the gold in the new dataset file (not in place). `note_adv_v2_0012` "low back pain" was graded `bad` (a symptom, correctly excluded), so no change.
- [ ] Langfuse tracing
- [ ] Reference run committed to `results/reference/` and used as the regression gate
- [ ] Write-up: *How I'd evaluate this for a clinic*
