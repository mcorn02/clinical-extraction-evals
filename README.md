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

### Round 3: prompt versions (Claude Haiku 5.5, `notes_v3`)

**Question:** Does prompt wording change accuracy, and what does it cost?
Prompt wording does not change the accuracy significantly, but does increase cost ever so slightly from $0.25 for v0 to $0.28 for v1 and $0.28 to $0.30 for v3.
**Setup:** Claude Haiku 5.5 on `notes_v3` (24 notes). Only the prompt changes. v0 and v1 were run once each; v2 was run 5 times to measure run-to-run noise.

| Prompt | F1 | Conditions F1 | Referrals F1 | Exact notes | Mean latency | Cost / 1k notes |
|---|---:|---:|---:|---:|---:|---:|
| v0 | 96.1 | 85.3 | 87.0 | 70.8% | 1.90s | $0.25 |
| v1 | 99.7 | 98.7 | 100.0 | 95.8% | 1.66s | $0.28 |
| v2 (mean of 5) | 99.6 | 99.0 | 98.3 | 95.0% | 1.66s | $0.29 |

v2 ranged from 98.9 to 100.0 F1 and from 91.7% to 100% exact across the 5 runs.

**What changed between prompts:**
- v0 to v1: a 16-word instruction became a 165-word prompt with per-field rules (leave out denied, ruled-out, family and discontinued items; no-allergy handling; canonical names from the vocab).
- v1 to v2: two added lines. Resolved or past conditions are excluded, and a model may give a short standard name when no canonical one fits.

**Findings**
- v0 to v1 is a real gain: +3.6 F1 and about 6 more notes exactly right, which is outside the v2 run-to-run spread. v0's errors were all in conditions and referrals, including `not_ruled_out`, `referral_status_mix`, `table_format`, `resolved_condition` and `symptom_not_dx`.
- v1 and v2 are not distinguishable. The 1-note gap between them is smaller than v2's own spread over 5 runs.
- Cost barely moves. v1 and v2 send about 60% more input tokens than v0 (about 40k vs 25k), which adds about $0.03 per 1,000 notes. Latency is the same or lower.

**Persistent errors:**
- `note_adv_v2_0012`: the model extracts "low back pain" as a condition when it is a symptom (`symptom_not_dx`). Graded `bad` (a hallucination) in every run, and the same error appears for all five models in Round 2.
- `note_adv_v2_0009`: the model extracts "diabetes education" as a referral, but the note only mentions a "diabetes educator call" and the gold has no referral. It appeared in 2 of 4 v2 reruns. Graded `ok` (a vocab gap, not a hallucination), and the gold still needs a decision.

**Caveats:**
- 24 notes: one note is about 4 points of exact-match.
- One model and one dataset.
- v0 and v1 were each run once. v2's spread shows a single run can move by up to 2 notes.
- Scores are strict and pre-alias, as saved. 7 of v0's 8 false positives were out-of-vocabulary values hand-graded as correct (e.g. "type 2 diabetes mellitus"), so v0's strict score understates it. Three aliases added after these runs would absorb 5 of them.
- Per-run folders are local (`results/` is gitignored): `20261114-184034` (v0), `184055` (v1), and `184113`, `184250`, `184306`, `184321`, `184338` (v2).

**Next:** fix the gold for 0009 and add the missing referral name to the vocab in the scale-up dataset (new file, not in place), then rerun on 75-100 notes.

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
- [x] Prompt v0 vs v1 vs v2 comparison
- [ ] Scale to 75–100 notes; optional LLM rewrite pass (`--rewrite-model`), with 20 rewritten notes checked by hand
  - **Fix before scaling:** `notes_v3` note `note_adv_v2_0009` ("diabetes educator call in 1 week") has gold `referrals: []`, but "diabetes education" was graded an acceptable extraction (verdict `ok`) and is flagged out-of-vocabulary, so it counts as a strict false positive. Decide on the label, add it to `vocab.json` if kept, and fix the gold in the new dataset file (not in place). `note_adv_v2_0012` "low back pain" was graded `bad` (a symptom, correctly excluded), so no change.
- [ ] Langfuse tracing
- [ ] Reference run committed to `results/reference/` and used as the regression gate
- [ ] Write-up: *How I'd evaluate this for a clinic*
