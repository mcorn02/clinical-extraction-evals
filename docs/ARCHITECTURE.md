# Architecture

## What gets extracted

Each note is turned into one `NoteExtraction` object (`schema.py`):

| Field | Type | Example from note text → extracted value |
|---|---|---|
| `visit_date` | date | "Seen 09/26/2026" → `2026-09-26` |
| `visit_type` | one of 3 | "Telehealth encounter" → `telehealth` |
| `patient_age` | int | "66 y/o F" → `66` |
| `conditions` | list | "Diagnoses: ASD and seizure disorder. Denies history of OSA." → `["autism spectrum disorder", "epilepsy"]` (OSA is denied, so it's left out) |
| `medications` | list | "Takes Keppra 250 mg BID. Previously on Prilosec, stopped." → `["levetiracetam"]` |
| `allergies` | list | "Allergic to shellfish and sulfa drugs." → `["shellfish", "sulfa drugs"]` |
| `referrals` | list | "Will refer to neuro." → `["neurology"]` |
| `follow_up_weeks` | int | "Return in four weeks." → `4` |

List values are always the **canonical names** from `data/vocab.json` (16 conditions, 14 medications, 5 allergies, 8 referrals), each with aliases and brand names.

## Two extractors, not a pipeline

It's **not** regex followed by an LLM. There are two interchangeable extractors, and `--extractor` picks one per run. Both produce the same `NoteExtraction` and are scored the same way, so they're directly comparable.

- **Baseline** (`--extractor baseline`): regex only. It matches every alias or brand name from the vocab and pulls out dates, age and follow-up with patterns. It's free and offline, but it can't read context, so "Denies asthma" still yields `asthma`. It's the floor an LLM has to beat.
- **LLM** (`--extractor llm`): one model call per note, no regex. The model gets a system prompt (rules, plus the canonical names, plus the JSON schema) and the note text, and replies with JSON.

## Components and data flow

```mermaid
flowchart TD
    V[(data/vocab.json<br/>canonical names, aliases,<br/>brands, doses)]

    subgraph GEN["1 · Make the dataset (offline, run once)"]
        G1[generate.sample_record<br/>pick the answers first:<br/>conditions, meds, allergies,<br/>referrals, follow-up, date<br/>+ distractors]
        G2[generate.render_note<br/>write note text from the answers<br/>+ tag edge cases]
        G1 --> G2
    end
    V --> G1
    G2 --> D[(data/synthetic/notes_v1.jsonl<br/>text + gold + tags)]
    G2 -. optional --rewrite-model .-> LLMW[llm.call_llm<br/>realism rewrite]

    subgraph RUN["2 · run_eval.py (per run)"]
        L[load_dataset] --> POOL[ThreadPoolExecutor<br/>4 notes in parallel]
        POOL -->|--extractor baseline| B[BaselineExtractor<br/>regex over vocab surface forms]
        POOL -->|--extractor llm| X[LLMExtractor]
    end
    D --> L
    V --> B

    subgraph LLMX["LLMExtractor, per note"]
        P[build_system_prompt<br/>prompts/extract_v1.md<br/>fills in VOCAB and SCHEMA] --> C1[llm.call_llm → LiteLLM → provider]
        C1 --> J{parse JSON +<br/>Pydantic validate}
        J -->|invalid, 1st time| R[append the error and<br/>ask again: retry once] --> C1
        J -->|ok| OK[NoteExtraction]
        J -->|invalid twice| F[prediction = None]
        C1 -->|exception| E[API error<br/>prediction = None]
    end
    X --> P
    V --> P

    B --> S
    OK --> S
    F --> S
    E --> S
    subgraph SCORE["3 · metrics.py"]
        S[score_note<br/>normalize via vocab,<br/>set compare vs gold:<br/>TP / FP / FN per field] --> A[aggregate<br/>per field, overall,<br/>exact match,<br/>by edge-case tag]
    end
    D -- gold --> S
    A --> OUT[(results/run_id/<br/>report.md · metrics.json ·<br/>predictions.jsonl)]
    OUT -. --compare .-> REG[check_regression<br/>exit 1 if a field's F1 drops]

    C1 -. every attempt .-> TR[(results/run_id/llm_calls.jsonl)]
    RUN -. config, per-note status,<br/>retries, API errors, timing .-> LOG[(results/run_id/run.log)]
    P -. once per run .-> SP[(results/run_id/system_prompt.txt)]
```

## What each run writes (`results/<run_id>/`)

| File | What's in it | Use it to |
|---|---|---|
| `report.md` | Precision, recall and F1 per field and per edge case; failures by type; latency; wall clock and parallelism; cost per 1k notes | Read the headline result |
| `metrics.json` | The same numbers in machine-readable form, plus run config and stats | Compare runs; regression gate |
| `predictions.jsonl` | Per note: prediction, gold, the FP/FN for each field, raw reply, `error_type`, `attempts` | Error analysis |
| `llm_calls.jsonl` | **One line per LLM attempt**, retries and failures included: start and end timestamps, latency, tokens, cost, status (`ok`, `invalid_output`, `rate_limit`, `auth`, `timeout`, …), raw reply, retry feedback, worker thread | Debug a note; find slow calls; see queueing and rate limits |
| `run.log` | Human-readable: config and git commit, one line per note, warnings for retries and API errors, summary | Follow a run while it's in progress (`tail -f`) |
| `system_prompt.txt` | The exact prompt the model saw (its sha256 is in every trace line) | Reproduce a run |

API errors are classified by type (`llm.classify_error`), and error messages are redacted so keys never reach the logs. Notes that failed on an API error are left out of cost per 1k and counted separately.

Datasets: `generate` refuses to overwrite an existing file (use `--out data/synthetic/notes_v2.jsonl`, or `--force` deliberately) and writes a `notes_vN.meta.json` sidecar recording the seed, n, vocab hash, git commit and file hash.

## One LLM call, step by step

```mermaid
sequenceDiagram
    participant RE as run_eval
    participant EX as LLMExtractor
    participant CL as llm.call_llm
    participant LT as LiteLLM
    participant M as Model provider

    RE->>EX: extractor(note.text)
    EX->>CL: [system: prompt+vocab+schema, user: note]
    CL->>LT: completion(model, temperature=0)
    LT->>M: HTTPS request (key from .env)
    M-->>LT: JSON text + token usage
    LT-->>CL: response
    CL-->>EX: text, latency, tokens, cost
    alt valid JSON matching the schema
        EX-->>RE: prediction
    else invalid
        EX->>CL: same messages + "That output failed validation: …"
        CL-->>EX: second reply
        EX-->>RE: prediction, or None if still invalid
    end
```

## Scoring in one picture

For every field, the prediction and the gold label are each turned into a **set** of normalized values ("Zoloft 50 mg" → `sertraline`). Then:

```
gold = {asthma, ASD}        pred = {asthma, OSA}
TP = pred ∩ gold = {asthma}   → 1
FP = pred − gold = {OSA}      → 1   (invented or not excluded)
FN = gold − pred = {ASD}      → 1   (missed)
```

Single-value fields are sets of size 0 or 1, so a wrong age counts as 1 FP and 1 FN, and a missing age counts as 1 FN. Counts are summed across all notes before computing precision, recall and F1.
