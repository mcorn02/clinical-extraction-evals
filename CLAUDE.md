# CLAUDE.md

Eval kit for LLM extraction from synthetic clinical-style notes. See README.md for the why.

## Commands
- Install: `pip install -e ".[dev]"`
- Tests: `pytest -q` (no network; LLM calls are faked)
- Generate data: `python -m extract_eval.generate --n 20 --seed 0`
- Baseline run: `python -m extract_eval.run_eval --extractor baseline`
- LLM run: `python -m extract_eval.run_eval --extractor llm --model <litellm model> --prompt prompts/extract_v1.md`

## Rules
- Synthetic data only. Never add real patient notes or anything copied from an employer's systems.
- Prompts are versioned. To change a prompt, copy it to the next `extract_vN.md`; never edit one that has published results.
- Don't regenerate an existing dataset file in place once results reference it. Write a new file (`notes_v2.jsonl`).
- Gold labels must stay derivable from the sampled record. If you change `generate.py`, `tests/test_generate.py` must still pass.
- Every model call goes through `llm.call_llm` so cost, tokens and latency are recorded the same way.
- Run `pytest -q` before committing.
