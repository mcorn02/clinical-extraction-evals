# claude-haiku-5-5__extract_v1

Data: `data/synthetic/notes_v1.jsonl` (20 notes) · Run: 20261111-195031_claude-haiku-5-5__extract_v1

| Field | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| visit_date | 100.0 | 100.0 | 100.0 | 20 | 0 | 0 |
| visit_type | 100.0 | 100.0 | 100.0 | 20 | 0 | 0 |
| patient_age | 100.0 | 100.0 | 100.0 | 20 | 0 | 0 |
| follow_up_weeks | 100.0 | 100.0 | 100.0 | 18 | 0 | 0 |
| conditions | 100.0 | 100.0 | 100.0 | 41 | 0 | 0 |
| medications | 100.0 | 100.0 | 100.0 | 30 | 0 | 0 |
| allergies | 100.0 | 100.0 | 100.0 | 16 | 0 | 0 |
| referrals | 100.0 | 100.0 | 100.0 | 18 | 0 | 0 |
| **overall** | **100.0** | **100.0** | **100.0** | 183 | 0 | 0 |

- Notes with every field correct: 100.0%
- Failed or invalid outputs: 0 (retries used: 0)
- Latency per note: mean 1.83s, p95 3.62s
- Wall clock: 10.01s; sum of call latency 36.65s (parallelism 3.7x)
- Tokens: 33,747 in / 4,945 out
- Cost per 1,000 notes: $0.29

## By edge case (scored on the field each one targets)

| Edge case | Field(s) | Notes | Precision | Recall | FP | FN |
|---|---|---:|---:|---:|---:|---:|
| alias | conditions, allergies, referrals | 19 | 100.0 | 100.0 | 0 | 0 |
| brand_name | medications | 11 | 100.0 | 100.0 | 0 | 0 |
| declined_referral | referrals | 8 | 100.0 | 100.0 | 0 | 0 |
| discontinued_med | medications | 9 | 100.0 | 100.0 | 0 | 0 |
| family_history | conditions | 5 | 100.0 | 100.0 | 0 | 0 |
| negation | conditions | 13 | 100.0 | 100.0 | 0 | 0 |
| no_allergies | allergies | 9 | – | – | 0 | 0 |
| number_words | follow_up_weeks | 4 | 100.0 | 100.0 | 0 | 0 |
| rule_out | conditions | 5 | 100.0 | 100.0 | 0 | 0 |
