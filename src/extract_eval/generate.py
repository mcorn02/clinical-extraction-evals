"""Generate synthetic clinical-style notes with known labels.

The trick: sample the structured record first, then write the note from it. The labels come
for free, and we control exactly which edge cases each note contains.

    python -m extract_eval.generate --n 20 --seed 0 --out data/synthetic/notes_v1.jsonl
"""

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from .schema import NoteExtraction, NoteRecord
from .vocab import Vocab, load_vocab

NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 6: "six", 8: "eight", 12: "twelve"}
RELATIVES = ["mother", "father", "maternal grandmother", "paternal uncle", "older brother"]
VISIT_PHRASES = {
    "initial_assessment": ["Initial assessment", "New patient intake", "Initial evaluation"],
    "follow_up": ["Follow-up visit", "F/U visit", "Return visit"],
    "telehealth": ["Telehealth visit (video)", "Virtual visit via video", "Telehealth encounter"],
}
FILLERS = {
    "any": ["Vitals stable.", "Sleep and appetite discussed.", "Patient engaged and cooperative."],
    "peds": ["Caregiver present for the visit.", "Discussed school accommodations.", "Reviewed growth chart."],
    "adult": ["Discussed work schedule.", "Reviewed home BP log.", "Exercise habits reviewed."],
}


def _eligible(population: str, entry_pop: str) -> bool:
    return entry_pop == "any" or entry_pop == population


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _surface(rng: random.Random, vocab: Vocab, field: str, canonical: str, tags: set) -> str:
    forms = vocab.surface_forms(field, canonical)
    if len(forms) > 1 and rng.random() < 0.5:
        form = rng.choice(forms[1:])
        tags.add("brand_name" if field == "medications" else "alias")
        return form
    return forms[0]


def _fmt_date(rng: random.Random, d: date) -> str:
    return rng.choice([
        d.strftime("%m/%d/%Y"),
        d.isoformat(),
        f"{d:%B} {d.day}, {d.year}",
    ])


def sample_record(rng: random.Random, vocab: Vocab) -> tuple[NoteExtraction, dict]:
    """Sample gold labels plus the distractors that must NOT end up in the labels."""
    population = rng.choice(["peds", "adult"])
    age = rng.randint(3, 17) if population == "peds" else rng.randint(18, 80)

    conds_all = vocab.raw["conditions"]
    eligible_conds = [c for c, e in conds_all.items() if _eligible(population, e["population"])]
    conditions = rng.sample(eligible_conds, k=rng.randint(1, 3))

    meds: list[str] = []
    for c in conditions:
        options = [m for m in conds_all[c]["meds"] if m not in meds]
        if options and rng.random() < 0.8:
            meds.append(rng.choice(options))

    allergy_names = list(vocab.raw["allergies"])
    allergies = [] if rng.random() < 0.4 else rng.sample(allergy_names, k=rng.randint(1, 2))

    refs_all = vocab.raw["referrals"]
    eligible_refs = [r for r, e in refs_all.items() if _eligible(population, e["population"])]
    referrals = rng.sample(eligible_refs, k=rng.choice([0, 1, 1, 2]))

    follow_up = rng.choice([2, 4, 6, 8, 12, None])
    visit_type = rng.choice(list(VISIT_PHRASES))
    visit_date = date(2025, 1, 1) + timedelta(days=rng.randint(0, 670))

    # Distractors: things mentioned in the note that are NOT part of the answer.
    unused_conds = [c for c in conds_all if c not in conditions and _eligible(population, conds_all[c]["population"])]
    rng.shuffle(unused_conds)
    distractors = {
        "negated": unused_conds.pop() if rng.random() < 0.5 else None,
        "family_history": unused_conds.pop() if rng.random() < 0.4 else None,
        "rule_out": unused_conds.pop() if rng.random() < 0.3 else None,
    }
    unused_meds = [m for m in vocab.raw["medications"] if m not in meds]
    distractors["discontinued_med"] = rng.choice(unused_meds) if rng.random() < 0.35 else None
    unused_refs = [r for r in eligible_refs if r not in referrals]
    distractors["declined_referral"] = rng.choice(unused_refs) if rng.random() < 0.3 else None

    gold = NoteExtraction(
        visit_date=visit_date,
        visit_type=visit_type,
        patient_age=age,
        conditions=[c.lower() for c in conditions],
        medications=[m.lower() for m in meds],
        allergies=[a.lower() for a in allergies],
        referrals=[r.lower() for r in referrals],
        follow_up_weeks=follow_up,
    )
    context = {"population": population, "sex": rng.choice(["M", "F"]), "distractors": distractors}
    return gold, context


def render_note(rng: random.Random, vocab: Vocab, gold: NoteExtraction, ctx: dict) -> tuple[str, list[str]]:
    tags: set[str] = set()
    d = ctx["distractors"]
    sex = ctx["sex"]
    peds = ctx["population"] == "peds"
    sex_word = ("boy" if sex == "M" else "girl") if peds else ("male" if sex == "M" else "female")
    age = gold.patient_age

    header = f"{rng.choice(VISIT_PHRASES[gold.visit_type])} - {rng.choice(['Date of service:', 'DOS', 'Seen'])} {_fmt_date(rng, gold.visit_date)}"
    intro = rng.choice([
        f"{age} y/o {sex}.",
        f"{age}-year-old {sex_word}.",
        f"Pt is a {age} yo {sex_word}.",
    ])

    # History
    cond_text = _join([_surface(rng, vocab, "conditions", c, tags) for c in gold.conditions])
    history = [rng.choice([f"PMH: {cond_text}.", f"Hx of {cond_text}.", f"Diagnoses: {cond_text}."])]
    if d["negated"]:
        tags.add("negation")
        x = _surface(rng, vocab, "conditions", d["negated"], tags)
        history.append(rng.choice([f"Denies history of {x}.", f"No hx of {x}.", f"Negative for {x}."]))
    if d["family_history"]:
        tags.add("family_history")
        x = _surface(rng, vocab, "conditions", d["family_history"], tags)
        rel = rng.choice(RELATIVES)
        history.append(rng.choice([f"Family history significant for {x} ({rel}).", f"FHx: {rel} with {x}."]))
    if d["rule_out"]:
        tags.add("rule_out")
        x = _surface(rng, vocab, "conditions", d["rule_out"], tags)
        history.append(rng.choice([
            f"R/O {x}; screening forms sent home.",
            f"Concern for possible {x}, not diagnosed at this time.",
            f"Rule out {x} pending further evaluation.",
        ]))

    # Medications
    meds = []
    for m in gold.medications:
        dose = rng.choice(vocab.raw["medications"][m]["doses"])
        meds.append(f"{_surface(rng, vocab, 'medications', m, tags)} {dose}")
    if meds:
        med_lines = [rng.choice([f"Current meds: {_join(meds)}.", f"Medications: {_join(meds)}.", f"Takes {_join(meds)}."])]
    else:
        med_lines = [rng.choice(["No current medications.", "Meds: none."])]
    if d["discontinued_med"]:
        tags.add("discontinued_med")
        x = _surface(rng, vocab, "medications", d["discontinued_med"], tags)
        med_lines.append(rng.choice([
            f"Discontinued {x} last month due to side effects.",
            f"Previously on {x}, stopped in the spring.",
            f"{x} was stopped at last visit.",
        ]))

    # Allergies
    if gold.allergies:
        al = _join([_surface(rng, vocab, "allergies", a, tags) for a in gold.allergies])
        allergy_line = rng.choice([f"Allergies: {al}.", f"Allergic to {al}."])
    else:
        tags.add("no_allergies")
        allergy_line = rng.choice(["No known allergies.", "NKA."])

    # Plan
    plan = []
    for r in gold.referrals:
        x = _surface(rng, vocab, "referrals", r, tags)
        plan.append(rng.choice([f"Refer to {x}.", f"Referral placed to {x}.", f"Will refer to {x}."]))
    if d["declined_referral"]:
        tags.add("declined_referral")
        x = _surface(rng, vocab, "referrals", d["declined_referral"], tags)
        who = "family" if peds else "pt"
        plan.append(rng.choice([f"Offered referral to {x}; {who} declined for now.", f"Discussed {x} referral, {who} declined."]))
    if gold.follow_up_weeks is None:
        plan.append(rng.choice(["Follow up PRN.", "Return as needed."]))
    else:
        n = gold.follow_up_weeks
        if rng.random() < 0.3:
            tags.add("number_words")
            plan.append(f"Return in {NUMBER_WORDS[n]} weeks.")
        else:
            plan.append(rng.choice([f"RTC in {n} wks.", f"Follow up in {n} weeks."]))

    filler = rng.sample(FILLERS["any"] + FILLERS[ctx["population"]], k=rng.randint(0, 2))

    if rng.random() < 0.5:  # sectioned note
        parts = [
            header,
            "S: " + " ".join([intro, *history]),
            "Meds/Allergies: " + " ".join([*med_lines, allergy_line]),
            "A/P: " + " ".join([*filler, *plan]),
        ]
        text = "\n".join(parts)
    else:  # narrative note
        text = header + "\n" + " ".join([intro, *history, *med_lines, allergy_line, *filler, *plan])
    return text, sorted(tags)


def generate(n: int, seed: int, vocab: Vocab | None = None) -> list[NoteRecord]:
    vocab = vocab or load_vocab()
    rng = random.Random(seed)
    records = []
    for i in range(n):
        gold, ctx = sample_record(rng, vocab)
        text, tags = render_note(rng, vocab, gold, ctx)
        records.append(NoteRecord(id=f"note_{seed}_{i:04d}", text=text, gold=gold, tags=tags))
    return records


REWRITE_PROMPT = """Rewrite this synthetic clinical note so it reads like a real clinician wrote it.
Keep every fact exactly: dates, age, diagnoses, medications and doses, allergies, referrals, follow-up timing.
Keep every negation, family-history mention, rule-out, stopped medicine and declined referral, with the same meaning.
Do not add any new diagnosis, medicine, allergy or referral. Return only the rewritten note."""


def rewrite_with_llm(records: list[NoteRecord], model: str) -> list[NoteRecord]:
    """Optional realism pass. Labels can drift here, so spot-check rewritten notes by hand."""
    from .llm import call_llm

    out = []
    for r in records:
        resp = call_llm(model, [
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": r.text},
        ])
        out.append(r.model_copy(update={
            "template_text": r.text,
            "text": resp.text.strip(),
            "source": f"{r.source}+rewrite:{model}",
        }))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/synthetic/notes_v1.jsonl")
    p.add_argument("--rewrite-model", default=None, help="e.g. gpt-5-mini; optional LLM realism pass")
    args = p.parse_args()

    records = generate(args.n, args.seed)
    if args.rewrite_model:
        from dotenv import load_dotenv

        load_dotenv()
        records = rewrite_with_llm(records, args.rewrite_model)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

    tag_counts: dict[str, int] = {}
    for r in records:
        for t in r.tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    print(f"Wrote {len(records)} notes to {out}")
    print("Edge cases:", json.dumps(dict(sorted(tag_counts.items())), indent=None))


if __name__ == "__main__":
    main()
