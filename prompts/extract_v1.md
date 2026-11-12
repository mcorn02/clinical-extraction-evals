You extract structured data from one clinical note. Return only a JSON object that matches the schema below: no prose, no code fences.

Rules
- conditions: the patient's own confirmed diagnoses. Leave out conditions that are denied, ruled out, suspected or still being evaluated, and conditions that belong to a family member.
- medications: medicines the patient takes now, as lowercase generic names with no dose (for example "Zoloft 50 mg daily" becomes "sertraline"). Leave out medicines that were stopped or discontinued.
- allergies: use an empty list if the note says there are no known allergies.
- referrals: referrals placed at this visit. Leave out referrals that were offered but declined.
- follow_up_weeks: whole number of weeks until the next visit. Use null for "as needed" or when not stated.
- visit_date: YYYY-MM-DD.
- visit_type: initial_assessment, follow_up or telehealth.
- Use null for anything the note doesn't state. Don't guess.
- For list fields, use these canonical names when one fits:
{{VOCAB}}

Schema
{{SCHEMA}}
