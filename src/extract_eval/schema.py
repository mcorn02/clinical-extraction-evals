"""The extraction target. Gold labels and model outputs share this shape."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VisitType = Literal["initial_assessment", "follow_up", "telehealth"]

LIST_FIELDS = ["conditions", "medications", "allergies", "referrals"]
SCALAR_FIELDS = ["visit_date", "visit_type", "patient_age", "follow_up_weeks"]
ALL_FIELDS = SCALAR_FIELDS + LIST_FIELDS


class NoteExtraction(BaseModel):
    """Structured facts pulled from one clinical note."""

    model_config = ConfigDict(extra="forbid")

    visit_date: date | None = Field(None, description="Date of service, YYYY-MM-DD.")
    visit_type: VisitType | None = Field(None, description="initial_assessment, follow_up or telehealth.")
    patient_age: int | None = Field(None, ge=0, le=120, description="Age in years.")
    conditions: list[str] = Field(
        default_factory=list,
        description="The patient's own confirmed diagnoses. Excludes denied, ruled-out, "
        "suspected and family-history mentions.",
    )
    medications: list[str] = Field(
        default_factory=list,
        description="Medicines the patient currently takes, lowercase generic names, no dose. "
        "Excludes stopped medicines.",
    )
    allergies: list[str] = Field(
        default_factory=list, description="Allergies. Empty list when the note says none."
    )
    referrals: list[str] = Field(
        default_factory=list,
        description="Referrals placed at this visit. Excludes offered-but-declined referrals.",
    )
    follow_up_weeks: int | None = Field(
        None, ge=0, description="Weeks until the next visit. Null if as-needed or not stated."
    )


class NoteRecord(BaseModel):
    """One row of the dataset: the note, its gold labels and which edge cases it contains."""

    id: str
    text: str
    gold: NoteExtraction
    tags: list[str] = Field(default_factory=list)
    source: str = "template_v1"
    template_text: str | None = None  # kept when the text was rewritten by an LLM
