import json
from typing import Literal

from pydantic import BaseModel, Field, ValidationError


class QAReport(BaseModel):
    summary: str
    issue_category: str
    resolution_status: Literal["resolved", "unresolved", "escalated", "unclear"]
    greeting_present: bool
    professionalism_score: int = Field(ge=1, le=5)
    compliance_flags: list[str]
    action_items: list[str]
    notes: str


def validate_json(raw_text: str) -> tuple[bool, QAReport | str]:
    """Parse and schema-validate a model's raw output.

    Returns (True, QAReport) on success, or (False, error message) on failure.
    Failures are not retried here — the caller records them as-is so the
    JSON-validity/failure-rate metrics reflect the model's actual behavior.
    """
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}"

    try:
        return True, QAReport.model_validate(parsed)
    except ValidationError as e:
        return False, f"schema mismatch: {e}"
