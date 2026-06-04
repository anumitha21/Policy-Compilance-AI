# guardrails/rails.py

from pydantic import BaseModel, Field
from typing import Literal


# =====================================================
# COMPLIANCE AGENT
# =====================================================

class ComplianceGuard(BaseModel):
    compliance_verdict: Literal[
        "compliant",
        "non_compliant"
    ]

    compliance_explanation: str

    violated_sections: list[str]

    citations: list[dict]

    confidence_score: float = Field(
        ge=0.0,
        le=1.0
    )


# =====================================================
# RISK AGENT
# =====================================================

class RiskGuard(BaseModel):
    risk_level: Literal[
        "critical",
        "major",
        "moderate",
        "minor"
    ]

    risk_score: int = Field(
        ge=1,
        le=10
    )

    risk_explanation: str


# =====================================================
# REWRITE AGENT
# =====================================================

class RewriteGuard(BaseModel):
    rewritten_clause: str = Field(
        min_length=1
    )


# =====================================================
# VALIDATOR AGENT
# =====================================================

class ValidatorGuard(BaseModel):
    validation_passed: bool

    validator_feedback: str


# =====================================================
# VALIDATION HELPER
# =====================================================

def validate_or_raise(
    schema,
    data: dict
):
    """
    Validates agent output.

    Raises ValidationError if invalid.
    """

    return schema.model_validate(data)