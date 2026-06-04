from typing import List, Optional, Literal, TypedDict
from pydantic import BaseModel, Field


class Citation(BaseModel):
    section: str
    excerpt: str


# Results produced by individual agents
class ComplianceResult(BaseModel):
    compliance_verdict: Literal["compliant", "non_compliant"]
    compliance_explanation: str
    violated_sections: List[str]
    citations: List[Citation]
    confidence_score: float = Field(..., ge=0.0, le=1.0)


class RiskResult(BaseModel):
    risk_level: Literal["critical", "major", "moderate", "minor"]
    risk_explanation: str
    risk_score: int = Field(..., ge=1, le=10)


class RewriteResult(BaseModel):
    rewritten_clause: str
    rewrite_reasoning: Optional[str] = None


class ValidatorResult(BaseModel):
    validation_passed: bool
    validator_feedback: Optional[str] = None


# Aggregated model stored/exported after all agents finish
class ClauseReviewResult(BaseModel):
    clause_id: str

    compliance_verdict: Optional[Literal["compliant", "non_compliant"]] = None
    compliance_explanation: Optional[str] = None

    risk_score: Optional[int] = None
    risk_level: Optional[Literal["critical", "major", "moderate", "minor"]] = None

    rewritten_clause: Optional[str] = None

    citations: Optional[List[Citation]] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    reasoning: Optional[str] = None
    suggested_rewrite: Optional[str] = None
    manual_review_required: Optional[bool] = None
    manual_review_reason: Optional[str] = None
    failed_rewrites: Optional[List[str]] = None
    violated_sections: Optional[List[str]] = None


class ComplianceState(TypedDict, total=False):
    # Clause input
    clause_id: str
    masked_text: str
    raw_text: str
    evidence: List[str]

    # Compliance agent outputs
    compliance_verdict: str
    compliance_explanation: str
    violated_sections: List[str]
    citations: List[dict]
    confidence_score: float

    # Risk agent outputs
    risk_level: str
    risk_score: int
    risk_explanation: str

    # Rewrite agent outputs
    rewritten_clause: str
    rewrite_reasoning: str

    # Validator agent outputs
    validation_passed: bool
    validator_feedback: str
    rewrite_attempts: int

    # Fallback
    manual_review_flag: bool
    manual_review_reason: str
    failed_rewrites: List[str]
