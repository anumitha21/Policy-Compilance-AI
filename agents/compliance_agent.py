# agents/compliance_agent.py

import json
import re
from langchain_classic.prompts import PromptTemplate
from pydantic import ValidationError
from models.schemas import ComplianceRiskResult
from utils.retry import llm_invoke_with_retry

COMBINED_PROMPT = """
You are a contract compliance and risk assessment agent.

Contract Clause:

{clause}

Policy Evidence:

{evidence}

Return a JSON object with ALL of these fields:

{{
    "compliance_verdict": "compliant" | "non_compliant",
    "compliance_explanation": "...",
    "violated_sections": ["section name"],
    "citations": [{{"section": "...", "excerpt": "..."}}],
    "confidence_score": 0.95,
    "risk_level": "critical" | "major" | "moderate" | "minor" | null,
    "risk_score": 7,
    "risk_explanation": "..." | null
}}

CITATION RULES — read carefully:
- citations must reference POLICY sections only (e.g. "1.1 Notice Period", "6.3 Arbitration").
- NEVER cite the contract clause itself, the contract party names, or "Policy Evidence" as a section.
- The "section" field must be the policy section number and title (e.g. "2.1 Liability Cap").
- The "excerpt" must be the verbatim policy text, not contract text.
- For compliant clauses, still populate citations with the policy sections you checked
  against to confirm compliance — do not return an empty citations array.

SCOPING RULE:
Only flag non_compliant if the clause ACTIVELY VIOLATES a policy requirement.
Do NOT flag for silence or omission unless the policy explicitly mandates that language.

RISK SCORING (only if non_compliant — set all risk fields to null if compliant):
  critical  9-10 : removes mandatory protections / severe legal or regulatory exposure
  major     6-8  : significant liability / procedural violation (skipping steps, missing notice)
  moderate  3-5  : partial non-compliance, limited exposure
  minor     1-2  : administrative or technical violation

IMPORTANT: Procedural violations (skipping mediation, missing notice period) = major, NOT critical.
Critical is reserved for removal of substantive rights.

Return ONLY valid JSON. No preamble, no markdown.
"""


def _load_json(raw):
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


class ComplianceAgent:

    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["clause", "evidence"],
            template=COMBINED_PROMPT
        )

    def _invoke(self, clause, evidence):
        prompt = self.prompt.format(
            clause=clause,
            evidence="\n\n".join(evidence)
        )
        return llm_invoke_with_retry(self.llm, prompt).content

    def run(self, state):
        raw = self._invoke(
            clause=state["masked_text"],
            evidence=state["evidence"]
        )

        try:
            parsed    = _load_json(raw)
            validated = ComplianceRiskResult.model_validate(parsed)

        except (json.JSONDecodeError, ValidationError):
            raw = self._invoke(
                clause=state["masked_text"],
                evidence=state["evidence"]
            )
            try:
                parsed    = _load_json(raw)
                validated = ComplianceRiskResult.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(
                    f"Compliance+Risk agent response invalid after retry: {raw!r}"
                ) from exc

        state["compliance_verdict"]     = validated.compliance_verdict
        state["compliance_explanation"] = validated.compliance_explanation
        state["violated_sections"]      = validated.violated_sections
        state["citations"]              = validated.citations
        state["confidence_score"]       = validated.confidence_score
        state["risk_level"]             = validated.risk_level
        state["risk_score"]             = validated.risk_score
        state["risk_explanation"]       = validated.risk_explanation

        return state
