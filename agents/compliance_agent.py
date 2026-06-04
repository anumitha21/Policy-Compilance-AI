# agents/compliance_agent.py

import json
import re
from langchain_classic.prompts import PromptTemplate

from pydantic import ValidationError

from models.schemas import ComplianceResult


COMPLIANCE_PROMPT = """
You are an AI Contract Compliance Reviewer.

Contract Clause:

{clause}

Policy Evidence:

{evidence}

Determine:

1. compliant OR non_compliant

2. compliance explanation

3. violated policy sections

4. citations

5. confidence score

IMPORTANT SCOPING RULE:
Only flag a clause as non_compliant if it contains language that
ACTIVELY VIOLATES a policy requirement.
Do NOT flag a clause as non_compliant solely because it is silent
on or omits a topic covered by policy. Absence of language is not
a violation unless the policy explicitly requires that language to
be present in every contract clause.

Return JSON only.

Schema:

{{
    "compliance_verdict":
        "compliant" | "non_compliant",

    "compliance_explanation":
        "...",

    "violated_sections":
        ["section"],

    "citations":
    [
        {{
            "section":"...",
            "excerpt":"..."
        }}
    ],

    "confidence_score":
        0.95
}}
"""


def _load_json(raw):
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


class ComplianceAgent:

    def __init__(
        self,
        llm
    ):

        self.llm = llm

        self.prompt = PromptTemplate(
            input_variables=[
                "clause",
                "evidence"
            ],
            template=
                COMPLIANCE_PROMPT
        )

    # =====================================
    # LLM CALL
    # =====================================

    def _invoke(
        self,
        clause,
        evidence
    ):

        prompt = self.prompt.format(
            clause=clause,
            evidence="\n\n".join(
                evidence
            )
        )

        response = self.llm.invoke(
            prompt
        )

        return response.content

    # =====================================
    # VALIDATED CALL
    # =====================================

    def run(
        self,
        state
    ):

        raw = self._invoke(
            clause=state[
                "masked_text"
            ],
            evidence=state[
                "evidence"
            ]
        )

        try:

            parsed = _load_json(
                raw
            )

            validated = ComplianceResult.model_validate(parsed)

        except (json.JSONDecodeError, ValidationError):

            # Retry once
            raw = self._invoke(
                clause=state[
                    "masked_text"
                ],
                evidence=state[
                    "evidence"
                ]
            )

            try:
                parsed = _load_json(
                    raw
                )

                validated = ComplianceResult.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(
                    "Compliance agent response invalid after retry: "
                    f"{raw!r}"
                ) from exc

        state[
            "compliance_verdict"
        ] = validated.compliance_verdict

        state[
            "compliance_explanation"
        ] = (
            validated
            .compliance_explanation
        )

        state[
            "violated_sections"
        ] = (
            validated
            .violated_sections
        )

        state[
            "citations"
        ] = validated.citations

        state[
            "confidence_score"
        ] = (
            validated
            .confidence_score
        )

        return state