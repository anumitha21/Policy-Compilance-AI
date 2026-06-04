# agents/risk_agent.py

import json
import re

from langchain_classic.prompts import PromptTemplate
from pydantic import ValidationError

from models.schemas import RiskResult


RISK_PROMPT = """
You are a contract risk assessor.

Compliance Explanation:

{explanation}

Violated Sections:

{violated_sections}

Classify the violation into ONE category:

critical
major
moderate
minor

Definitions:

critical (score 9-10)
- removes or waives a mandatory contractual protection
- creates severe legal or regulatory exposure
- exposes a party to unlimited liability
- Example: removing a required indemnity, waiving data protection rights

major (score 6-8)
- significant liability or obligation issue
- violates a procedural requirement (e.g. skipping mediation, missing notice period)
- materially alters contractual rights without removing them entirely
- Example: skipping dispute resolution steps, shortening mandatory notice periods

moderate (score 3-5)
- partial non-compliance with limited exposure
- minor deviation from required process or language
- Example: incomplete disclosure language, imprecise termination wording

minor (score 1-2)
- administrative or technical violation
- no material impact on rights or obligations
- Example: missing clause numbering, incorrect date format

IMPORTANT: Procedural violations (skipping steps, missing notice) are major, NOT critical.
Critical is reserved for removal of substantive rights or severe legal/regulatory exposure.

Return JSON only.

{{
    "risk_level":"critical|major|moderate|minor",
    "risk_explanation":"..."
}}
"""


RISK_MAPPING = {
    "critical": 10,
    "major": 7,
    "moderate": 4,
    "minor": 2
}


def _load_json(raw):
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


class RiskAgent:

    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["explanation", "violated_sections"],
            template=RISK_PROMPT
        )

    def _invoke(self, explanation, violated_sections):
        prompt = self.prompt.format(
            explanation=explanation,
            violated_sections="\n".join(violated_sections)
        )
        return self.llm.invoke(prompt).content

    def run(self, state):
        raw = self._invoke(
            explanation=state["compliance_explanation"],
            violated_sections=state["violated_sections"]
        )

        try:
            parsed = _load_json(raw)
            parsed["risk_score"] = RISK_MAPPING[parsed["risk_level"]]
            validated = RiskResult.model_validate(parsed)

        except (json.JSONDecodeError, ValidationError, KeyError):
            raw = self._invoke(
                explanation=state["compliance_explanation"],
                violated_sections=state["violated_sections"]
            )
            try:
                parsed = _load_json(raw)
                parsed["risk_score"] = RISK_MAPPING[parsed["risk_level"]]
                validated = RiskResult.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError, KeyError) as exc:
                raise ValueError(
                    f"Risk agent response invalid after retry: {raw!r}"
                ) from exc

        state["risk_level"] = validated.risk_level
        state["risk_score"] = validated.risk_score
        state["risk_explanation"] = validated.risk_explanation

        return state
