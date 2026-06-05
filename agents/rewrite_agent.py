# agents/rewrite_agent.py

import json
import re
from models.schemas import RewriteResult
from pydantic import ValidationError
from langchain_classic.prompts import PromptTemplate
from utils.retry import llm_invoke_with_retry


REWRITE_PROMPT = """
You are a contract remediation specialist.

Original Clause:

{clause_text}

Policy Evidence:

{evidence}

Violated Sections:

{violated_sections}

Policy Citations:

{citations}

Rewrite the clause.

MANDATORY RULES:

1. Preserve original business intent
2. Modify ONLY violating language
3. Do NOT introduce new obligations
4. Align with cited policy sections
5. Keep legal drafting style
6. Produce a complete clause
7.Do not reference internal policy section numbers in the rewritten clause. Write only contract language.

POLICY-SPECIFIC RULES (override clause language if conflicting):

Termination:
- Even for material breach, termination is NOT immediate.
- The non-breaching party MUST first issue a 14-day written cure
  notice per Section 1.3 before termination takes effect.
- Do NOT preserve or reintroduce "immediately" or "without prior
  notice" for any termination scenario.

Dispute Resolution:
- The correct resolution path per policy is:
  1. Mediation (30 days minimum)
  2. If mediation fails → binding arbitration under AAA rules
- Do NOT preserve court-based litigation as the resolution
  mechanism. Replace any reference to courts with AAA arbitration.

Return JSON only.

{{
    "rewritten_clause":"..."
}}
"""


def _load_json(raw):
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


class RewriteAgent:

    def __init__(self, llm):

        self.llm = llm

        self.prompt = PromptTemplate(
            input_variables=[
                "clause_text",
                "evidence",
                "violated_sections",
                "citations"
            ],
            template=REWRITE_PROMPT
        )

    # ====================================
    # LLM CALL
    # ====================================

    def _invoke(
        self,
        clause_text,
        evidence,
        violated_sections,
        citations
    ):

        prompt = self.prompt.format(
            clause_text=clause_text,
            evidence="\n\n".join(
                evidence
            ),
            violated_sections="\n".join(
                violated_sections
            ),
            citations=json.dumps(
            [c.model_dump() for c in citations],
            indent=2
           )
        )

        response = llm_invoke_with_retry(self.llm, prompt)

        return response.content

    # ====================================
    # MAIN ENTRY
    # ====================================

    def run(self, state):


    # existing code below

        raw = self._invoke(
            
            clause_text=state["masked_text"],
            evidence=state[
                "evidence"
            ],
            violated_sections=state[
                "violated_sections"
            ],
            citations=state[
                "citations"
            ]
        )

        try:

            parsed = _load_json(raw)


            validated = RewriteResult.model_validate(parsed)

        except (json.JSONDecodeError, ValidationError):

            raw = self._invoke(
                clause_text=state["clause_text"],
                evidence=state["evidence"],
                violated_sections=state["violated_sections"],
                citations=state["citations"]
            )

            try:
                parsed = _load_json(raw)
                validated = RewriteResult.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(
                    "Rewrite agent response invalid after retry: "
                    f"{raw!r}"
                ) from exc

        state["rewritten_clause"] = (
            parsed.get(
                "rewritten_clause"
            )
        )

        return state