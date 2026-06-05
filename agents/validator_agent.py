# agents/validator_agent.py

import json
import re
from langchain_classic.prompts import PromptTemplate
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from models.schemas import ValidatorResult
from utils.retry import llm_invoke_with_retry


VALIDATOR_PROMPT = """
You are a contract compliance validator.

Rewritten Clause:

{rewritten_clause}

Violated Sections:

{violated_sections}

Evidence:

{evidence}

Determine whether the rewritten
clause now satisfies the cited
policy sections.

Return JSON only.

{{
    "validation_passed": true,
    "validator_feedback": "..."
}}
"""


def _load_json(raw):
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


class ValidatorAgent:

    SIMILARITY_THRESHOLD = 0.75

    def __init__(
        self,
        llm
    ):

        self.llm = llm

        self.embedder = (
            SentenceTransformer(
                "all-MiniLM-L6-v2"
            )
        )

        self.prompt = PromptTemplate(
            input_variables=[
                "rewritten_clause",
                "violated_sections",
                "evidence"
            ],
            template=VALIDATOR_PROMPT
        )

    # ====================================
    # LLM VALIDATION
    # ====================================

    def llm_check(
        self,
        rewritten_clause,
        violated_sections,
        evidence
    ):

        prompt = self.prompt.format(
            rewritten_clause=
                rewritten_clause,

            violated_sections=
                "\n".join(
                    violated_sections
                ),

            evidence=
                "\n\n".join(
                    evidence
                )
        )

        response = llm_invoke_with_retry(self.llm, prompt)

        return response.content

    # ====================================
    # SIMILARITY
    # ====================================

    def similarity_check(
        self,
        original_clause,
        rewritten_clause
    ):

        emb1 = self.embedder.encode(
            original_clause
        )

        emb2 = self.embedder.encode(
            rewritten_clause
        )

        score = cosine_similarity(
            [emb1],
            [emb2]
        )[0][0]

        return float(score)

    # ====================================
    # MAIN ENTRY
    # ====================================

    def run(
        self,
        state
    ):

        raw = self.llm_check(
            rewritten_clause=
                state[
                    "rewritten_clause"
                ],

            violated_sections=
                state[
                    "violated_sections"
                ],

            evidence=
                state[
                    "evidence"
                ]
        )

        try:

            parsed = _load_json(raw)


            validated = ValidatorResult.model_validate(parsed)

        except (json.JSONDecodeError, ValidationError):

            raw = self.llm_check(
                rewritten_clause=state["rewritten_clause"],
                violated_sections=state["violated_sections"],
                evidence=state["evidence"]
            )

            try:
                parsed = _load_json(raw)
                validated = ValidatorResult.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(
                    "Validator agent response invalid after retry: "
                    f"{raw!r}"
                ) from exc

        similarity_score = (
            self.similarity_check(
                state["masked_text"],
                state["rewritten_clause"]
            )
        )

        llm_check_passed = parsed["validation_passed"]

        if similarity_score < self.SIMILARITY_THRESHOLD:

            if llm_check_passed:

                parsed["validation_passed"] = True

                parsed["validator_feedback"] = (
                    f"Warning: similarity {similarity_score:.3f} "
                    f"below threshold but LLM confirmed compliant"
                )

            else:

                parsed["validation_passed"] = False

                parsed["validator_feedback"] += (
                    f"\nSimilarity below "
                    f"threshold: "
                    f"{similarity_score:.3f}"
                )

        state[
            "validation_passed"
        ] = (
            parsed["validation_passed"]
        )

        state[
            "validator_feedback"
        ] = (
            parsed["validator_feedback"]
        )

        # ==========================
        # RETRY LOGIC
        # ==========================

        if not state[
            "validation_passed"
        ]:

            state[
                "failed_rewrites"
            ].append(
                state[
                    "rewritten_clause"
                ]
            )

            state[
                "rewrite_attempts"
            ] += 1

            if (
                state[
                    "rewrite_attempts"
                ] >= 2
            ):

                state[
                    "manual_review_flag"
                ] = True

        print(
            f"[VALIDATOR] "
            f"clause={state['clause_id']} "
            f"passed={state['validation_passed']} "
            f"attempts={state['rewrite_attempts']} "
            f"manual_review={state['manual_review_flag']}"
        )

        return state