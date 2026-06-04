# retrieval/evidence_extractor.py
from langchain_classic.chains import LLMChain
from langchain_classic.prompts import PromptTemplate


EVIDENCE_EXTRACTION_PROMPT = """
You are a policy evidence extraction system.

Contract Clause:

{clause}

Policy Context:

{context}

Task:

Extract ONLY the sentences and paragraphs
from the policy context that are directly
relevant to assessing the clause.

Rules:

- Remove unrelated text
- Preserve exact wording
- Do not summarize
- Do not explain
- Do not analyze
- Return only extracted evidence

Relevant Evidence:
"""


class EvidenceExtractor:

    def __init__(
        self,
        llm
    ):

        self.chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=[
                    "clause",
                    "context"
                ],
                template=
                    EVIDENCE_EXTRACTION_PROMPT
            )
        )

    # =====================================
    # EXTRACT SINGLE EVIDENCE
    # =====================================

    def extract(
        self,
        clause_text: str,
        context: str
    ) -> str:

        result = self.chain.invoke(
            {"clause": clause_text, "context": context}
        )

        return result["text"].strip()

    # =====================================
    # EXTRACT MULTIPLE
    # =====================================

    def extract_all(
        self,
        clause_text: str,
        retrieved_contexts: list
    ) -> list[str]:

        evidence_list = []

        for item in retrieved_contexts:

            evidence = self.extract(
                clause_text=clause_text,
                context=item[
                    "combined_context"
                ]
            )

            if evidence:

                evidence_list.append(
                    evidence
                )

        return evidence_list