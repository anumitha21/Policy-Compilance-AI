# retrieval/evidence_extractor.py

from langchain_core.prompts import PromptTemplate


EVIDENCE_PROMPT = """You are a legal evidence extraction assistant.

Contract Clause:
{clause}

Policy Section:
{context}

Extract only the sentences and paragraphs from the policy section
directly relevant to assessing this clause.
Remove all unrelated text.
Return only the extracted evidence, nothing else.
"""


class EvidenceExtractor:

    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["clause", "context"],
            template=EVIDENCE_PROMPT
        )
        self.chain = self.prompt | self.llm  # LCEL — no LLMChain

    def extract(self, clause_text: str, chunks: list[dict]) -> list[str]:
        """
        One LLM call per chunk. Returns a list of extracted evidence strings.
        Each string maps to one policy chunk — citations stay attributable.
        """
        from utils.retry import llm_invoke_with_retry

        evidence = []
        for chunk in chunks:
            context = chunk.get("combined_context") or chunk.get("text", "")
            words   = context.split()
            if len(words) > 300:
                context = " ".join(words[:300])

            try:
                result = llm_invoke_with_retry(
                    self.llm,
                    self.prompt.format(clause=clause_text, context=context)
                ).content.strip()

                if result:
                    evidence.append(result)

            except Exception as e:
                print(f"[EVIDENCE] extraction failed for chunk, using raw: {e}")
                section_label = chunk.get("title") or chunk.get("section_number") or "Policy Section"
                evidence.append(f"[{section_label}]\n{context}")

        return evidence

    # keep extract_all as alias so existing call sites don't break
    def extract_all(self, clause_text: str, retrieved_contexts: list) -> list[str]:
        return self.extract(clause_text, retrieved_contexts)
