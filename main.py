# Entry point placeholder following provided architecture
# main.py

import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from ingestion.policy_ingester import (
    PolicyIngester
)

from ingestion.contract_parser import (
    ContractParser
)

from preprocessing.pii_masker import (
    PIIMasker
)

from retrieval.hybrid_retriever import (
    HybridRetriever
)

from retrieval.parent_child import (
    ParentChildRetriever
)

from retrieval.evidence_extractor import (
    EvidenceExtractor
)

from agents.compliance_agent import (
    ComplianceAgent
)

from agents.risk_agent import (
    RiskAgent
)

from agents.rewrite_agent import (
    RewriteAgent
)

from agents.validator_agent import (
    ValidatorAgent
)

from graph.pipeline import (
    CompliancePipeline
)

from models.schemas import (
    ClauseReviewResult
)


# ==========================================
# ENV
# ==========================================

load_dotenv()

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY missing"
    )


# ==========================================
# LLM
# ==========================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=GROQ_API_KEY
    #model_kwargs={"response_format": {"type": "json_object"}}
)


# ==========================================
# INGESTION
# ==========================================

def ingest_policies(
    policy_paths: list[str]
):

    ingester = PolicyIngester()

    for path in policy_paths:

        ingester.ingest_policy(
            path
        )


# ==========================================
# BUILD GRAPH
# ==========================================

def build_graph():

    compliance_agent = (
        ComplianceAgent(llm)
    )

    risk_agent = (
        RiskAgent(llm)
    )

    rewrite_agent = (
        RewriteAgent(llm)
    )

    validator_agent = (
        ValidatorAgent(llm)
    )

    pipeline = (
        CompliancePipeline(
            compliance_agent,
            risk_agent,
            rewrite_agent,
            validator_agent
        )
    )

    return pipeline.build()


# ==========================================
# CLAUSE REVIEW
# ==========================================

def review_clause(
    clause,
    graph,
    retriever,
    parent_child,
    extractor
):

    semantic_context = (
        retriever.retrieve(
            clause["masked_text"]
        )
    )

    parent_context = (
        parent_child.combine_context(
            semantic_context
        )
    )

    evidence = (
        extractor.extract_all(
            clause["masked_text"],
            parent_context
        )
    )

    state = {

        "clause_id":
            clause["clause_id"],

        "clause_text":
            clause["raw_text"],

        "masked_text":
            clause["masked_text"],

        "evidence":
            evidence,

        "compliance_verdict": None,
        "compliance_explanation": None,
        "violated_sections": [],
        "citations": [],
        "confidence_score": 0.0,

        "risk_level": None,
        "risk_score": None,
        "risk_explanation": None,

        "rewritten_clause":None,

        "validation_passed":None,
        "validator_feedback":None,

        "rewrite_attempts":0,

        "manual_review_flag":False,

        "failed_rewrites":[]
    }

    final_state = graph.invoke(
        state
    )

    result = ClauseReviewResult(

        clause_id=
            final_state["clause_id"],

        compliance_verdict=
            final_state[
                "compliance_verdict"
            ],

        confidence_score=
            final_state[
                "confidence_score"
            ],

        risk_level=
            final_state.get(
                "risk_level"
            ),

        risk_score=
            final_state.get(
                "risk_score"
            ),

        risk_explanation=
            final_state.get(
                "risk_explanation"
            ),

        violated_sections=
            final_state[
                "violated_sections"
            ],

        citations=
            final_state[
                "citations"
            ],

        reasoning=
            final_state[
                "compliance_explanation"
            ],

        compliance_explanation=
            final_state[
                "compliance_explanation"
            ],
        suggested_rewrite=(
            None
            if final_state[
                "manual_review_flag"
            ]
            else final_state.get(
                "rewritten_clause"
            )
        ),

        manual_review_required=
            final_state[
                "manual_review_flag"
            ],

        manual_review_reason=
            final_state[
                "validator_feedback"
            ]
            if final_state[
                "manual_review_flag"
            ]
            else None,

        failed_rewrites=
            final_state[
                "failed_rewrites"
            ]
    )

    return result


# ==========================================
# CONTRACT REVIEW
# ==========================================

def review_contract(
    contract_pdf: str
):

    parser = ContractParser()

    masker = PIIMasker()

    retriever = HybridRetriever()

    parent_child = (
        ParentChildRetriever()
    )

    extractor = (
        EvidenceExtractor(llm)
    )

    graph = build_graph()

    clauses = parser.parse_contract(
        contract_pdf
    )

    clauses = masker.mask_clauses(
        clauses
    )

    results = []

    for clause in clauses:

        print(
            f"\nReviewing "
            f"{clause['clause_id']}"
        )

        result = review_clause(
            clause,
            graph,
            retriever,
            parent_child,
            extractor
        )

        results.append(
            result
        )

    return results


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":

    POLICY_FILES = [
        "policies/company_policy.pdf"
    ]

    CONTRACT_FILE = (
        "contracts/sample_contract.pdf"
    )

    ingest_policies(
        POLICY_FILES
    )

    results = review_contract(
        CONTRACT_FILE
    )

    for result in results:

        print(
            result.model_dump_json(
                indent=2
            )
        )