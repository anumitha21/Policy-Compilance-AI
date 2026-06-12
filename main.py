# Entry point placeholder following provided architecture
# main.py

import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from ingestion.policy_ingester import ingest_all_policies, get_collection_name, ingest_policy
from ingestion.contract_parser import ContractParser
from preprocessing.pii_masker import PIIMasker
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.parent_child import ParentChildRetriever
from retrieval.evidence_extractor import EvidenceExtractor
from agents.compliance_agent import ComplianceAgent
from agents.rewrite_agent import RewriteAgent
from agents.validator_agent import ValidatorAgent
from graph.pipeline import CompliancePipeline
from models.schemas import ClauseReviewResult
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


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
# GLOBALS & LLM
# ==========================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=GROQ_API_KEY,
)

chroma_client = PersistentClient(path="data/chroma")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ==========================================
# INGESTION
# ==========================================

def ingest_policies(
    policy_paths: list[str],
    force: bool = False
):
    for path in policy_paths:
        ingest_policy(
            path,
            chroma_client=chroma_client,
            embed_model=embed_model,
            force=force
        )


# ==========================================
# BUILD GRAPH
# ==========================================

def build_graph():

    compliance_agent = ComplianceAgent(llm)
    rewrite_agent    = RewriteAgent(llm)
    validator_agent  = ValidatorAgent(llm)

    pipeline = CompliancePipeline(
        compliance_agent,
        rewrite_agent,
        validator_agent
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
# POLICY SELECTION & BM25 UTILS
# ==========================================

def list_policy_collections(chroma_client) -> list[str]:
    """
    Returns all policy collection names currently in ChromaDB.
    Filters to only collections starting with 'policy_' but not 'policy_chunks'.
    """
    return sorted([
        c.name
        for c in chroma_client.list_collections()
        if c.name.startswith("policy_") and c.name != "policy_chunks"
    ])


def detect_policy_collection(clauses: list[dict],
                              collection_names: list[str],
                              llm) -> str:
    """
    Uses the LLM to select the most relevant policy collection
    for this contract. Returns the collection name.
    Falls back to the first available collection on any error.
    """
    if not collection_names:
        raise ValueError("No policy collections found in ChromaDB.")

    if len(collection_names) == 1:
        print(f"[PIPELINE] Only one policy available: "
              f"{collection_names[0]}")
        return collection_names[0]

    sample = "\n\n".join(
        c["text"][:200] for c in clauses[:4]
    )
    options = "\n".join(f"- {n}" for n in collection_names)

    prompt = (
        "You are given sample clauses from a contract and a list "
        "of available policy collections stored in a database.\n\n"
        f"Contract sample:\n{sample}\n\n"
        f"Available policy collections:\n{options}\n\n"
        "Which single collection name is most relevant for "
        "reviewing this contract?\n"
        "Reply with ONLY the exact collection name from the list. "
        "No explanation."
    )

    try:
        result = llm.invoke(prompt).content.strip()
        if result in collection_names:
            print(f"[PIPELINE] Selected policy collection: {result}")
            return result
        print(f"[PIPELINE] LLM returned '{result}' - "
              f"not recognised, using {collection_names[0]}")
        return collection_names[0]
    except Exception as e:
        print(f"[PIPELINE] Collection detection failed: {e} "
              f"- using {collection_names[0]}")
        return collection_names[0]


def build_bm25_corpus(collection) -> BM25Okapi:
    """
    Builds a fresh BM25 index from the selected collection only.
    Ensures BM25 scoring is never influenced by other policies.
    """
    data = collection.get()
    corpus = [doc.lower().split() for doc in data.get("documents", [])]
    return BM25Okapi(corpus)


# ==========================================
# CONTRACT REVIEW
# ==========================================

def review_contract(
    contract_pdf: str
):

    parser       = ContractParser()
    masker       = PIIMasker()
    graph        = build_graph()

    clauses = parser.parse_contract(contract_pdf)
    clauses = masker.mask_clauses(clauses)

    # Select correct policy collection — fresh every run
    collection_names = list_policy_collections(chroma_client)
    selected = detect_policy_collection(clauses, collection_names, llm)
    collection = chroma_client.get_collection(selected)

    # Scope parent child retriever to the selected collection
    parent_child = ParentChildRetriever(collection=collection)
    extractor    = EvidenceExtractor(llm)

    # Build retriever scoped to this collection only
    retriever = HybridRetriever(
        collection=collection,
        bm25_corpus=build_bm25_corpus(collection),
        reranker=reranker,
        llm=llm
    )

    results  = [None] * len(clauses)
    MAX_WORKERS = 1  # sequential — avoids burst on 12k TPM free tier

    def _run(idx_clause):
        idx, clause = idx_clause
        print(f"\nReviewing {clause['clause_id']}")
        print(f"[RETRIEVAL] Using collection: {selected}")
        return idx, review_clause(clause, graph, retriever, parent_child, extractor)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_run, (i, c)): i
            for i, c in enumerate(clauses)
        }
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return results


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":

    force_ingest = "--force-ingest" in sys.argv

    # Ingest all PDFs in policies/ into separate collections
    ingest_all_policies(
        policy_dir="policies",
        chroma_client=chroma_client,
        embed_model=embed_model,
        force=force_ingest
    )

    # Detect contract file from --review arg
    if "--review" in sys.argv:
        try:
            idx = sys.argv.index("--review")
            CONTRACT_FILE = sys.argv[idx + 1]
        except (ValueError, IndexError):
            CONTRACT_FILE = "contracts/sample_contract.pdf"
    else:
        CONTRACT_FILE = "contracts/sample_contract.pdf"

    results = review_contract(CONTRACT_FILE)

    for result in results:
        print(result.model_dump_json(indent=2))