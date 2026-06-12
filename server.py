# server.py

import os
import uuid
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

load_dotenv()

# ── Pipeline imports ───────────────────────────────────────────────
from main import (
    build_graph,
    review_clause,
    llm,
    chroma_client,
    embed_model,
    reranker,
    list_policy_collections,
    detect_policy_collection,
    build_bm25_corpus
)
from ingestion.policy_ingester import ingest_policy
from ingestion.contract_parser import ContractParser
from preprocessing.pii_masker import PIIMasker
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.parent_child import ParentChildRetriever
from retrieval.evidence_extractor import EvidenceExtractor
from chromadb import PersistentClient

# ── Directories ────────────────────────────────────────────────────
POLICY_DIR = Path("policies")
POLICY_DIR.mkdir(exist_ok=True)

# ── App ────────────────────────────────────────────────────────────
app = FastAPI(title="ComplianceAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (index.html + assets)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ══════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


# ── Policy library ─────────────────────────────────────────────────

@app.get("/api/policies")
def list_policies():
    """Return all ingested policy PDFs with metadata."""
    try:
        cols = list_policy_collections(chroma_client)
        chunk_count = 0
        for name in cols:
            try:
                col = chroma_client.get_collection(name)
                chunk_count += col.count()
            except Exception:
                pass
    except Exception:
        chunk_count = 0

    pdfs = sorted(POLICY_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {
        "policies": [
            {
                "name":       p.name,
                "size_kb":    round(p.stat().st_size / 1024, 1),
                "ingested_at": p.stat().st_mtime,
            }
            for p in pdfs
        ],
        "chunk_count": chunk_count,
    }


@app.post("/api/policies/ingest")
async def ingest_policies(files: list[UploadFile] = File(...)):
    """Upload and ingest one or more policy PDFs."""
    saved = []

    for f in files:
        dest = POLICY_DIR / f.filename
        content = await f.read()
        dest.write_bytes(content)
        ingest_policy(str(dest), chroma_client=chroma_client, embed_model=embed_model, force=True)
        saved.append(f.filename)

    return {"ingested": saved, "count": len(saved)}


# ── Contract review ────────────────────────────────────────────────

@app.post("/api/review")
async def review_contract(contract: UploadFile = File(...)):
    """
    Run the full compliance pipeline on an uploaded contract PDF.
    Returns a list of ClauseReviewResult dicts.
    """
    # Check policies exist
    if not list(POLICY_DIR.glob("*.pdf")):
        raise HTTPException(
            status_code=400,
            detail="No policy documents ingested. Upload policies first."
        )

    # Save contract to temp file
    tmp_path = Path(tempfile.mktemp(suffix=".pdf"))
    try:
        content = await contract.read()
        tmp_path.write_bytes(content)

        parser       = ContractParser()
        masker       = PIIMasker()

        clauses = parser.parse_contract(str(tmp_path))
        clauses = masker.mask_clauses(clauses)

        # Detect the correct policy collection for this contract
        collection_names = list_policy_collections(chroma_client)
        selected = detect_policy_collection(clauses, collection_names, llm)
        collection = chroma_client.get_collection(selected)

        # Scope retriever and parent child retriever instances to this collection
        parent_child = ParentChildRetriever(collection=collection)
        retriever = HybridRetriever(
            collection=collection,
            bm25_corpus=build_bm25_corpus(collection),
            reranker=reranker,
            llm=llm
        )
        extractor    = EvidenceExtractor(llm)
        graph        = build_graph()

        results = [None] * len(clauses)

        def _run(idx_clause):
            idx, clause = idx_clause
            return idx, review_clause(clause, graph, retriever, parent_child, extractor)

        # 1 worker — sequential, avoids burst on 12k TPM free tier
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = {
                executor.submit(_run, (i, c)): i
                for i, c in enumerate(clauses)
            }
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        # Serialize — enrich with policy_area + title fields for the frontend
        output = []
        for r in results:
            d = r.model_dump()
            vsecs = d.get("violated_sections") or []
            d["policy_area"] = vsecs[0] if vsecs else "—"
            d["title"]       = vsecs[0].split("—")[0].strip() if vsecs else d["clause_id"]
            # Normalise citations to plain dicts
            d["citations"] = [
                {"section": c.get("section","") if isinstance(c, dict) else c.section,
                 "excerpt":  c.get("excerpt","")  if isinstance(c, dict) else c.excerpt}
                for c in (d.get("citations") or [])
            ]
            output.append(d)

        return JSONResponse(content={"filename": contract.filename, "results": output})

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ── Health ─────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}
