# AI Contract Compliance Review System

An agentic, production-oriented pipeline that automatically reviews contract clauses against company policy documents — returning compliance verdicts, risk scores, policy citations, and suggested rewrites.

Built with **FastAPI**, **LangChain**, **LangGraph**, **Groq (LLaMA 3.3-70B)**, **ChromaDB**, and **Presidio**.

---

## What It Does

Upload a contract PDF through the web UI. The system reviews every clause against your ingested policy documents and returns — for each clause:

- ✅ / ❌ Compliance verdict with explanation
- 📋 Exact policy sections violated, with citations and excerpts
- 🔢 Confidence score (0.0 – 1.0)
- ⚠️ Risk level (Critical / Major / Moderate / Minor) with risk score (1–10)
- ✏️ Suggested compliant rewrite (recommendation only — not an auto-edit)
- 🚩 Manual review flag if a valid rewrite cannot be produced after two attempts

---

## Architecture Overview

```
Policy PDFs ──► Hierarchical Chunking ──► ChromaDB + BM25 Index
                                                    │
Contract PDF ──► Clause Splitting ──► PII Masking   │
                                          │         │
                                    Hybrid Retrieval (Semantic + BM25)
                                          │
                                    Cross-Encoder Reranking (Top-5)
                                          │
                                    Parent-Child Retrieval
                                          │
                                    Evidence Extraction
                                          │
                              ┌──────────────────────────────┐
                              │       LangGraph Pipeline      │
                              │                              │
                              │  [Compliance + Risk Agent]   │  ← single LLM call
                              │         │                    │
                              │    compliant? ──► END        │
                              │         │                    │
                              │  [Rewrite Agent]  ◄──┐       │
                              │         │            │       │
                              │  [Validator Agent]   │       │
                              │    pass? ──No──► retry (max 2)
                              │    fail after 2? ──► Manual Review Flag
                              └──────────────────────────────┘
                                          │
                              FastAPI  ──► JSON Response
                                          │
                              static/index.html  (browser UI)
```

---

## Pipeline — Step by Step

### 1. Policy Ingestion *(upload once via UI or API)*
- Parses policy PDFs with `pdfplumber`
- Chunks hierarchically — section and subsection boundaries are preserved
- Tags each chunk with metadata: `section_number`, `title`, `category`
- Embeds chunks with `all-MiniLM-L6-v2` → stored in **ChromaDB** (persistent)
- Builds a **BM25 index** (`rank_bm25`) over all chunks for keyword search
- Skips re-ingestion automatically if ChromaDB is already populated — pass `force=True` to override

### 2. Contract Parsing
- Parses contract PDF with `pdfplumber`
- Splits into individual clauses using regex patterns (`Clause 1`, `1.`, `(1)`)
- Preamble / header text is filtered out before review
- Each clause carries: `clause_id`, `text`, `raw_text`

### 3. PII Masking
- Runs `presidio-analyzer` to detect: names, emails, phone numbers, SSNs, card numbers, IBANs
- Runs `presidio-anonymizer` to replace entities with `<ENTITY_TYPE>` placeholders
- Only the masked text is sent to retrieval and agents — raw text is preserved for rewrites

### 4. Hybrid Retrieval
- Embeds masked clause text → queries ChromaDB for **top-10 semantic results**
- Tokenizes masked clause text → queries BM25 index for **top-10 keyword results**
- Merges and deduplicates results by `chunk_id`
- Runs `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker → selects **top-5 chunks**

### 5. Parent-Child Retrieval
- For each of the top-5 chunks, retrieves the **parent section** from ChromaDB using `section_number`
- Passes both the child chunk and parent section to the next step

### 6. Evidence Extraction
- Groq LLM reads the full parent+child context and extracts **only the sentences directly relevant** to the clause
- Reduces noise and token consumption before passing to agents
- Uses LCEL chain (`prompt | llm`) — no deprecated `LLMChain`

---

## LangGraph Agents

### Compliance + Risk Agent *(merged — single LLM call)*
Compares the clause against extracted evidence and assesses risk in one call.

Returns: `compliance_verdict`, `compliance_explanation`, `violated_sections`, `citations`, `confidence_score`, `risk_level`, `risk_score`, `risk_explanation`

Compliant clauses exit the graph immediately — no downstream calls.

Risk bands (only populated for non-compliant clauses):

| Level    | Score | Description |
|----------|-------|-------------|
| Critical | 9–10  | Removes mandatory protections / severe legal or regulatory exposure |
| Major    | 6–8   | Significant obligation, liability, or procedural violation |
| Moderate | 3–5   | Partial non-compliance, limited exposure |
| Minor    | 1–2   | Technical or administrative violation |

> Procedural violations (skipping mediation, missing notice period) = **Major**, not Critical.

### Rewrite Agent *(runs only if non-compliant)*
Generates a corrected clause following four hard rules:
1. Preserve the original business intent
2. Modify only the violating language
3. Do not introduce new obligations
4. Align with the cited policy sections

Policy-specific rules are baked into the prompt (e.g. 14-day cure notice for termination, AAA arbitration for dispute resolution).

### Validator Agent *(runs only if non-compliant)*
Runs two checks on every rewrite:
- **LLM re-check** — does the rewrite satisfy the violated policy sections?
- **Semantic similarity check** — cosine similarity between rewrite and original clause must be ≥ 0.75

Similarity threshold is a soft guard: if the LLM confirms compliance, a low similarity score is logged as a warning rather than a hard failure.

If validation fails:
- Retries up to **2 attempts** with validator feedback passed back to the Rewrite Agent
- After 2 failed attempts → sets `manual_review_required = True` and stops

---

## Web Interface

The UI is a single-page HTML application served by FastAPI from `static/index.html`.

**Pages:**
- **Dashboard** — metric cards, risk distribution chart, violations by policy area chart, clause results table, click-to-inspect detail panel
- **Review contract** — upload policy PDFs + contract PDF, animated progress bar, auto-navigates to Dashboard on completion
- **Policy library** — lists all ingested policy PDFs with file size and ingestion date
- **Manual review queue** — shows all clauses flagged for human review with escalation reason

**Features:**
- Clause detail slide-in panel with assessment, violated sections, citations, suggested rewrite, confidence bar
- Export full results as JSON from the browser
- Policy PDFs are auto-saved to `policies/` and auto-ingested on upload

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the HTML frontend |
| `GET` | `/api/policies` | Lists ingested policy PDFs and chunk count |
| `POST` | `/api/policies/ingest` | Uploads and ingests one or more policy PDFs |
| `POST` | `/api/review` | Runs the full pipeline on an uploaded contract PDF |
| `GET` | `/api/health` | Health check |

---

## Output

Each clause produces a `ClauseReviewResult`:

```json
{
  "clause_id": "CLAUSE_3",
  "compliance_verdict": "non_compliant",
  "confidence_score": 0.99,
  "risk_level": "major",
  "risk_score": 7,
  "risk_explanation": "Clause removes the mandatory 30-day notice period required under Section 1.1.",
  "violated_sections": ["1.1 Notice Period", "1.3 Termination for Cause"],
  "citations": [
    {
      "section": "1.1 Notice Period",
      "excerpt": "Either party must provide a minimum of thirty (30) days written notice prior to termination."
    }
  ],
  "compliance_explanation": "The clause allows immediate termination without any notice period, directly violating Section 1.1.",
  "suggested_rewrite": "Either party may terminate this Agreement upon providing thirty (30) days written notice...",
  "manual_review_required": false,
  "manual_review_reason": null,
  "failed_rewrites": []
}
```

> `suggested_rewrite` is a recommendation only. No contract is modified automatically.

---

## Project Structure

```
Policy_Compilance/
├── agents/
│   ├── compliance_agent.py      # Merged compliance + risk — single LLM call
│   ├── rewrite_agent.py
│   └── validator_agent.py
├── ingestion/
│   ├── policy_ingester.py       # Hierarchical chunking, ChromaDB, BM25
│   └── contract_parser.py       # PDF parsing, clause splitting, preamble filtering
├── preprocessing/
│   └── pii_masker.py            # Presidio PII detection and anonymization
├── retrieval/
│   ├── hybrid_retriever.py      # Semantic + BM25 retrieval and reranking
│   ├── parent_child.py          # Parent section expansion
│   └── evidence_extractor.py   # LCEL evidence pruning
├── graph/
│   └── pipeline.py              # LangGraph StateGraph — 3 nodes, conditional routing
├── models/
│   └── schemas.py               # Pydantic v2 models + ComplianceState TypedDict
├── static/
│   └── index.html               # Single-page HTML frontend
├── data/
│   ├── chroma/                  # Persistent ChromaDB vector store
│   └── bm25.pkl                 # Serialized BM25 index
├── policies/                    # Ingested policy PDFs (auto-populated via UI)
├── contracts/                   # Sample contracts
├── server.py                    # FastAPI backend
├── main.py                      # Pipeline logic (imported by server.py)
├── .env                         # GROQ_API_KEY
└── requirements.txt
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/contract-compliance.git
cd contract-compliance
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 3. Configure environment
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Start the server
```bash
uvicorn server:app --reload --port 8000
```

### 5. Open the UI
```
http://localhost:8000
```

Upload policy PDFs and a contract PDF directly from the browser. Policy ingestion and contract review run entirely through the web interface.

---

## Requirements

```
langchain
langchain-groq
langchain-core
langgraph
langchain-community
langchain-huggingface
chromadb
pdfplumber
rank_bm25
sentence-transformers
presidio-analyzer
presidio-anonymizer
pydantic>=2.0
python-dotenv
scikit-learn
fastapi
uvicorn[standard]
python-multipart
```

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| Merged Compliance + Risk agent | Saves 1 LLM call per clause — both verdicts share the same context |
| Hybrid retrieval (BM25 + semantic) | Legal text requires both exact term matching and semantic understanding |
| Hierarchical chunking | Preserves document structure; avoids breaking mid-section |
| PII masking before retrieval | Sensitive data never reaches the vector DB or LLM |
| Predefined risk bands | Prevents LLM from hallucinating arbitrary risk scores |
| Procedural vs substantive risk distinction | Skipping a process step ≠ removing a right — avoids over-scoring |
| ChromaDB skip-on-populated | Eliminates embedding overhead on every run — force=True to re-ingest |
| Parallel clause processing (4 workers) | Cuts wall time by ~65% on 10-clause contracts |
| Max 2 rewrite retries | Prevents infinite loops; unresolvable clauses escalated to humans |
| Similarity threshold as soft guard | LLM compliance confirmation overrides low similarity — avoids false blocks |
| LCEL chains over LLMChain | Removes deprecated LangChain overhead |
| FastAPI + static HTML | No framework overhead; UI loads instantly; API is independently testable |
| Rewrites as recommendations | Legal changes require human sign-off — system assists, does not decide |

---

## Limitations

- Clause splitting relies on numbering patterns — contracts with non-standard formatting may need a custom parser
- Policy version management is not included — all reviews use the currently ingested policy
- Confidence scores from the LLM are indicative, not statistically calibrated
- Parallel clause processing shares one LangGraph instance — thread safety depends on LangGraph's internal state isolation

---

## License

MIT
