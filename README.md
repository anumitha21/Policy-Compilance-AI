# AI Contract Compliance Review System

An agentic, production-oriented pipeline that automatically reviews contract clauses against company policy documents — returning compliance verdicts, risk scores, policy citations, and suggested rewrites.

Built with **LangChain**, **LangGraph**, **Groq (LLaMA 3.3-70B)**, **ChromaDB**, and **Guardrails AI**.

---

## What It Does

Upload a contract PDF. The system reviews every clause against your company's policy documents and returns — for each clause:

- ✅ / ❌ Compliance verdict with explanation
- 📋 Exact policy sections violated, with citations and excerpts
- 🔢 Confidence score (0.0 – 1.0)
- ⚠️ Risk level (Critical / Major / Moderate / Minor) with a risk score (1–10)
- ✏️ A suggested compliant rewrite (presented as a recommendation, not an auto-edit)
- 🚩 Manual review flag if the system cannot produce a valid rewrite after two attempts

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
                              ┌─────────────────────────┐
                              │      LangGraph Pipeline  │
                              │                         │
                              │  [Compliance Agent]      │
                              │         │               │
                              │    (non-compliant?)      │
                              │         │               │
                              │  [Risk Agent]            │
                              │         │               │
                              │  [Rewrite Agent] ◄──┐   │
                              │         │           │   │
                              │  [Validator Agent]  │   │
                              │    pass? ──No──►retry(max 2)
                              │    fail after 2? ──► Manual Review Flag
                              └─────────────────────────┘
                                          │
                              Structured JSON Output per Clause
```

---

## Pipeline — Step by Step

### 1. Policy Ingestion *(offline, run once)*
- Parses policy PDFs with `pdfplumber`
- Chunks hierarchically — section and subsection boundaries are preserved
- Tags each chunk with metadata: `section_number`, `title`, `category`
- Embeds chunks with `all-MiniLM-L6-v2` → stored in **ChromaDB**
- Builds a **BM25 index** (`rank_bm25`) over all chunks for keyword search

### 2. Contract Parsing
- Parses contract PDF with `pdfplumber`
- Splits into individual clauses using regex patterns (`Clause 1`, `1.`, `(1)`)
- Each clause carries: `clause_id`, `text`, `raw_text`

### 3. PII Masking
- Runs `presidio-analyzer` to detect: names, emails, phone numbers, SSNs, card numbers, IBANs
- Runs `presidio-anonymizer` to replace entities with `<ENTITY_TYPE>` placeholders
- Only the masked text is sent to retrieval and agents — raw text is preserved for the rewrite

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

---

## LangGraph Agents

### Compliance Agent
Compares the clause against extracted evidence.
Returns: `verdict`, `explanation`, `violated_sections`, `citations [{section, excerpt}]`, `confidence_score`

### Risk Agent
*(runs only if non-compliant)*
Maps findings to a predefined risk band — LLM selects the band, does not invent scores:

| Level    | Score | Description |
|----------|-------|-------------|
| Critical | 9–10  | Removes mandatory protections / major legal or regulatory exposure |
| Major    | 6–8   | Significant obligation or liability issue |
| Moderate | 3–5   | Partial non-compliance, limited exposure |
| Minor    | 1–2   | Technical or administrative violation |

### Rewrite Agent
*(runs only if non-compliant)*
Generates a corrected clause following four hard rules:
1. Preserve the original business intent
2. Modify only the violating language
3. Do not introduce new obligations
4. Align with the cited policy sections

### Validator Agent
Runs two checks on the rewrite:
- **LLM re-check** — does the rewrite satisfy the violated policy sections?
- **Semantic similarity check** — cosine similarity between rewrite and original clause must be ≥ 0.75

If either check fails:
- Retries up to **2 attempts** (feeds validator feedback back to Rewrite Agent)
- After 2 failed attempts → sets `manual_review_required = True` and stops

---

## Guardrails

Every agent output is validated with **Guardrails AI** before the pipeline proceeds:

| Field | Constraint |
|-------|-----------|
| `compliance_verdict` | Enum: `"compliant"` or `"non_compliant"` |
| `confidence_score` | Float: 0.0 – 1.0 |
| `risk_score` | Integer: 1 – 10 |
| `risk_level` | Enum: `critical`, `major`, `moderate`, `minor` |
| `rewritten_clause` | Non-empty string |
| All required fields | Present in every response |

If validation fails, the agent is re-prompted once before raising an error.

---

## Output

Each clause produces a `ClauseReviewResult`:

```json
{
  "clause_id": "clause_3",
  "compliance_verdict": "non_compliant",
  "confidence_score": 0.91,
  "risk_level": "major",
  "risk_score": 7,
  "risk_explanation": "Clause removes the mandatory 30-day notice period required under Section 4.2.",
  "violated_sections": ["Section 4.2 — Termination Rights"],
  "citations": [
    {
      "section": "Section 4.2",
      "excerpt": "Either party must provide a minimum of 30 days written notice prior to termination."
    }
  ],
  "reasoning": "The clause allows immediate termination without notice, directly violating Section 4.2.",
  "suggested_rewrite": "Either party may terminate this agreement upon providing thirty (30) days written notice to the other party.",
  "manual_review_required": false,
  "manual_review_reason": null,
  "failed_rewrites": []
}
```

> **Note:** `suggested_rewrite` is a recommendation only. No contract is modified automatically. A legal or compliance professional must review and approve any changes.

---

## Project Structure

```
contract_compliance/
├── ingestion/
│   ├── policy_ingester.py       # Parse, chunk, embed, and index policy PDFs
│   └── contract_parser.py       # Parse contract PDF and split into clauses
├── preprocessing/
│   └── pii_masker.py            # Presidio PII detection and anonymization
├── retrieval/
│   ├── hybrid_retriever.py      # Semantic + BM25 retrieval and reranking
│   ├── parent_child.py          # Parent section expansion
│   └── evidence_extractor.py   # LLM-based evidence pruning
├── agents/
│   ├── compliance_agent.py
│   ├── risk_agent.py
│   ├── rewrite_agent.py
│   └── validator_agent.py
├── graph/
│   └── pipeline.py              # LangGraph StateGraph — nodes and edges
├── models/
│   └── schemas.py               # ComplianceState, ClauseReviewResult (Pydantic v2)
├── guardrails/
│   └── rails.py                 # Guardrails AI specs per agent
├── main.py                      # Entry point
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
python -m spacy download en_core_web_lg   # required by presidio-analyzer
```

### 3. Configure environment
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Ingest policy documents
Place your policy PDFs in a `policies/` folder, then run:
```bash
python main.py --ingest --policy-dir policies/
```
This only needs to be run once (or whenever policies are updated).

### 5. Review a contract
```bash
python main.py --review --contract path/to/contract.pdf
```

---

## Requirements

```
langchain
langchain-groq
langgraph
langchain-community
langchain-huggingface
chromadb
pdfplumber
rank_bm25
sentence-transformers
presidio-analyzer
presidio-anonymizer
guardrails-ai
pydantic>=2.0
python-dotenv
```

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| Hybrid retrieval (BM25 + semantic) | Legal text requires both exact term matching and semantic understanding |
| Hierarchical chunking | Preserves document structure; avoids breaking mid-clause |
| PII masking before retrieval | Sensitive data never reaches the vector DB or LLM |
| Predefined risk bands | Prevents LLM from hallucinating arbitrary risk scores |
| Max 2 rewrite retries | Prevents infinite loops; unresolvable clauses are escalated to humans |
| Semantic similarity threshold (0.75) | Ensures rewrites don't change the clause's original business meaning |
| Guardrails AI on every agent | Catches malformed outputs before they propagate through the pipeline |
| Rewrites as recommendations | Legal changes require human sign-off — system assists, does not decide |

---

## Limitations

- Clause splitting relies on numbering patterns — contracts with non-standard formatting may require a custom parser
- Policy version management is not included in this version — all reviews use the current ingested policy
- Confidence scores from the LLM are indicative, not statistically calibrated

---

## License

MIT