# ingestion/policy_ingester.py

import re
import uuid
import pdfplumber
from pathlib import Path

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def get_collection_name(policy_filepath: str) -> str:
    """
    Derives a safe ChromaDB collection name from the policy filename.
    company_policy.pdf  -> policy_company_policy
    hr_policy.pdf       -> policy_hr_policy
    Any new PDF         -> policy_<sanitised_stem>
    """
    stem = Path(policy_filepath).stem.lower()
    safe = re.sub(r'[^a-z0-9]+', '_', stem).strip('_')
    return f"policy_{safe}"

def load_pdf(pdf_path: str) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)

def hierarchical_chunk(text: str) -> list[dict]:
    """
    Preserve section boundaries. No arbitrary token splitting.
    """
    chunks = []
    current_section = None
    current_title = None
    buffer = []

    section_pattern = re.compile(
        r"^(\d+(\.\d+)*)\s+(.+)$",
        re.MULTILINE
    )

    lines = text.split("\n")
    for line in lines:
        match = section_pattern.match(line.strip())
        if match:
            if buffer:
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "section_number": current_section,
                    "title": current_title,
                    "text": "\n".join(buffer)
                })
            current_section = match.group(1)
            current_title = match.group(3)
            buffer = [line]
        else:
            buffer.append(line)

    if buffer:
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "section_number": current_section,
            "title": current_title,
            "text": "\n".join(buffer)
        })

    return chunks

def detect_category(title: str) -> str:
    title = (title or "").lower()
    mapping = {
        "termination": "termination",
        "liability": "liability",
        "confidential": "confidentiality",
        "privacy": "privacy",
        "security": "security",
        "data": "data_protection",
        "payment": "payment",
        "indemn": "indemnification"
    }
    for key, value in mapping.items():
        if key in title:
            return value
    return "general"

def store_chunks(collection, embed_model, chunks: list[dict]):
    documents = []
    ids = []
    metadatas = []

    for chunk in chunks:
        documents.append(chunk["text"])
        ids.append(chunk["chunk_id"])

        section_number = chunk.get("section_number") or ""
        title = chunk.get("title") or ""
        source = chunk.get("source") or ""
        source_file = chunk.get("source_file") or ""
        collection_name = chunk.get("collection") or ""

        metadatas.append({
            "section_number": section_number,
            "title": title,
            "category": detect_category(title),
            "source": source,
            "source_file": source_file,
            "collection": collection_name
        })

    embeddings = embed_model.encode(
        documents,
        show_progress_bar=True
    ).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

def ingest_policy(policy_filepath: str,
                  chroma_client,
                  embed_model,
                  force: bool = False) -> str:
    """
    Ingests one policy PDF into its own ChromaDB collection.
    Returns the collection name it was ingested into.
    """
    collection_name = get_collection_name(policy_filepath)
    collection = chroma_client.get_or_create_collection(collection_name)

    if collection.count() > 0 and not force:
        print(f"[INGEST] Skipping {policy_filepath} "
              f"- already in {collection_name} "
              f"({collection.count()} chunks)")
        return collection_name

    if force:
        try:
            chroma_client.delete_collection(collection_name)
        except Exception:
            pass
        collection = chroma_client.get_or_create_collection(collection_name)

    print(f"Ingesting policy: {policy_filepath}")
    text = load_pdf(policy_filepath)
    chunks = hierarchical_chunk(text)

    for chunk in chunks:
        chunk["source"] = Path(policy_filepath).name
        chunk["source_file"] = policy_filepath
        chunk["collection"] = collection_name

    store_chunks(collection, embed_model, chunks)

    print(f"[INGEST] {policy_filepath} -> {collection_name} ({len(chunks)} chunks)")
    return collection_name

def ingest_all_policies(policy_dir: str,
                        chroma_client,
                        embed_model,
                        force: bool = False) -> list[str]:
    """
    Ingests every PDF in policy_dir into its own collection.
    Returns list of collection names.
    """
    policy_dir = Path(policy_dir)
    pdfs = sorted(policy_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDF files found in {policy_dir}"
        )
    collections = []
    for pdf in pdfs:
        name = ingest_policy(
            str(pdf), chroma_client, embed_model, force
        )
        collections.append(name)
    return collections

# For backwards compatibility
class PolicyIngester:
    def __init__(self, chroma_path: str = "data/chroma"):
        from chromadb import PersistentClient
        from sentence_transformers import SentenceTransformer
        self.client = PersistentClient(path=chroma_path)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def ingest_policy(self, pdf_path: str, force: bool = False):
        return ingest_policy(pdf_path, self.client, self.embedder, force)