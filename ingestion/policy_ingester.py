# ingestion/policy_ingester.py

import re
import uuid
import pickle
import pdfplumber

from pathlib import Path

from chromadb import PersistentClient

from sentence_transformers import SentenceTransformer

from rank_bm25 import BM25Okapi


class PolicyIngester:

    def __init__(
        self,
        chroma_path: str = "data/chroma"
    ):

        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.client = PersistentClient(
            path=chroma_path
        )

        self.collection = (
            self.client.get_or_create_collection(
                name="policy_chunks"
            )
        )

    # =====================================================
    # PDF LOADING
    # =====================================================

    def load_pdf(
        self,
        pdf_path: str
    ) -> str:

        pages = []

        with pdfplumber.open(pdf_path) as pdf:

            for page in pdf.pages:

                text = page.extract_text()

                if text:
                    pages.append(text)

        return "\n".join(pages)

    # =====================================================
    # HIERARCHICAL CHUNKING
    # =====================================================

    def hierarchical_chunk(
        self,
        text: str
    ) -> list[dict]:

        """
        Preserve section boundaries.

        No arbitrary token splitting.
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

            match = section_pattern.match(
                line.strip()
            )

            if match:

                if buffer:

                    chunks.append(
                        {
                            "chunk_id": str(uuid.uuid4()),
                            "section_number":
                                current_section,
                            "title":
                                current_title,
                            "text":
                                "\n".join(buffer)
                        }
                    )

                current_section = (
                    match.group(1)
                )

                current_title = (
                    match.group(3)
                )

                buffer = [line]

            else:
                buffer.append(line)

        if buffer:

            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "section_number":
                        current_section,
                    "title":
                        current_title,
                    "text":
                        "\n".join(buffer)
                }
            )

        return chunks

    # =====================================================
    # CATEGORY DETECTION
    # =====================================================

    def detect_category(
        self,
        title: str
    ) -> str:

        title = (title or "").lower()

        mapping = {
            "termination":
                "termination",

            "liability":
                "liability",

            "confidential":
                "confidentiality",

            "privacy":
                "privacy",

            "security":
                "security",

            "data":
                "data_protection",

            "payment":
                "payment",

            "indemn":
                "indemnification"
        }

        for key, value in mapping.items():

            if key in title:
                return value

        return "general"

    # =====================================================
    # CHROMA STORAGE
    # =====================================================

    def store_chunks(
        self,
        chunks: list[dict]
    ):

        documents = []
        ids = []
        metadatas = []

        for chunk in chunks:

            documents.append(
                chunk["text"]
            )

            ids.append(
                chunk["chunk_id"]
            )

            section_number = chunk.get(
                "section_number"
            ) or ""
            title = chunk.get(
                "title"
            ) or ""

            metadatas.append(
                {
                    "section_number":
                        section_number,

                    "title":
                        title,

                    "category":
                        self.detect_category(
                            title
                        )
                }
            )

        embeddings = (
            self.embedder.encode(
                documents,
                show_progress_bar=True
            ).tolist()
        )

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    # =====================================================
    # BM25 INDEX
    # =====================================================

    def build_bm25(
        self,
        chunks: list[dict],
        save_path="data/bm25.pkl"
    ):

        corpus = [
            chunk["text"]
            for chunk in chunks
        ]

        tokenized = [
            doc.split()
            for doc in corpus
        ]

        bm25 = BM25Okapi(
            tokenized
        )

        Path(save_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            save_path,
            "wb"
        ) as f:

            pickle.dump(
                {
                    "bm25": bm25,
                    "corpus": corpus
                },
                f
            )

        return bm25

    # =====================================================
    # MAIN INGESTION
    # =====================================================

    def ingest_policy(
        self,
        pdf_path: str,
        force: bool = False
    ):

        # Fix 3 — skip re-ingestion if DB already populated
        if self.collection.count() > 0 and not force:
            print(
                f"[INGEST] Skipping — "
                f"{self.collection.count()} chunks already in DB. "
                f"Pass force=True to re-ingest."
            )
            return []

        print(f"Ingesting policy: {pdf_path}")

        text = self.load_pdf(
            pdf_path
        )

        chunks = (
            self.hierarchical_chunk(
                text
            )
        )

        self.store_chunks(
            chunks
        )

        self.build_bm25(
            chunks
        )

        print(
            f"Stored {len(chunks)} chunks"
        )

        return chunks