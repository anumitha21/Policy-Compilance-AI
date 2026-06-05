# retrieval/hybrid_retriever.py

import pickle

from chromadb import PersistentClient

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)


class HybridRetriever:

    def __init__(
        self,
        chroma_path: str = "data/chroma",
        bm25_path: str = "data/bm25.pkl"
    ):

        # ---------------------------------
        # Embedding Model
        # ---------------------------------

        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # ---------------------------------
        # Cross Encoder Reranker
        # ---------------------------------

        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        # ---------------------------------
        # Chroma
        # ---------------------------------

        client = PersistentClient(
            path=chroma_path
        )

        self.collection = (
            client.get_collection(
                "policy_chunks"
            )
        )

        # ---------------------------------
        # BM25
        # ---------------------------------

        with open(
            bm25_path,
            "rb"
        ) as f:

            bm25_data = pickle.load(f)

        self.bm25 = bm25_data["bm25"]
        self.corpus = bm25_data["corpus"]

    # ==================================================
    # SEMANTIC RETRIEVAL
    # ==================================================

    def semantic_search(
        self,
        query: str,
        top_k: int = 10
    ):

        query_embedding = (
            self.embedder.encode(
                query
            ).tolist()
        )

        results = self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=top_k
        )

        documents = []

        for i in range(
            len(results["ids"][0])
        ):

            documents.append(
                {
                    "chunk_id":
                        results["ids"][0][i],

                    "text":
                        results["documents"][0][i],

                    "metadata":
                        results["metadatas"][0][i]
                }
            )

        return documents

    # ==================================================
    # BM25 RETRIEVAL
    # ==================================================

    def bm25_search(
        self,
        query: str,
        top_k: int = 10
    ):

        tokenized_query = (
            query.split()
        )

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for idx in ranked_indices[:top_k]:

            results.append(
                {
                    "chunk_id":
                        f"bm25_{idx}",

                    "text":
                        self.corpus[idx],

                    "metadata":
                        {}
                }
            )

        return results

    # ==================================================
    # MERGE + DEDUP
    # ==================================================

    def merge_results(
        self,
        semantic_results,
        bm25_results
    ):

        merged = {}

        for doc in (
            semantic_results +
            bm25_results
        ):

            text = doc["text"]

            if text not in merged:

                merged[text] = doc

        return list(
            merged.values()
        )

    # ==================================================
    # RERANK
    # ==================================================

    def rerank(
        self,
        query: str,
        documents: list,
        top_n: int = 5
    ):

        pairs = []

        for doc in documents:

            pairs.append(
                [
                    query,
                    doc["text"]
                ]
            )

        scores = (
            self.reranker.predict(
                pairs
            )
        )

        ranked = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True
        )

        return [
            doc
            for score, doc
            in ranked[:top_n]
        ]

    # ==================================================
    # FULL PIPELINE
    # ==================================================

    def retrieve(
        self,
        query: str
    ):

        semantic_results = (
            self.semantic_search(
                query=query,
                top_k=15
            )
        )

        bm25_results = (
            self.bm25_search(
                query=query,
                top_k=15
            )
        )

        merged = (
            self.merge_results(
                semantic_results,
                bm25_results
            )
        )

        reranked = (
            self.rerank(
                query=query,
                documents=merged,
                top_n=10
            )
        )

        return reranked