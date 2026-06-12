# retrieval/parent_child.py

from chromadb import PersistentClient


class ParentChildRetriever:

    def __init__(
        self,
        collection
    ):
        self.collection = collection

    # ==========================================
    # GET PARENT SECTION
    # ==========================================

    def get_parent_section(
        self,
        section_number: str
    ):

        results = self.collection.get(
            where={
                "section_number":
                    section_number
            }
        )

        if not results["documents"]:
            return ""

        documents = results["documents"]

        parent_text = "\n\n".join(
            documents
        )

        return parent_text

    # ==========================================
    # COMBINE PARENT + CHILD
    # ==========================================

    def combine_context(
        self,
        retrieved_chunks: list
    ):

        combined_contexts = []

        for chunk in retrieved_chunks:

            metadata = chunk["metadata"]

            section_number = (
                metadata.get(
                    "section_number"
                )
            )

            if not section_number:
                parent_text = ""
            else:
                parent_text = (
                    self.get_parent_section(
                        section_number
                    )
                )

            combined_contexts.append(
                {
                    "chunk_id":
                        chunk["chunk_id"],

                    "section_number":
                        section_number,

                    "child_text":
                        chunk["text"],

                    "parent_text":
                        parent_text,

                    "combined_context":
                        f"""
PARENT SECTION:

{parent_text}

--------------------------------

CHILD CHUNK:

{chunk["text"]}
"""
                }
            )

        return combined_contexts