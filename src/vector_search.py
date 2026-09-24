import os
import json
import re

import faiss
import numpy as np

from rank_bm25 import BM25Okapi


# ============================================================
# HYBRID SEARCH STORE
# ============================================================

class VectorStore:

    def __init__(self):

        # ----------------------------------------------------
        # Vector search
        # ----------------------------------------------------

        self.index = None

        # ----------------------------------------------------
        # BM25 search
        # ----------------------------------------------------

        self.bm25 = None

        self.bm25_corpus = []

        # ----------------------------------------------------
        # Original chunks
        # ----------------------------------------------------

        self.chunks = []

    # ========================================================
    # TOKENIZE FOR BM25
    # ========================================================

    def tokenize(self, text):

        """
        Simple tokenizer for BM25.

        We:
        - lowercase
        - keep words/numbers
        - remove punctuation
        """

        if not text:
            return []

        return re.findall(
            r"\b\w+\b",
            text.lower()
        )

    # ========================================================
    # BUILD INDEX
    # ========================================================

    def build(
        self,
        chunks,
        embedding_model
    ):

        print(
            "\n========== BUILDING HYBRID SEARCH STORE =========="
        )

        if not chunks:

            raise ValueError(
                "No chunks available."
            )

        self.chunks = chunks

        # ====================================================
        # TEXT FOR BOTH SEARCH METHODS
        # ====================================================

        texts = [
            chunk.get("content", "")
            for chunk in chunks
        ]

        # ====================================================
        # 1. VECTOR INDEX
        # ====================================================

        print(
            "\nBuilding vector index..."
        )

        embeddings = (
            embedding_model.encode(
                texts
            )
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            embeddings.astype(
                np.float32
            )
        )

        print(
            f"Vector index: "
            f"{len(chunks)} chunks"
        )

        # ====================================================
        # 2. BM25 INDEX
        # ====================================================

        print(
            "Building BM25 index..."
        )

        self.bm25_corpus = [
            self.tokenize(text)
            for text in texts
        ]

        self.bm25 = BM25Okapi(
            self.bm25_corpus
        )

        print(
            f"BM25 index: "
            f"{len(self.bm25_corpus)} chunks"
        )

        print(
            "\nHybrid search store ready."
        )

    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    def vector_search(
        self,
        query,
        embedding_model,
        top_k=5
    ):

        if self.index is None:

            raise RuntimeError(
                "Vector index has not been built."
            )

        query_embedding = (
            embedding_model.encode(
                [query]
            )
        )

        scores, indices = (
            self.index.search(
                query_embedding.astype(
                    np.float32
                ),
                top_k
            )
        )

        results = []

        for rank, (score, index) in enumerate(
            zip(
                scores[0],
                indices[0]
            ),
            start=1
        ):

            if index == -1:
                continue

            chunk = self.chunks[index]

            results.append({

                "rank": rank,

                "score": float(score),

                "chunk": chunk,
            })

        return results

    # ========================================================
    # BM25 SEARCH
    # ========================================================

    def bm25_search(
        self,
        query,
        top_k=5
    ):

        if self.bm25 is None:

            raise RuntimeError(
                "BM25 index has not been built."
            )

        # ----------------------------------------------------
        # Tokenize query
        # ----------------------------------------------------

        query_tokens = self.tokenize(
            query
        )

        if not query_tokens:

            return []

        # ----------------------------------------------------
        # BM25 scores
        # ----------------------------------------------------

        scores = self.bm25.get_scores(
            query_tokens
        )

        # ----------------------------------------------------
        # Get top K indices
        # ----------------------------------------------------

        top_indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            score = scores[index]

            chunk = self.chunks[index]

            results.append({

                "rank": rank,

                "score": float(score),

                "chunk": chunk,
            })

        return results

    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    def hybrid_search(
        self,
        query,
        embedding_model,
        top_k=5
    ):

        """
        Return top K vector results and
        top K BM25 results separately.

        We intentionally don't merge them yet.
        """

        vector_results = (
            self.vector_search(
                query,
                embedding_model,
                top_k=top_k
            )
        )

        bm25_results = (
            self.bm25_search(
                query,
                top_k=top_k
            )
        )

        return {
            "vector": vector_results,
            "bm25": bm25_results,
        }

    # ========================================================
    # BACKWARD COMPATIBILITY
    # ========================================================

    def search(
        self,
        query,
        embedding_model,
        top_k=5
    ):

        return self.vector_search(
            query,
            embedding_model,
            top_k
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save(self, directory):

        os.makedirs(
            directory,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Save FAISS
        # ----------------------------------------------------

        faiss.write_index(
            self.index,
            os.path.join(
                directory,
                "index.faiss"
            )
        )

        # ----------------------------------------------------
        # Save chunks
        # ----------------------------------------------------

        with open(
            os.path.join(
                directory,
                "chunks.json"
            ),
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.chunks,
                file,
                ensure_ascii=False,
                indent=2
            )

        # ----------------------------------------------------
        # Save BM25 corpus
        # ----------------------------------------------------

        with open(
            os.path.join(
                directory,
                "bm25_corpus.json"
            ),
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.bm25_corpus,
                file,
                ensure_ascii=False
            )

    # ========================================================
    # LOAD
    # ========================================================

    def load(self, directory):

        # ----------------------------------------------------
        # Load FAISS
        # ----------------------------------------------------

        self.index = faiss.read_index(
            os.path.join(
                directory,
                "index.faiss"
            )
        )

        # ----------------------------------------------------
        # Load chunks
        # ----------------------------------------------------

        with open(
            os.path.join(
                directory,
                "chunks.json"
            ),
            "r",
            encoding="utf-8"
        ) as file:

            self.chunks = json.load(
                file
            )

        # ----------------------------------------------------
        # Load BM25 corpus
        # ----------------------------------------------------

        corpus_path = os.path.join(
            directory,
            "bm25_corpus.json"
        )

        if os.path.exists(corpus_path):

            with open(
                corpus_path,
                "r",
                encoding="utf-8"
            ) as file:

                self.bm25_corpus = json.load(
                    file
                )

        else:

            self.bm25_corpus = [
                self.tokenize(
                    chunk.get(
                        "content",
                        ""
                    )
                )
                for chunk in self.chunks
            ]

        self.bm25 = BM25Okapi(
            self.bm25_corpus
        )