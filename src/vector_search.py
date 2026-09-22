import os
import json

import faiss
import numpy as np


# ============================================================
# VECTOR STORE
# ============================================================

class VectorStore:

    def __init__(self):

        self.index = None

        self.chunks = []

    # ========================================================
    # BUILD INDEX
    # ========================================================

    def build(
        self,
        chunks,
        embedding_model
    ):

        print(
            "\n========== BUILDING VECTOR STORE =========="
        )

        texts = [
            chunk["content"]
            for chunk in chunks
        ]

        if not texts:

            raise ValueError(
                "No chunks available."
            )

        embeddings = (
            embedding_model.encode(texts)
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

        self.chunks = chunks

        print(
            f"Indexed {len(chunks)} chunks."
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query,
        embedding_model,
        top_k=5
    ):

        if self.index is None:

            raise RuntimeError(
                "Vector store has not been built."
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

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            chunk = self.chunks[index]

            results.append({
                "score": float(score),

                "chunk": chunk,
            })

        return results

    # ========================================================
    # SAVE
    # ========================================================

    def save(self, directory):

        os.makedirs(
            directory,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            os.path.join(
                directory,
                "index.faiss"
            )
        )

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

    # ========================================================
    # LOAD
    # ========================================================

    def load(self, directory):

        self.index = faiss.read_index(
            os.path.join(
                directory,
                "index.faiss"
            )
        )

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