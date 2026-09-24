import os
import json
import re

import faiss
import numpy as np

from rank_bm25 import BM25Okapi


class VectorStore:

    def __init__(self):

        self.index = None

        self.bm25 = None

        self.bm25_corpus = []

        self.chunks = []


    # =========================================================
    # TOKENIZATION
    # =========================================================

    def tokenize(self, text):

        if not text:
            return []

        return re.findall(
            r"\b\w+\b",
            text.lower()
        )


    # =========================================================
    # BUILD INDEX
    # =========================================================

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

        texts = [
            chunk.get(
                "content",
                ""
            )
            for chunk in chunks
        ]


        # =====================================================
        # VECTOR INDEX
        # =====================================================

        print(
            "\nBuilding vector index..."
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

        print(
            f"Vector index: "
            f"{len(chunks)} chunks"
        )


        # =====================================================
        # BM25 INDEX
        # =====================================================

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


    # =========================================================
    # VECTOR SEARCH
    # =========================================================

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

        for rank, (
            score,
            index
        ) in enumerate(
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

                "score": float(
                    score
                ),

                "chunk": chunk

            })

        return results


    # =========================================================
    # BM25 SEARCH
    # =========================================================

    def bm25_search(
        self,
        query,
        top_k=5
    ):

        if self.bm25 is None:

            raise RuntimeError(
                "BM25 index has not been built."
            )

        query_tokens = (
            self.tokenize(query)
        )

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

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

                "score": float(
                    score
                ),

                "chunk": chunk

            })

        return results


    # =========================================================
    # HYBRID SEARCH
    # =========================================================

    def hybrid_search(
        self,
        query,
        embedding_model,
        top_k=5
    ):

        vector_results = (
            self.vector_search(
                query,
                embedding_model,
                top_k
            )
        )

        bm25_results = (
            self.bm25_search(
                query,
                top_k
            )
        )

        return {

            "vector": vector_results,

            "bm25": bm25_results

        }


    # =========================================================
    # DEDUPLICATE CANDIDATE POOL
    # =========================================================

    def build_candidate_pool(
        self,
        hybrid_results
    ):

        print(
            "\n========== BUILDING CANDIDATE POOL =========="
        )

        candidates = {}

        vector_results = (
            hybrid_results.get(
                "vector",
                []
            )
        )

        bm25_results = (
            hybrid_results.get(
                "bm25",
                []
            )
        )


        # =====================================================
        # ADD VECTOR RESULTS
        # =====================================================

        for result in vector_results:

            chunk = result["chunk"]

            chunk_id = chunk.get(
                "chunk_id"
            )

            if chunk_id is None:
                continue

            candidates[chunk_id] = {

                "chunk": chunk,

                "vector_score":
                    result["score"],

                "vector_rank":
                    result["rank"],

                "bm25_score":
                    None,

                "bm25_rank":
                    None,

                "retrieval_sources":
                    ["vector"]

            }


        # =====================================================
        # ADD BM25 RESULTS
        # =====================================================

        for result in bm25_results:

            chunk = result["chunk"]

            chunk_id = chunk.get(
                "chunk_id"
            )

            if chunk_id is None:
                continue


            # Same chunk already found
            # by vector search.
            if chunk_id in candidates:

                candidates[
                    chunk_id
                ]["bm25_score"] = (
                    result["score"]
                )

                candidates[
                    chunk_id
                ]["bm25_rank"] = (
                    result["rank"]
                )

                candidates[
                    chunk_id
                ]["retrieval_sources"].append(
                    "bm25"
                )


            # New BM25-only candidate
            else:

                candidates[chunk_id] = {

                    "chunk": chunk,

                    "vector_score":
                        None,

                    "vector_rank":
                        None,

                    "bm25_score":
                        result["score"],

                    "bm25_rank":
                        result["rank"],

                    "retrieval_sources":
                        ["bm25"]

                }


        candidate_pool = list(
            candidates.values()
        )


        print(
            f"Vector candidates: "
            f"{len(vector_results)}"
        )

        print(
            f"BM25 candidates: "
            f"{len(bm25_results)}"
        )

        print(
            f"Unique candidates: "
            f"{len(candidate_pool)}"
        )


        for i, candidate in enumerate(
            candidate_pool,
            start=1
        ):

            chunk = candidate["chunk"]

            print(
                f"\nCandidate {i}"
            )

            print(
                f"Chunk ID: "
                f"{chunk.get('chunk_id')}"
            )

            print(
                f"Section: "
                f"{chunk.get('section_title')}"
            )

            print(
                f"Sources: "
                f"{candidate['retrieval_sources']}"
            )


        return candidate_pool


    # =========================================================
    # RERANK
    # =========================================================

    def rerank(
        self,
        query,
        candidate_pool,
        reranker_model,
        top_k=5
    ):

        print(
            "\n========== RERANKING CANDIDATES =========="
        )

        if not candidate_pool:

            return []


        reranked = reranker_model.rerank(
            query=query,
            candidates=candidate_pool,
            top_k=top_k
        )


        print(
            f"\nTop {len(reranked)} reranked chunks:"
        )


        for rank, result in enumerate(
            reranked,
            start=1
        ):

            chunk = result["chunk"]

            print(
                f"\nRank {rank}"
            )

            print(
                f"Rerank score: "
                f"{result['rerank_score']:.4f}"
            )

            print(
                f"Section: "
                f"{chunk.get('section_title')}"
            )

            print(
                f"Type: "
                f"{chunk.get('chunk_type')}"
            )

            print(
                f"Sources: "
                f"{result['retrieval_sources']}"
            )

            print(
                f"Content: "
                f"{chunk.get('content', '')[:300]}"
            )


        return reranked


    # =========================================================
    # COMPLETE RETRIEVAL PIPELINE
    # =========================================================

    def search_with_reranker(
        self,
        query,
        embedding_model,
        reranker_model,
        retrieval_top_k=5,
        final_top_k=5
    ):

        # -----------------------------------------------------
        # STEP 1: Dense + BM25
        # -----------------------------------------------------

        hybrid_results = (
            self.hybrid_search(
                query=query,
                embedding_model=embedding_model,
                top_k=retrieval_top_k
            )
        )


        # -----------------------------------------------------
        # STEP 2: Deduplicate
        # -----------------------------------------------------

        candidate_pool = (
            self.build_candidate_pool(
                hybrid_results
            )
        )


        # -----------------------------------------------------
        # STEP 3: Cross-encoder reranking
        # -----------------------------------------------------

        final_results = (
            self.rerank(
                query=query,
                candidate_pool=candidate_pool,
                reranker_model=reranker_model,
                top_k=final_top_k
            )
        )


        return {

            "hybrid": hybrid_results,

            "candidates": candidate_pool,

            "reranked": final_results

        }


    # =========================================================
    # BACKWARD COMPATIBILITY
    # =========================================================

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


    # =========================================================
    # SAVE
    # =========================================================

    def save(
        self,
        directory
    ):

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


    # =========================================================
    # LOAD
    # =========================================================

    def load(
        self,
        directory
    ):

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

            self.chunks = json.load(file)


        corpus_path = os.path.join(
            directory,
            "bm25_corpus.json"
        )


        if os.path.exists(
            corpus_path
        ):

            with open(
                corpus_path,
                "r",
                encoding="utf-8"
            ) as file:

                self.bm25_corpus = (
                    json.load(file)
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