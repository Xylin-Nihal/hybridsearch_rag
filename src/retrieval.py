import numpy as np


def expand_with_neighbors(
    retrieved_chunks,
    chunks,
    window=1,
):
    chunk_lookup = {
        chunk.metadata["chunk_id"]: chunk
        for chunk in chunks
    }

    expanded = []
    already_added = set()

    for retrieved in retrieved_chunks:
        center = retrieved["chunk"]

        center_id = center.metadata["chunk_id"]

        chain = []

        current = center
        previous_ids = []

        for _ in range(window):
            prev_id = current.metadata.get(
                "prev_chunk_id"
            )

            if not prev_id:
                break

            previous_ids.append(prev_id)

            if prev_id not in chunk_lookup:
                break

            current = chunk_lookup[prev_id]

        previous_ids.reverse()

        chain.extend(previous_ids)
        chain.append(center_id)

        current = center

        for _ in range(window):
            next_id = current.metadata.get(
                "next_chunk_id"
            )

            if not next_id:
                break

            chain.append(next_id)

            if next_id not in chunk_lookup:
                break

            current = chunk_lookup[next_id]

        for chunk_id in chain:
            if chunk_id in already_added:
                continue

            if chunk_id not in chunk_lookup:
                continue

            expanded.append(
                chunk_lookup[chunk_id]
            )

            already_added.add(chunk_id)

    return expanded


def retrieve(
    query,
    embedding_model,
    index,
    chunks,
    top_k=5,
    neighbor_window=1,
):
    print("\n========== RETRIEVAL ==========")
    print("Query:", query)

    query_vector = embedding_model.embed_query(query)

    query_vector = np.array(
        [query_vector],
        dtype="float32"
    )

    scores, indices = index.search(
        query_vector,
        top_k
    )

    retrieved = []

    for score, chunk_index in zip(
        scores[0],
        indices[0]
    ):
        if chunk_index < 0:
            continue

        chunk = chunks[chunk_index]

        retrieved.append({
            "score": float(score),
            "chunk": chunk,
        })

    expanded_chunks = expand_with_neighbors(
        retrieved,
        chunks,
        window=neighbor_window,
    )

    score_lookup = {
        item["chunk"].metadata["chunk_id"]: item["score"]
        for item in retrieved
    }

    results = []

    for chunk in expanded_chunks:
        chunk_id = chunk.metadata["chunk_id"]

        results.append({
            "score": score_lookup.get(
                chunk_id,
                None
            ),
            "content": chunk.page_content,
            "metadata": chunk.metadata,
        })

    return results