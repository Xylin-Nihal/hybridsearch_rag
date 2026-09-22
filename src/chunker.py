import re
import uuid
import numpy as np

from src.config import (
    SMALL_SECTION_TOKENS,
    TARGET_CHUNK_TOKENS,
    MAX_CHUNK_TOKENS,
    SEMANTIC_SIMILARITY_THRESHOLD,
    CHUNK_OVERLAP_SENTENCES,
)


# ============================================================
# TOKEN ESTIMATION
# ============================================================

def estimate_tokens(text):
    """
    Approximate token count.

    Later we can replace this with the actual tokenizer.
    """

    if not text:
        return 0

    return max(1, len(text) // 4)


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# CREATE TEXT CHUNK
# ============================================================

def create_text_chunk(
    text,
    section,
    chunk_index
):

    return {

        "chunk_id": str(uuid.uuid4()),

        "chunk_type": "text",

        "content": text,

        "token_count": estimate_tokens(text),

        "section_id": section["section_id"],

        "section_title": section["section_title"],

        "section_path": section["section_path"],

        "chunk_index": chunk_index,

        # No visual associated with this chunk
        "image_base64": None,

        "mime_type": None,
    }


# ============================================================
# CREATE VISUAL CHUNK
# ============================================================

def create_visual_chunk(
    element,
    section,
    chunk_index
):

    visual_type = element["type"]

    return {

        "chunk_id": str(uuid.uuid4()),

        # image OR table
        "chunk_type": visual_type,

        # IMPORTANT:
        # This will contain the vision-model description.
        # It is used for embedding/retrieval.
        "content": element.get("text", ""),

        "token_count": estimate_tokens(
            element.get("text", "")
        ),

        "section_id": section["section_id"],

        "section_title": section["section_title"],

        "section_path": section["section_path"],

        "chunk_index": chunk_index,

        # ====================================================
        # ACTUAL VISUAL
        # ====================================================

        "image_base64": element.get(
            "image_base64"
        ),

        "mime_type": element.get(
            "mime_type"
        ),
    }


# ============================================================
# SEMANTIC CHUNKING
# ============================================================

def semantic_chunk_sentences(
    sentences,
    section,
    embedding_model,
    starting_chunk_index=0,
):

    if not sentences:
        return []

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = embedding_model.encode(
        sentences
    )

    chunks = []

    current_sentences = []
    current_tokens = 0

    chunk_index = starting_chunk_index

    for i, sentence in enumerate(sentences):

        sentence_tokens = estimate_tokens(sentence)

        # ----------------------------------------------------
        # First sentence
        # ----------------------------------------------------

        if not current_sentences:

            current_sentences.append(sentence)
            current_tokens = sentence_tokens

            continue

        # ----------------------------------------------------
        # Semantic similarity
        # ----------------------------------------------------

        previous_embedding = embeddings[i - 1]
        current_embedding = embeddings[i]

        similarity = float(
            np.dot(
                previous_embedding,
                current_embedding
            )
        )

        semantic_boundary = (
            similarity <
            SEMANTIC_SIMILARITY_THRESHOLD
        )

        size_boundary = (
            current_tokens >=
            TARGET_CHUNK_TOKENS
        )

        max_boundary = (
            current_tokens + sentence_tokens
            > MAX_CHUNK_TOKENS
        )

        # ----------------------------------------------------
        # Create boundary
        # ----------------------------------------------------

        if (
            max_boundary
            or
            (
                semantic_boundary
                and size_boundary
            )
        ):

            chunk_text = " ".join(
                current_sentences
            )

            chunks.append(
                create_text_chunk(
                    chunk_text,
                    section,
                    chunk_index
                )
            )

            chunk_index += 1

            # ------------------------------------------------
            # Sentence overlap
            # ------------------------------------------------

            overlap = current_sentences[
                -CHUNK_OVERLAP_SENTENCES:
            ]

            current_sentences = overlap.copy()

            current_tokens = sum(
                estimate_tokens(s)
                for s in current_sentences
            )

        # ----------------------------------------------------
        # Add current sentence
        # ----------------------------------------------------

        current_sentences.append(sentence)

        current_tokens += sentence_tokens

    # ========================================================
    # FINAL CHUNK
    # ========================================================

    if current_sentences:

        chunk_text = " ".join(
            current_sentences
        )

        chunks.append(
            create_text_chunk(
                chunk_text,
                section,
                chunk_index
            )
        )

    return chunks


# ============================================================
# PROCESS LARGE SECTION
# ============================================================

def process_large_section(
    section,
    embedding_model
):

    final_chunks = []

    text_buffer = []

    chunk_index = 0

    def flush_text():

        nonlocal chunk_index

        if not text_buffer:
            return

        text = " ".join(text_buffer)

        sentences = split_sentences(text)

        new_chunks = semantic_chunk_sentences(
            sentences,
            section,
            embedding_model,
            starting_chunk_index=chunk_index,
        )

        final_chunks.extend(new_chunks)

        chunk_index += len(new_chunks)

        text_buffer.clear()

    # ========================================================
    # PROCESS ELEMENTS IN ORIGINAL PDF ORDER
    # ========================================================

    for element in section["elements"]:

        element_type = element["type"]

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        if element_type == "text":

            text = element.get("text", "").strip()

            if text:
                text_buffer.append(text)

        # ----------------------------------------------------
        # IMAGE / TABLE
        # ----------------------------------------------------

        elif element_type in ["image", "table"]:

            # Finish text before visual
            flush_text()

            # Add visual exactly where it appeared
            visual_chunk = create_visual_chunk(
                element,
                section,
                chunk_index
            )

            final_chunks.append(
                visual_chunk
            )

            chunk_index += 1

    # Flush remaining text
    flush_text()

    return final_chunks


# ============================================================
# BUILD ALL CHUNKS
# ============================================================

def build_chunks(
    sections,
    embedding_model
):

    print("\n========== BUILDING CHUNKS ==========")

    all_chunks = []

    for section in sections:

        section_text = []

        for element in section["elements"]:

            if element["type"] == "text":

                section_text.append(
                    element.get("text", "")
                )

            elif element["type"] in [
                "image",
                "table"
            ]:

                # Vision description
                section_text.append(
                    element.get("text", "")
                )

        combined_text = " ".join(
            section_text
        )

        estimated_tokens = estimate_tokens(
            combined_text
        )

        # ====================================================
        # SMALL SECTION
        # ====================================================

        if estimated_tokens <= SMALL_SECTION_TOKENS:

            # Even small sections must preserve
            # image/table positions.

            chunks = process_large_section(
                section,
                embedding_model
            )

        # ====================================================
        # LARGE SECTION
        # ====================================================

        else:

            chunks = process_large_section(
                section,
                embedding_model
            )

        all_chunks.extend(chunks)

    return all_chunks


# ============================================================
# PRINT CHUNKS
# ============================================================

def print_chunks(chunks):

    print("\n========== CHUNKS ==========")

    for i, chunk in enumerate(chunks):

        print("\n--------------------------------")

        print(
            f"Chunk {i}"
        )

        print(
            f"ID: {chunk['chunk_id']}"
        )

        print(
            f"Type: {chunk['chunk_type']}"
        )

        print(
            f"Section: {chunk['section_title']}"
        )

        print(
            f"Path: "
            f"{' > '.join(chunk['section_path'])}"
        )

        print(
            f"Tokens: {chunk['token_count']}"
        )

        # Don't print the huge base64 string
        if chunk["chunk_type"] in [
            "image",
            "table"
        ]:

            print(
                "Actual visual: "
                + (
                    "AVAILABLE"
                    if chunk.get("image_base64")
                    else "NOT AVAILABLE"
                )
            )

            print(
                f"Description:\n"
                f"{chunk['content'][:500]}"
            )

        else:

            print(
                f"Content:\n"
                f"{chunk['content'][:500]}"
            )