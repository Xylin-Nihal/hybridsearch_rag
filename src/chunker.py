import re
import uuid


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):

    if not text:
        return []

    # Basic sentence splitter.
    #
    # We are deliberately keeping this simple for now.
    # Later we can replace this with a more robust
    # sentence segmentation model if necessary.

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
# TOKEN COUNT
# ============================================================

def estimate_tokens(text):

    if not text:
        return 0

    # Approximate token count.
    #
    # This is intentionally simple because the actual
    # embedding tokenizer depends on the API provider.
    #
    # Rough approximation:
    #
    # 1 token ≈ 4 characters
    #

    return max(
        1,
        len(text) // 4
    )


# ============================================================
# CONVERT SECTION TO TEXT
# ============================================================

def section_to_text(section):

    parts = []

    for element in section["elements"]:

        element_type = element["type"]

        content = element["content"]

        if not content:
            continue

        # ----------------------------------------------------
        # Normal text
        # ----------------------------------------------------

        if element_type == "text":

            parts.append(content)

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        elif element_type == "image":

            parts.append(
                "[IMAGE]\n"
                + content
            )

        # ----------------------------------------------------
        # Table
        # ----------------------------------------------------

        elif element_type == "table":

            parts.append(
                "[TABLE]\n"
                + content
            )

    return "\n\n".join(parts)


# ============================================================
# SPLIT LARGE SECTION INTO PARAGRAPHS
# ============================================================

def split_paragraphs(section):

    paragraphs = []

    for element in section["elements"]:

        content = element["content"]

        if not content:
            continue

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        if element["type"] == "image":

            paragraphs.append({
                "type": "image",
                "content": (
                    "[IMAGE]\n"
                    + content
                ),
            })

        # ----------------------------------------------------
        # Table
        # ----------------------------------------------------

        elif element["type"] == "table":

            paragraphs.append({
                "type": "table",
                "content": (
                    "[TABLE]\n"
                    + content
                ),
            })

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        else:

            text_parts = re.split(
                r"\n\s*\n",
                content
            )

            for text in text_parts:

                text = text.strip()

                if text:

                    paragraphs.append({
                        "type": "text",
                        "content": text,
                    })

    return paragraphs


# ============================================================
# SEMANTIC CHUNKING
# ============================================================

def semantic_chunk_sentences(
    sentences,
    embedding_model,
    target_tokens=400,
    max_tokens=500,
    threshold=0.65,
    overlap_sentences=1,
):

    if not sentences:

        return []

    # --------------------------------------------------------
    # Generate embeddings for all sentences
    # --------------------------------------------------------

    embeddings = embedding_model.encode(
        sentences
    )

    chunks = []

    current_sentences = []

    current_tokens = 0

    # ========================================================
    # Iterate through sentences
    # ========================================================

    for i, sentence in enumerate(sentences):

        sentence_tokens = estimate_tokens(
            sentence
        )

        # ----------------------------------------------------
        # First sentence
        # ----------------------------------------------------

        if not current_sentences:

            current_sentences.append(
                sentence
            )

            current_tokens = sentence_tokens

            continue

        # ----------------------------------------------------
        # Similarity with previous sentence
        # ----------------------------------------------------

        similarity = float(
            embeddings[i - 1]
            @
            embeddings[i]
        )

        # ----------------------------------------------------
        # Decide whether to create boundary
        # ----------------------------------------------------

        semantic_boundary = (
            similarity < threshold
        )

        size_boundary = (
            current_tokens
            + sentence_tokens
            > max_tokens
        )

        target_reached = (
            current_tokens
            >= target_tokens
        )

        # ----------------------------------------------------
        # Create new chunk
        # ----------------------------------------------------

        if (
            (semantic_boundary and target_reached)
            or size_boundary
        ):

            chunks.append(
                " ".join(current_sentences)
            )

            # ------------------------------------------------
            # Overlap previous sentences
            # ------------------------------------------------

            if overlap_sentences > 0:

                overlap = current_sentences[
                    -overlap_sentences:
                ]

            else:

                overlap = []

            current_sentences = (
                overlap.copy()
            )

            current_tokens = sum(
                estimate_tokens(sentence)
                for sentence in current_sentences
            )

        # ----------------------------------------------------
        # Add current sentence
        # ----------------------------------------------------

        current_sentences.append(
            sentence
        )

        current_tokens += sentence_tokens

    # ========================================================
    # Final chunk
    # ========================================================

    if current_sentences:

        chunks.append(
            " ".join(current_sentences)
        )

    return chunks


# ============================================================
# PROCESS LARGE SECTION
# ============================================================

def process_large_section(
    section,
    embedding_model,
    target_tokens,
    max_tokens,
    threshold,
    overlap_sentences,
):

    paragraphs = split_paragraphs(
        section
    )

    final_chunks = []

    current_sentences = []

    # ========================================================
    # Process paragraph by paragraph
    # ========================================================

    for paragraph in paragraphs:

        paragraph_type = paragraph["type"]

        content = paragraph["content"]

        # ----------------------------------------------------
        # IMAGE / TABLE
        #
        # Keep them at their original position.
        # ----------------------------------------------------

        if paragraph_type in [
            "image",
            "table"
        ]:

            # Process accumulated text first.
            if current_sentences:

                text_chunks = (
                    semantic_chunk_sentences(
                        current_sentences,
                        embedding_model,
                        target_tokens,
                        max_tokens,
                        threshold,
                        overlap_sentences,
                    )
                )

                for chunk in text_chunks:

                    final_chunks.append({
                        "chunk_type": "text",
                        "content": chunk,
                    })

                current_sentences = []

            # Add image/table exactly here.
            final_chunks.append({
                "chunk_type": paragraph_type,
                "content": content,
            })

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        else:

            sentences = split_sentences(
                content
            )

            current_sentences.extend(
                sentences
            )

    # ========================================================
    # Process remaining text
    # ========================================================

    if current_sentences:

        text_chunks = (
            semantic_chunk_sentences(
                current_sentences,
                embedding_model,
                target_tokens,
                max_tokens,
                threshold,
                overlap_sentences,
            )
        )

        for chunk in text_chunks:

            final_chunks.append({
                "chunk_type": "text",
                "content": chunk,
            })

    return final_chunks


# ============================================================
# BUILD ALL CHUNKS
# ============================================================

def build_chunks(
    sections,
    embedding_model,
    small_section_tokens=800,
    target_chunk_tokens=400,
    max_chunk_tokens=500,
    similarity_threshold=0.65,
    overlap_sentences=1,
):

    print("\n========== BUILDING CHUNKS ==========")

    all_chunks = []

    # ========================================================
    # Process each section
    # ========================================================

    for section in sections:

        section_text = section_to_text(
            section
        )

        section_tokens = estimate_tokens(
            section_text
        )

        print(
            f"\nSection: "
            f"{section['section_title']}"
        )

        print(
            f"Estimated tokens: "
            f"{section_tokens}"
        )

        # ====================================================
        # SMALL SECTION
        # ====================================================

        if section_tokens <= small_section_tokens:

            print(
                "→ Small section: keeping intact"
            )

            if section_text:

                all_chunks.append({
                    "chunk_id": str(uuid.uuid4()),

                    "section_id":
                        section["section_id"],

                    "section_index":
                        section["section_index"],

                    "section_title":
                        section["section_title"],

                    "section_path":
                        section["section_path"],

                    "chunk_index": 0,

                    "chunk_type": "section",

                    "content": section_text,

                    "token_count":
                        estimate_tokens(
                            section_text
                        ),
                })

            continue

        # ====================================================
        # LARGE SECTION
        # ====================================================

        print(
            "→ Large section: semantic chunking"
        )

        section_chunks = (
            process_large_section(
                section,
                embedding_model,
                target_chunk_tokens,
                max_chunk_tokens,
                similarity_threshold,
                overlap_sentences,
            )
        )

        # ====================================================
        # Add metadata
        # ====================================================

        for chunk_index, chunk in enumerate(
            section_chunks
        ):

            content = chunk["content"]

            all_chunks.append({
                "chunk_id": str(uuid.uuid4()),

                "section_id":
                    section["section_id"],

                "section_index":
                    section["section_index"],

                "section_title":
                    section["section_title"],

                "section_path":
                    section["section_path"],

                "chunk_index":
                    chunk_index,

                "chunk_type":
                    chunk["chunk_type"],

                "content":
                    content,

                "token_count":
                    estimate_tokens(
                        content
                    ),
            })

    print(
        f"\nTotal chunks created: "
        f"{len(all_chunks)}"
    )

    return all_chunks


# ============================================================
# DISPLAY CHUNKS
# ============================================================

def print_chunks(chunks):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL CHUNKS"
    )

    print(
        "=" * 80
    )

    for i, chunk in enumerate(chunks):

        print(
            f"\nCHUNK {i + 1}"
        )

        print(
            "-" * 80
        )

        print(
            f"Chunk ID     : "
            f"{chunk['chunk_id']}"
        )

        print(
            f"Section      : "
            f"{chunk['section_title']}"
        )

        print(
            f"Chunk Type   : "
            f"{chunk['chunk_type']}"
        )

        print(
            f"Tokens       : "
            f"{chunk['token_count']}"
        )

        print(
            "\nContent:"
        )

        print(
            chunk["content"]
        )