from src.config import (
    PDF_PATH,
    SMALL_SECTION_TOKENS,
    TARGET_CHUNK_TOKENS,
    MAX_CHUNK_TOKENS,
    SEMANTIC_SIMILARITY_THRESHOLD,
    CHUNK_OVERLAP_SENTENCES,
    validate_config,
)

from src.models import (
    EmbeddingModel,
    VisionModel,
)

from src.pdf_processor import (
    extract_pdf,
    build_sections,
    
)

from src.chunker import (
    build_chunks,
    print_chunks,
)

from src.vector_search import (
    VectorStore,
)


def main():

    print("\n" + "=" * 80)
    print("HYBRID SEARCH RAG")
    print("=" * 80)

    # ========================================================
    # Validate .env
    # ========================================================

    validate_config()

    # ========================================================
    # Initialize API clients
    # ========================================================

    print("\nInitializing API clients...")

    embedding_model = EmbeddingModel()

    vision_model = VisionModel()

    print("API clients initialized.")

    # ========================================================
    # PDF extraction
    # ========================================================

    elements = extract_pdf(
        PDF_PATH,
        vision_model
    )
    print("\n" + "=" * 80)
    print("RAW EXTRACTED ELEMENTS")
    print("=" * 80)

    for i, element in enumerate(elements):

        print(f"\nELEMENT {i}")
        print("-" * 80)

        print("TYPE:", element["type"])
        print("CONTENT:", element["text"][:500])
    # ========================================================
    # Build sections
    # ========================================================

    sections = build_sections(
        elements
    )

    """print_sections(
        sections
    )"""

    # ========================================================
    # Build chunks
    # ========================================================

    chunks = build_chunks(
    sections,
    embedding_model
)

    # ========================================================
    # Print chunks
    # ========================================================

    #print_chunks(
        #chunks
    #)

    # ========================================================
    # Temporary FAISS store
    # ========================================================

    vector_store = VectorStore()

    vector_store.build(
        chunks,
        embedding_model
    )

    # ========================================================
    # Retrieval test
    # ========================================================

    print("\n" + "=" * 80)
    print("RAG RETRIEVAL TEST")
    print("Type 'exit' to quit.")
    print("=" * 80)

    while True:

        query = input(
            "\nAsk a question: "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = vector_store.search(
            query,
            embedding_model,
            top_k=5
        )

        print(
            "\n========== SEARCH RESULTS =========="
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            chunk = result["chunk"]

            print(
                f"\nRESULT {rank}"
            )

            print("-" * 80)

            print(
                f"Similarity : "
                f"{result['score']:.4f}"
            )

            print(
                f"Section    : "
                f"{chunk['section_title']}"
            )

            print(
                f"Type       : "
                f"{chunk['chunk_type']}"
            )

            print(
                f"\n{chunk['content']}"
            )


if __name__ == "__main__":
    main()