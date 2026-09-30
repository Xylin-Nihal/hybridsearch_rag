from src.config import (
    PDF_PATH,
    validate_config,
)

from src.models import (
    EmbeddingModel,
    VisionModel,
    RerankerModel,
    GroqModel,
)

from src.pdf_processor import (
    extract_pdf,
    build_sections,
)

from src.chunker import (
    build_chunks,
)

from src.vector_search import (
    VectorStore,
)


def main():

    print("\n" + "=" * 80)
    print("HYBRID SEARCH RAG CHATBOT")
    print("=" * 80)


    # ========================================================
    # Validate .env
    # ========================================================

    validate_config()


    # ========================================================
    # Initialize API clients
    # ========================================================

    print("\nInitializing models...")

    embedding_model = EmbeddingModel()

    vision_model = VisionModel()

    reranker_model = RerankerModel()

    llm = GroqModel()

    print("\nAll models initialized.")


    # ========================================================
    # PDF extraction
    # ========================================================

    print("\n" + "=" * 80)
    print("PDF EXTRACTION")
    print("=" * 80)

    elements = extract_pdf(
        PDF_PATH,
        vision_model
    )


    # ========================================================
    # Build sections
    # ========================================================

    print("\n" + "=" * 80)
    print("BUILDING SECTIONS")
    print("=" * 80)

    sections = build_sections(
        elements
    )


    # ========================================================
    # Build chunks
    # ========================================================

    print("\n" + "=" * 80)
    print("BUILDING CHUNKS")
    print("=" * 80)

    chunks = build_chunks(
        sections,
        embedding_model
    )

    print(
        f"\nTotal chunks: {len(chunks)}"
    )


    # ========================================================
    # Build hybrid search store
    # ========================================================

    print("\n" + "=" * 80)
    print("BUILDING HYBRID SEARCH")
    print("=" * 80)

    vector_store = VectorStore()

    vector_store.build(
        chunks,
        embedding_model
    )


    # ========================================================
    # Conversation history
    # ========================================================

    conversation_history = []


    # ========================================================
    # Chatbot
    # ========================================================

    print("\n" + "=" * 80)
    print("RAG CHATBOT")
    print("=" * 80)

    print(
        "\nAsk questions about the document."
    )

    print(
        "Type 'exit' to quit."
    )


    while True:

        query = input(
            "\nYou: "
        ).strip()


        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if query.lower() == "exit":

            print(
                "\nGoodbye!"
            )

            break


        if not query:
            continue


        try:

            # =================================================
            # RETRIEVAL
            # =================================================

            results = (
                vector_store
                .search_with_reranker(

                    query=query,

                    embedding_model=
                        embedding_model,

                    reranker_model=
                        reranker_model,

                    retrieval_top_k=5,

                    final_top_k=5
                )
            )


            # =================================================
            # FINAL RERANKED CHUNKS
            # =================================================

            reranked_results = (
                results["reranked"]
            )


            if not reranked_results:

                print(
                    "\nAssistant: "
                    "I couldn't find relevant "
                    "information in the document."
                )

                continue


            # =================================================
            # BUILD LLM CONTEXT
            # =================================================

            context = (
                vector_store
                .build_llm_context(
                    reranked_results
                )
            )


            # =================================================
            # GENERATE ANSWER
            # =================================================

            answer = llm.generate(

                query=query,

                context=context,

                history=
                    conversation_history

            )


            # =================================================
            # PRINT ANSWER
            # =================================================

            print(
                "\nAssistant:"
            )

            print(
                answer
            )
            print(
                "\nSources:"
            )

            for rank, result in enumerate(
                reranked_results,
                start=1
            ):

                chunk = result["chunk"]

                print(
                    f"  [{rank}] "
                    f"{chunk.get('section_title')} "
                    f"| {chunk.get('chunk_type')} "
                    f"| score="
                    f"{result['rerank_score']:.4f}"
                )

            # =================================================
            # UPDATE CONVERSATION HISTORY
            # =================================================

            conversation_history.append({

                "role": "user",

                "content": query

            })


            conversation_history.append({

                "role": "assistant",

                "content": answer

            })


            # =================================================
            # KEEP HISTORY REASONABLE
            # =================================================

            # Keep the latest 10 messages
            # = 5 conversation turns.

            if len(
                conversation_history
            ) > 10:

                conversation_history = (
                    conversation_history[-10:]
                )


        except Exception as e:

            print(
                "\n[ERROR]"
            )

            print(e)


if __name__ == "__main__":
    main()