def display_results(results):
    print("\n========== RETRIEVED CONTEXT ==========")

    for i, result in enumerate(results):
        metadata = result["metadata"]

        print(f"\nCONTEXT {i + 1}")
        print("Score:", result["score"])
        print("Type:", metadata.get("chunk_type"))
        print("Page:", metadata.get("page_number"))
        print("Section:", metadata.get("section_title"))
        print("Section Path:", metadata.get("section_path"))

        if metadata.get("chunk_type") == "image":
            print(
                "Image ID:",
                metadata.get("image_id")
            )
            print(
                "Image Path:",
                metadata.get("image_path")
            )

        print("\nContent:")
        print(result["content"][:2000])

        print("\n" + "-" * 70)