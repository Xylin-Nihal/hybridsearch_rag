import os
import json
import uuid
from pathlib import Path

import pymupdf
import faiss
import numpy as np
import torch

from unstructured.partition.pdf import partition_pdf

from unstructured.documents.elements import (
    Text,
    Title,
    NarrativeText,
    ListItem,
    Table,
)

from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_experimental.text_splitter import SemanticChunker


# ============================================================
# CONFIG
# ============================================================

PDF_PATH = "data/attention-is-all-you-need.pdf"

IMAGE_DIR = Path("extracted_images")
VECTOR_DIR = Path("vectorstore")

IMAGE_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

TOP_K = 5


# ============================================================
# 1. EXTRACT IMAGES USING PYMUPDF
# ============================================================

def extract_images(pdf_path):

    print("\n========== IMAGE EXTRACTION ==========")

    pdf = pymupdf.open(pdf_path)

    image_records = []

    for page_index, page in enumerate(pdf):

        page_number = page_index + 1

        images = page.get_images(
            full=True
        )

        print(
            f"Page {page_number}: "
            f"{len(images)} images"
        )

        for image_index, image in enumerate(images):

            xref = image[0]

            image_data = pdf.extract_image(
                xref
            )

            image_bytes = image_data[
                "image"
            ]

            extension = image_data[
                "ext"
            ]

            image_id = str(
                uuid.uuid4()
            )

            filename = (
                f"page_{page_number}_"
                f"image_{image_index + 1}_"
                f"{image_id}.{extension}"
            )

            image_path = (
                IMAGE_DIR / filename
            )

            with open(
                image_path,
                "wb"
            ) as f:

                f.write(
                    image_bytes
                )

            record = {

                "image_id": image_id,

                "page_number":
                    page_number,

                "image_index":
                    image_index,

                "image_path":
                    str(image_path),

                "type":
                    "image",

            }

            image_records.append(
                record
            )

            print(
                "Saved:",
                image_path
            )

    pdf.close()

    return image_records


# ============================================================
# 2. EXTRACT TEXT + TABLES USING UNSTRUCTURED
# ============================================================

def extract_elements(pdf_path):

    print("\n========== DOCUMENT EXTRACTION ==========")

    elements = partition_pdf(

        filename=pdf_path,

        # Better layout understanding
        strategy="fast",

        # Detect table structure
        infer_table_structure=True,

    )

    print(
        f"Total elements: {len(elements)}"
    )

    documents = []


    for element in elements:

        page_number = getattr(
            element.metadata,
            "page_number",
            None
        )

        element_id = str(
            uuid.uuid4()
        )


        # ====================================================
        # TEXT ELEMENT
        # ====================================================

        if isinstance(
            element,
            (
                Text,
                Title,
                NarrativeText,
                ListItem,
            )
        ):

            content = str(
                element
            ).strip()

            if not content:
                continue

            documents.append(

                Document(

                    page_content=content,

                    metadata={

                        "element_id":
                            element_id,

                        "type":
                            "text",

                        "page_number":
                            page_number,

                        "source":
                            PDF_PATH,

                    }
                )
            )


        # ====================================================
        # TABLE ELEMENT
        # ====================================================

        elif isinstance(
            element,
            Table
        ):

            table_html = getattr(
                element.metadata,
                "text_as_html",
                None
            )

            if table_html:

                content = (
                    "TABLE:\n"
                    + table_html
                )

            else:

                content = (
                    "TABLE:\n"
                    + str(element)
                )


            documents.append(

                Document(

                    page_content=content,

                    metadata={

                        "element_id":
                            element_id,

                        "type":
                            "table",

                        "page_number":
                            page_number,

                        "source":
                            PDF_PATH,

                        # Never split tables
                        "atomic":
                            True,

                    }
                )
            )


    print(
        f"Text/Table elements: "
        f"{len(documents)}"
    )

    return documents


# ============================================================
# 3. LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():

    print(
        "\n========== EMBEDDING MODEL =========="
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    embeddings = HuggingFaceEmbeddings(

        model_name=
            EMBEDDING_MODEL,

        model_kwargs={
            "device": device
        },

        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings


# ============================================================
# 4. SEMANTIC CHUNKING
# ============================================================

def semantic_chunking(
    documents,
    embeddings
):

    print(
        "\n========== SEMANTIC CHUNKING =========="
    )

    splitter = SemanticChunker(

        embeddings=embeddings,

        breakpoint_threshold_type=
            "percentile",

        breakpoint_threshold_amount=
            85,

        add_start_index=True,
    )


    final_chunks = []


    for document in documents:

        element_type = (
            document.metadata["type"]
        )


        # ====================================================
        # TEXT
        # ====================================================

        if element_type == "text":

            chunks = (
                splitter
                .split_documents(
                    [document]
                )
            )

            for chunk in chunks:

                chunk.metadata[
                    "chunk_id"
                ] = str(uuid.uuid4())

                chunk.metadata[
                    "chunk_type"
                ] = "text"

                final_chunks.append(
                    chunk
                )


        # ====================================================
        # TABLE
        # ====================================================

        elif element_type == "table":

            # Keep entire table together

            document.metadata[
                "chunk_id"
            ] = str(uuid.uuid4())

            document.metadata[
                "chunk_type"
            ] = "table"

            final_chunks.append(
                document
            )


    print(
        f"Final chunks: "
        f"{len(final_chunks)}"
    )

    return final_chunks


# ============================================================
# 5. CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    chunks,
    embedding_model
):

    print(
        "\n========== CREATING EMBEDDINGS =========="
    )

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    vectors = (
        embedding_model
        .embed_documents(
            texts
        )
    )

    vectors = np.array(
        vectors,
        dtype="float32"
    )

    print(
        "Embedding shape:",
        vectors.shape
    )

    return vectors


# ============================================================
# 6. CREATE FAISS INDEX
# ============================================================

def create_faiss_index(
    vectors
):

    print(
        "\n========== FAISS =========="
    )

    dimension = vectors.shape[1]

    # Because embeddings are normalized,
    # Inner Product = cosine similarity

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        vectors
    )

    print(
        "Vectors stored:",
        index.ntotal
    )

    return index


# ============================================================
# 7. SAVE FAISS INDEX
# ============================================================

def save_vectorstore(
    index,
    chunks
):

    index_path = (
        VECTOR_DIR
        / "index.faiss"
    )

    metadata_path = (
        VECTOR_DIR
        / "chunks.json"
    )


    # Save FAISS

    faiss.write_index(
        index,
        str(index_path)
    )


    # Save chunks/metadata

    chunk_data = []


    for chunk in chunks:

        chunk_data.append({

            "page_content":
                chunk.page_content,

            "metadata":
                chunk.metadata,

        })


    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunk_data,
            f,
            indent=2,
            ensure_ascii=False
        )


    print(
        "\nVector store saved."
    )

    print(
        "FAISS:",
        index_path
    )

    print(
        "Metadata:",
        metadata_path
    )


# ============================================================
# 8. LOAD VECTORSTORE
# ============================================================

def load_vectorstore():

    index_path = (
        VECTOR_DIR
        / "index.faiss"
    )

    metadata_path = (
        VECTOR_DIR
        / "chunks.json"
    )


    index = faiss.read_index(
        str(index_path)
    )


    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as f:

        chunk_data = json.load(
            f
        )


    chunks = []


    for item in chunk_data:

        chunks.append(

            Document(

                page_content=
                    item["page_content"],

                metadata=
                    item["metadata"]
            )
        )


    return index, chunks


# ============================================================
# 9. RETRIEVAL
# ============================================================

def retrieve(
    query,
    embedding_model,
    index,
    chunks,
    image_records,
    top_k=5
):

    print(
        "\n========== RETRIEVAL =========="
    )

    print(
        "Query:",
        query
    )


    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_vector = (
        embedding_model
        .embed_query(
            query
        )
    )

    query_vector = np.array(
        [query_vector],
        dtype="float32"
    )


    # --------------------------------------------------------
    # FAISS search
    # --------------------------------------------------------

    scores, indices = (
        index.search(
            query_vector,
            top_k
        )
    )


    results = []


    for score, chunk_index in zip(
        scores[0],
        indices[0]
    ):

        chunk = chunks[
            chunk_index
        ]


        result = {

            "score":
                float(score),

            "content":
                chunk.page_content,

            "metadata":
                chunk.metadata,

        }


        # ====================================================
        # FIND RELATED IMAGES
        # ====================================================

        page_number = (
            chunk.metadata
            .get("page_number")
        )


        related_images = [

            image

            for image
            in image_records

            if image[
                "page_number"
            ] == page_number

        ]


        result[
            "related_images"
        ] = related_images


        results.append(
            result
        )


    return results


# ============================================================
# 10. PRINT RETRIEVAL RESULTS
# ============================================================

def display_results(
    results
):

    print(
        "\n\n========== RETRIEVED CONTEXT =========="
    )


    for i, result in enumerate(
        results
    ):

        print(
            f"\nRESULT {i + 1}"
        )

        print(
            "Score:",
            result["score"]
        )

        print(
            "Type:",
            result[
                "metadata"
            ]["chunk_type"]
        )

        print(
            "Page:",
            result[
                "metadata"
            ]["page_number"]
        )

        print(
            "\nContent:"
        )

        print(
            result["content"][
                :1500
            ]
        )


        # ----------------------------------------------------
        # RELATED IMAGES
        # ----------------------------------------------------

        images = (
            result[
                "related_images"
            ]
        )


        if images:

            print(
                "\nRelated images:"
            )

            for image in images:

                print(
                    image[
                        "image_path"
                    ]
                )


        print(
            "\n" + "-" * 70
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def build_rag():

    # --------------------------------------------------------
    # STEP 1
    # Extract images
    # --------------------------------------------------------

    image_records = extract_images(
        PDF_PATH
    )


    # --------------------------------------------------------
    # STEP 2
    # Extract text + tables
    # --------------------------------------------------------

    documents = extract_elements(
        PDF_PATH
    )


    # --------------------------------------------------------
    # STEP 3
    # Load embedding model
    # --------------------------------------------------------

    embedding_model = (
        load_embedding_model()
    )


    # --------------------------------------------------------
    # STEP 4
    # Semantic chunking
    # --------------------------------------------------------

    chunks = semantic_chunking(

        documents,

        embedding_model

    )


    # --------------------------------------------------------
    # STEP 5
    # Create embeddings
    # --------------------------------------------------------

    vectors = create_embeddings(

        chunks,

        embedding_model

    )


    # --------------------------------------------------------
    # STEP 6
    # FAISS
    # --------------------------------------------------------

    index = create_faiss_index(
        vectors
    )


    # --------------------------------------------------------
    # STEP 7
    # Save
    # --------------------------------------------------------

    save_vectorstore(

        index,

        chunks

    )


    return (
        embedding_model,
        index,
        chunks,
        image_records
    )


# ============================================================
# APPLICATION
# ============================================================

if __name__ == "__main__":

    (
        embedding_model,
        index,
        chunks,
        image_records
    ) = build_rag()


    print(
        "\n\nRAG SYSTEM READY."
    )


    while True:

        query = input(
            "\nAsk a question "
            "(type 'exit' to quit): "
        )


        if query.lower() == "exit":

            break


        results = retrieve(

            query,

            embedding_model,

            index,

            chunks,

            image_records,

            TOP_K

        )


        display_results(
            results
        )