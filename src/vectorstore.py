import json
import faiss

from langchain_core.documents import Document

from .config import VECTOR_DIR


def create_faiss_index(vectors):
    print("\n========== FAISS ==========")

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    print("Vectors stored:", index.ntotal)

    return index


def save_vectorstore(index, chunks):
    index_path = VECTOR_DIR / "index.faiss"
    metadata_path = VECTOR_DIR / "chunks.json"

    faiss.write_index(
        index,
        str(index_path)
    )

    chunk_data = []

    for chunk in chunks:
        chunk_data.append({
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
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
            ensure_ascii=False,
        )

    print("\nVector store saved.")
    print("FAISS:", index_path)
    print("Metadata:", metadata_path)


def load_vectorstore():
    index_path = VECTOR_DIR / "index.faiss"
    metadata_path = VECTOR_DIR / "chunks.json"

    index = faiss.read_index(
        str(index_path)
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as f:
        chunk_data = json.load(f)

    chunks = []

    for item in chunk_data:
        chunks.append(
            Document(
                page_content=item["page_content"],
                metadata=item["metadata"],
            )
        )

    return index, chunks