import numpy as np
import torch

from langchain_huggingface import HuggingFaceEmbeddings

from .config import EMBEDDING_MODEL


def load_embedding_model():
    print("\n========== EMBEDDING MODEL ==========")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": device
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    return embeddings


def create_embeddings(chunks, embedding_model):
    print("\n========== CREATING EMBEDDINGS ==========")

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    vectors = embedding_model.embed_documents(texts)

    vectors = np.array(
        vectors,
        dtype="float32"
    )

    print("Embedding shape:", vectors.shape)

    return vectors