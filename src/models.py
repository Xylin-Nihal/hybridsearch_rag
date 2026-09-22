import base64
import numpy as np

from huggingface_hub import InferenceClient
from google import genai
from google.genai import types

from src.config import (
    HF_TOKEN,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    VISION_MODEL,
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

class EmbeddingModel:

    def __init__(self):

        if not HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN is not configured."
            )

        self.model = EMBEDDING_MODEL

        self.client = InferenceClient(
            provider="hf-inference",
            api_key=HF_TOKEN,
        )

        print(
            f"Embedding API initialized: "
            f"{self.model}"
        )

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    def encode(self, texts):

        if isinstance(texts, str):

            texts = [texts]

        if not texts:

            return np.array(
                [],
                dtype=np.float32
            )

        embeddings = []

        for text in texts:

            result = self.client.feature_extraction(
                text,
                model=self.model,
            )

            # ------------------------------------------------
            # BGE-small returns an embedding vector.
            # ------------------------------------------------

            embedding = np.asarray(
                result,
                dtype=np.float32
            )

            # ------------------------------------------------
            # Some providers/models may return token-level
            # representations instead of one vector.
            #
            # If that happens, mean-pool them.
            # ------------------------------------------------

            if embedding.ndim == 2:

                embedding = embedding.mean(
                    axis=0
                )

            embeddings.append(
                embedding
            )

        embeddings = np.vstack(
            embeddings
        )

        # ----------------------------------------------------
        # Normalize embeddings
        # ----------------------------------------------------

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        norms[norms == 0] = 1

        embeddings = (
            embeddings / norms
        )

        return embeddings.astype(
            np.float32
        )

    # --------------------------------------------------------
    # Single embedding
    # --------------------------------------------------------

    def encode_single(self, text):

        return self.encode(
            [text]
        )[0]


# ============================================================
# VISION MODEL
# ============================================================

class VisionModel:

    def __init__(self):

        if not GEMINI_API_KEY:

            raise RuntimeError(
                "GEMINI_API_KEY is not configured."
            )

        self.model = VISION_MODEL

        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        print(
            f"Vision API initialized: "
            f"{self.model}"
        )

    # --------------------------------------------------------
    # Describe image/table
    # --------------------------------------------------------

    def describe(
        self,
        image_base64,
        visual_type="image"
    ):

        # ====================================================
        # Prompt for table
        # ====================================================

        if visual_type == "table":

            prompt = """
Analyze this table from a PDF.

Extract the information accurately.

Preserve:

- table title if visible
- column names
- row names
- values
- units
- relationships between values

Convert the table into clear structured text
that can be used for retrieval in a RAG system.

Do not invent information that is not visible.
"""

        # ====================================================
        # Prompt for image
        # ====================================================

        else:

            prompt = """
Analyze this image or diagram from a PDF.

Describe the important information accurately.

Include:

- what the image represents
- important labels
- important components
- relationships between components
- technical concepts
- visible text relevant to the concept

The output will be stored as text inside
a RAG system.

Do not invent information that is not visible.
"""

        # ====================================================
        # Decode base64 image
        # ====================================================

        image_bytes = base64.b64decode(
            image_base64
        )

        # ====================================================
        # Gemini request
        # ====================================================

        response = self.client.models.generate_content(

            model=self.model,

            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),

                prompt,
            ],
        )

        # ====================================================
        # Extract response
        # ====================================================

        if not response.text:

            return (
                "[Vision model returned no description]"
            )

        return response.text.strip()