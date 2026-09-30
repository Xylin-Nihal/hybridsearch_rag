import base64
import numpy as np
import torch
from huggingface_hub import InferenceClient
from google import genai
from google.genai import types
from sentence_transformers import CrossEncoder
from src.config import (
    HF_TOKEN,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    VISION_MODEL,
)
from groq import Groq

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
class RerankerModel:

    def __init__(self):

        from src.config import RERANKER_MODEL

        self.model_name = RERANKER_MODEL

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Loading reranker: "
            f"{self.model_name}"
        )

        print(
            f"Reranker device: {device}"
        )

        self.model = CrossEncoder(
            self.model_name,
            max_length=512,
            device=device
        )

        print(
            "Reranker initialized."
        )

    def rerank(
        self,
        query,
        candidates,
        top_k=5
    ):

        if not candidates:
            return []

        pairs = []

        for candidate in candidates:

            chunk = candidate["chunk"]

            section_title = chunk.get(
                "section_title",
                ""
            )

            section_path = chunk.get(
                "section_path",
                []
            )

            chunk_type = chunk.get(
                "chunk_type",
                "text"
            )

            content = chunk.get(
                "content",
                ""
            )

            # Give the reranker structural
            # information along with content.
            document_text = (
                f"Section: {section_title}\n"
                f"Section path: "
                f"{' > '.join(section_path)}\n"
                f"Content type: {chunk_type}\n"
                f"Content:\n{content}"
            )

            pairs.append(
                (
                    query,
                    document_text
                )
            )

        scores = self.model.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False
        )

        reranked = []

        for candidate, score in zip(
            candidates,
            scores
        ):

            result = dict(candidate)

            result["rerank_score"] = float(
                score
            )

            reranked.append(result)

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]
class GroqModel:

    def __init__(self):

        from src.config import (
            GROQ_API_KEY,
            GROQ_MODEL
        )

        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not configured."
            )

        self.model_name = GROQ_MODEL

        self.client = Groq(
            api_key=GROQ_API_KEY
        )

        print(
            f"LLM initialized: "
            f"{self.model_name}"
        )


    def generate(
        self,
        query,
        context,
        history=None
    ):

        if history is None:
            history = []


        system_prompt = """
You are a helpful RAG assistant.

Your job is to answer the user's question
using the retrieved document context provided
to you.

IMPORTANT RULES:

1. Answer using the retrieved context whenever
   the question is about the document.

2. Do not invent facts that are not supported
   by the retrieved context.

3. If the retrieved context does not contain
   enough information to answer the question,
   clearly say that the information was not
   found in the document.

4. You may combine information from multiple
   retrieved chunks when necessary.

5. Prefer a clear and concise explanation.

6. Preserve technical terminology from the
   source document.

7. When useful, mention the relevant section
   name.

8. The retrieved visual elements may contain
   descriptions generated from images or tables.
   Treat those descriptions as document evidence,
   but do not claim to see visual details that
   are not present in the supplied description.

9. Do not mention internal retrieval mechanisms
   such as BM25, vector search, embeddings,
   reranking, or candidate pools unless the user
   explicitly asks about the system.

10. If the user asks a follow-up question,
    use the conversation history together with
    the retrieved context.
"""


        messages = [

            {
                "role": "system",
                "content": system_prompt
            }

        ]


        # ----------------------------------------------------
        # Conversation history
        # ----------------------------------------------------

        for message in history:

            messages.append({
                "role": message["role"],
                "content": message["content"]
            })


        # ----------------------------------------------------
        # Current RAG context
        # ----------------------------------------------------

        user_prompt = f"""
Retrieved document context:

{context}


User question:

{query}


Answer the question using the retrieved
document context.
"""

        messages.append({
            "role": "user",
            "content": user_prompt
        })


        # ----------------------------------------------------
        # Groq generation
        # ----------------------------------------------------

        response = self.client.chat.completions.create(

            model=self.model_name,

            messages=messages,

            temperature=0.2,

            max_tokens=2048

        )


        answer = (
            response
            .choices[0]
            .message
            .content
        )


        return answer.strip()