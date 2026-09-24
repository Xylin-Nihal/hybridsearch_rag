import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# PDF CONFIGURATION
# ============================================================

PDF_PATH = "data/attention-is-all-you-need.pdf"


# ============================================================
# CHUNKING CONFIGURATION
# ============================================================

SMALL_SECTION_TOKENS = 800

TARGET_CHUNK_TOKENS = 400

MAX_CHUNK_TOKENS = 500

SEMANTIC_SIMILARITY_THRESHOLD = 0.75

CHUNK_OVERLAP_SENTENCES = 1


# ============================================================
# HUGGING FACE EMBEDDING CONFIGURATION
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5"
)


# ============================================================
# GEMINI VISION CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

VISION_MODEL = os.getenv(
    "VISION_MODEL",
    "gemini-2.5-flash"
)


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

def validate_config():

    required_variables = {
        "HF_TOKEN": HF_TOKEN,
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "VISION_MODEL": VISION_MODEL,
        "RERANKER_MODEL": RERANKER_MODEL    
    }

    missing = [
        name
        for name, value in required_variables.items()
        if not value
    ]

    if missing:

        raise RuntimeError(
            "Missing environment variables:\n"
            + "\n".join(
                f"- {name}"
                for name in missing
            )
        )