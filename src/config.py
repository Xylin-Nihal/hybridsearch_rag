from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "data" / "attention-is-all-you-need.pdf"

IMAGE_DIR = BASE_DIR / "extracted_images"
VECTOR_DIR = BASE_DIR / "vectorstore"

IMAGE_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

TOP_K = 5
NEIGHBOR_WINDOW = 1
FIXED_OVERLAP = 50