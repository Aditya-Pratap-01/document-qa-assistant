from pathlib import Path


# Base project directory
BASE_DIR = Path(__file__).parent


# Storage directories
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"


# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

#LLM model
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Chunk configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# Vector database collection
COLLECTION_NAME = "document_qa"