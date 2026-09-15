from pathlib import Path
from typing import List, Dict

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    COLLECTION_NAME,
)


# Project directories
BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"

DOCUMENTS_DIR.mkdir(exist_ok=True)


# Lazy-loaded models and database
_embedding_model = None
_chroma_client = None
_collection = None


def get_embedding_model():
    """
    Load the embedding model only when needed.
    """

    global _embedding_model

    if _embedding_model is None:
        print("Loading embedding model...")

        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    return _embedding_model


def get_collection():
    """
    Get or create the ChromaDB collection.
    """

    global _chroma_client
    global _collection

    if _collection is None:

        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    return _collection


def load_pdf(file_path: str) -> List[Dict]:
    """
    Extract text from a PDF page by page.
    """

    documents = []

    reader = PdfReader(file_path)

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = page.extract_text()

        if not text:
            continue

        text = text.strip()

        if not text:
            continue

        documents.append(
            {
                "text": text,
                "source": Path(file_path).name,
                "page": page_number,
            }
        )

    return documents


def load_text_file(file_path: str) -> List[Dict]:
    """
    Load TXT or Markdown files.
    """

    path = Path(file_path)

    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).strip()

    if not text:
        return []

    return [
        {
            "text": text,
            "source": path.name,
            "page": None,
        }
    ]


def load_document(file_path: str) -> List[Dict]:
    """
    Automatically select the correct document loader.
    """

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    if extension in {".txt", ".md"}:
        return load_text_file(file_path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks.
    """

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length,
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def process_document(file_path: str) -> List[Dict]:
    """
    Load a document and convert it into chunks
    while preserving metadata.
    """

    pages = load_document(file_path)

    chunks = []

    chunk_id = 0

    for page in pages:

        page_chunks = chunk_text(
            page["text"]
        )

        for chunk in page_chunks:

            chunks.append(
                {
                    "text": chunk,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_id": chunk_id,
                }
            )

            chunk_id += 1

    return chunks


def add_document_to_collection(
    file_path: str,
) -> int:
    """
    Process a document, generate embeddings,
    and store chunks in ChromaDB.
    """

    collection = get_collection()
    model = get_embedding_model()

    chunks = process_document(file_path)

    if not chunks:
        return 0

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    ).tolist()

    ids = []

    metadatas = []

    for chunk in chunks:

        chunk_id = (
            f"{chunk['source']}_"
            f"{chunk['chunk_id']}"
        )

        ids.append(chunk_id)

        metadatas.append(
            {
                "source": chunk["source"],
                "page": (
                    chunk["page"]
                    if chunk["page"] is not None
                    else -1
                ),
                "chunk_id": chunk["chunk_id"],
            }
        )

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)


def search_documents(
    question: str,
    top_k: int = 5,
) -> List[Dict]:
    """
    Perform semantic search against ChromaDB.
    """

    collection = get_collection()
    model = get_embedding_model()

    if collection.count() == 0:
        return []

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    output = []

    for text, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        output.append(
            {
                "text": text,
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk_id": metadata["chunk_id"],
                "distance": distance,
            }
        )

    return output


def get_collection_count() -> int:
    """
    Return the number of chunks stored in ChromaDB.
    """

    return get_collection().count()


def clear_collection():
    """
    Delete all indexed chunks.
    """

    global _collection

    client = get_collection()

    ids = client.get()["ids"]

    if ids:
        client.delete(ids=ids)

    _collection = None