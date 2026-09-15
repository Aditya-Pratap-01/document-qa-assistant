from typing import List, Dict
import re

from sentence_transformers import CrossEncoder

from rag import search_documents


# Lightweight Hugging Face reranker
RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


_reranker = None


def get_reranker():
    """
    Load the cross-encoder reranker lazily.
    """

    global _reranker

    if _reranker is None:
        print("Loading reranker...")

        _reranker = CrossEncoder(
            RERANKER_MODEL_NAME
        )

    return _reranker


def tokenize(text: str) -> List[str]:
    """
    Convert text into lowercase word tokens.
    """

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower(),
    )


def keyword_score(
    question: str,
    document: str,
) -> float:
    """
    Calculate keyword overlap score.
    """

    question_tokens = set(
        tokenize(question)
    )

    document_tokens = set(
        tokenize(document)
    )

    if not question_tokens:
        return 0.0

    overlap = question_tokens.intersection(
        document_tokens
    )

    return len(overlap) / len(question_tokens)


def normalize_semantic_score(
    distance: float,
) -> float:
    """
    Convert Chroma distance into a similarity-like
    score between 0 and 1.
    """

    return 1 / (1 + max(distance, 0))


def hybrid_search(
    question: str,
    top_k: int = 5,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> List[Dict]:
    """
    Perform semantic + keyword hybrid retrieval.
    """

    if not question.strip():
        return []

    if abs(
        semantic_weight
        + keyword_weight
        - 1.0
    ) > 1e-6:

        raise ValueError(
            "semantic_weight + keyword_weight "
            "must equal 1.0"
        )

    candidates = search_documents(
        question,
        top_k=max(top_k * 2, 10),
    )

    if not candidates:
        return []

    results = []

    for candidate in candidates:

        semantic_score = (
            normalize_semantic_score(
                candidate["distance"]
            )
        )

        keyword_match = keyword_score(
            question,
            candidate["text"],
        )

        hybrid_score = (
            semantic_weight * semantic_score
            + keyword_weight * keyword_match
        )

        result = candidate.copy()

        result["semantic_score"] = semantic_score
        result["keyword_score"] = keyword_match
        result["hybrid_score"] = hybrid_score

        results.append(result)

    results.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )

    return results[:top_k]


def rerank_documents(
    question: str,
    documents: List[Dict],
    top_k: int = 3,
) -> List[Dict]:
    """
    Rerank retrieved documents using a
    Hugging Face Cross-Encoder.
    """

    if not documents:
        return []

    reranker = get_reranker()

    pairs = [
        (
            question,
            document["text"],
        )
        for document in documents
    ]

    scores = reranker.predict(pairs)

    reranked = []

    for document, score in zip(
        documents,
        scores,
    ):
        result = document.copy()

        result["reranker_score"] = float(
            score
        )

        reranked.append(result)

    reranked.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    return reranked[:top_k]


def retrieve_documents(
    question: str,
    candidate_k: int = 10,
    final_k: int = 3,
) -> List[Dict]:
    """
    Complete retrieval pipeline:

    1. Hybrid retrieval
    2. Cross-encoder reranking
    3. Return best chunks
    """

    candidates = hybrid_search(
        question,
        top_k=candidate_k,
    )

    if not candidates:
        return []

    return rerank_documents(
        question,
        candidates,
        top_k=final_k,
    )