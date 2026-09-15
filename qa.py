from typing import Dict

from retrieval import retrieve_documents
from llm import generate_answer


def answer_question(
    question: str,
    candidate_k: int = 10,
    final_k: int = 3,
) -> Dict:
    """
    Complete document question-answering pipeline.

    Pipeline:
    1. Hybrid retrieval
    2. Cross-encoder reranking
    3. Grounded local LLM generation
    4. Source + evidence information
    """

    question = question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    # ----------------------------------------------
    # Retrieval + Reranking
    # ----------------------------------------------

    documents = retrieve_documents(
        question,
        candidate_k=candidate_k,
        final_k=final_k,
    )

    # ----------------------------------------------
    # Generate grounded answer
    # ----------------------------------------------

    answer = generate_answer(
        question,
        documents,
    )

    # ----------------------------------------------
    # Prepare sources and evidence
    # ----------------------------------------------

    sources = []

    for document in documents:

        source = document["source"]
        page = document["page"]

        sources.append(
            {
                "source": source,
                "page": page,
                "chunk_id": document["chunk_id"],
                "text": document["text"],
                "reranker_score": document.get(
                    "reranker_score"
                ),
                "hybrid_score": document.get(
                    "hybrid_score"
                ),
            }
        )

    # ----------------------------------------------
    # If the LLM refused to answer, don't present
    # irrelevant retrieved chunks as supporting
    # sources.
    # ----------------------------------------------

    refusal_message = (
        "I couldn't find enough information in "
        "the uploaded documents."
    )

    if answer.strip() == refusal_message:
        sources = []

    return {
        "answer": answer,
        "sources": sources,
    }