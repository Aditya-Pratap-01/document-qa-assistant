from typing import List, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from config import LLM_MODEL_NAME


_tokenizer = None
_model = None


def get_llm():
    """
    Load the Hugging Face instruction-tuned LLM lazily.
    """

    global _tokenizer
    global _model

    if _tokenizer is None or _model is None:

        print("Loading LLM...")

        _tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME
        )

        _model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            dtype=torch.float32,
        )

        _model.eval()

    return _tokenizer, _model


def build_prompt(
    question: str,
    documents: List[Dict],
) -> str:
    """
    Build a grounded prompt using retrieved document chunks.
    """

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        source = document["source"]
        page = document["page"]
        text = document["text"]

        if page is not None and page != -1:
            source_info = (
                f"{source}, Page {page}"
            )
        else:
            source_info = source

        context_parts.append(
            f"[Source {index}: {source_info}]\n"
            f"{text}"
        )

    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a document question-answering assistant. "
                "Answer ONLY using the provided document context. "
                "Do not use outside knowledge. "
                "Do not invent or assume facts. "
                "If the answer is not present in the context, "
                "reply exactly: "
                "\"I couldn't find enough information in the uploaded documents.\" "
                "Give a concise and factual answer. "
                "Do not repeat the document context or the instructions."
            ),
        },
        {
            "role": "user",
            "content": (
                f"DOCUMENT CONTEXT:\n\n"
                f"{context}\n\n"
                f"QUESTION:\n\n"
                f"{question}"
            ),
        },
    ]

    return _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_answer(
    question: str,
    documents: List[Dict],
    max_new_tokens: int = 150,
) -> str:
    """
    Generate a grounded answer from retrieved document chunks.
    """

    if not documents:
        return (
            "I couldn't find enough information in "
            "the uploaded documents."
        )

    tokenizer, model = get_llm()

    prompt = build_prompt(
        question,
        documents,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return answer

