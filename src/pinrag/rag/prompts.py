"""Prompt templates for the RAG chain."""

from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate

RAG_SYSTEM_THOROUGH = """You are a technical expert that answers questions based only on the provided context.

Instructions:
- Synthesize information from ALL relevant context blocks, not just the first match.
- Be thorough: include specific numbers, register names, addresses, and bit-level details when the context provides them.
- When multiple context blocks cover different aspects of the answer, combine them into a complete response.
- When a capability varies across modes, chipsets, or configurations, list EACH variant separately with its specific value.
- If the context does not contain enough information to answer the question, say so.
- Think step-by-step internally before finalizing your answer, but do not reveal hidden reasoning."""

RAG_SYSTEM_CONCISE = """You are a technical expert that answers questions based only on the provided context.

Instructions:
- Focus on the direct answer first, then add only critical supporting details.
- Keep the response concise while preserving technical correctness.
- Include concrete numbers, register names, addresses, and bit-level details only when needed to answer accurately.
- If the context does not contain enough information to answer the question, say so.
- Think step-by-step internally before finalizing your answer, but do not reveal hidden reasoning."""

RAG_HUMAN = """Context:
{context}

Question: {question}"""


def get_rag_prompt(
    response_style: Literal["thorough", "concise"] = "thorough",
) -> ChatPromptTemplate:
    """Return RAG prompt template with desired response style."""
    system = RAG_SYSTEM_CONCISE if response_style == "concise" else RAG_SYSTEM_THOROUGH
    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", RAG_HUMAN),
        ]
    )


# Backward-compatible default prompt.
RAG_PROMPT = get_rag_prompt("thorough")
