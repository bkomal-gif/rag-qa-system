"""
LLM Chain: sends retrieved context + user question to the LLM
and generates an answer.

This is the "G" in RAG — the generation step. The key insight is
that we DON'T ask the LLM to answer from its own knowledge. Instead,
we provide specific context and ask it to answer ONLY from that context.
"""

import logging
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.config import Config

logger = logging.getLogger(__name__)

# ─── PROMPT TEMPLATE ───
# This is where prompt engineering matters most.
# The prompt instructs the LLM to:
# 1. Answer ONLY from the provided context
# 2. Say "I don't know" if the answer isn't in the context
# 3. Cite sources for transparency

RAG_PROMPT_TEMPLATE = """You are a helpful research assistant. Answer the user's 
question based ONLY on the provided context. If the answer cannot be found in 
the context, say "I don't have enough information in the provided documents to 
answer this question."

Always cite which source(s) you used in your answer.

Context:
{context}

Question: {question}

Answer:"""


def get_llm() -> ChatOpenAI:
    """
    Initialize the LLM.
    Uses GPT-4o-mini by default (cheap, fast, good quality).
    """
    Config.validate()

    llm = ChatOpenAI(
        model=Config.LLM_MODEL,
        openai_api_key=Config.OPENAI_API_KEY,
        temperature=0.1,  # Low temperature for factual Q&A
        max_tokens=1000,
    )

    logger.info(f"Initialized LLM: {Config.LLM_MODEL}")
    return llm


def generate_answer(context: str, question: str) -> Dict[str, str]:
    """
    Generate an answer using the LLM with retrieved context.

    Args:
        context: Formatted context string from retriever
        question: User's question

    Returns:
        Dict with 'answer' and 'prompt' keys
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    chain = prompt | llm

    formatted_prompt = RAG_PROMPT_TEMPLATE.format(
        context=context, question=question
    )

    logger.info(f"Generating answer for: '{question[:80]}...'")
    response = chain.invoke({"context": context, "question": question})

    return {
        "answer": response.content,
        "prompt": formatted_prompt,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Quick test
    test_context = """[Source 1: ml_basics.pdf, Page 5]
    Machine learning is a subset of artificial intelligence that enables 
    systems to learn from data without being explicitly programmed.
    Supervised learning uses labeled data to train models."""

    test_question = "What is machine learning?"

    result = generate_answer(test_context, test_question)
    print(f"Answer: {result['answer']}")
