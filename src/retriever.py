"""
Retriever: finds the most relevant document chunks for a given question.

This is the "R" in RAG — the retrieval step that finds context
before the LLM generates an answer.
"""

import logging
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import Config

logger = logging.getLogger(__name__)


def retrieve_relevant_chunks(
    vector_store: FAISS,
    query: str,
    k: int = None,
) -> List[Document]:
    """
    Retrieve the top-k most relevant chunks for a given query.

    Uses cosine similarity between the query embedding and
    stored document embeddings.

    Args:
        vector_store: FAISS vector store with indexed documents
        query: User's question
        k: Number of chunks to retrieve (default: 4)

    Returns:
        List of most relevant Document objects
    """
    k = k or Config.RETRIEVER_K

    results = vector_store.similarity_search(query, k=k)

    logger.info(f"Retrieved {len(results)} chunks for query: '{query[:80]}...'")
    return results


def retrieve_with_scores(
    vector_store: FAISS,
    query: str,
    k: int = None,
) -> List[Tuple[Document, float]]:
    """
    Retrieve chunks with similarity scores (useful for debugging
    and confidence display).

    Args:
        vector_store: FAISS vector store
        query: User's question
        k: Number of chunks to retrieve

    Returns:
        List of (Document, score) tuples. Lower score = more similar.
    """
    k = k or Config.RETRIEVER_K

    results = vector_store.similarity_search_with_score(query, k=k)

    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        logger.debug(f"  Score: {score:.4f} | Source: {source} | Page: {page}")

    return results


def format_context(documents: List[Document]) -> str:
    """
    Format retrieved documents into a context string for the LLM.

    Includes source attribution so the LLM can reference where
    information came from.
    """
    context_parts = []

    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        source_name = source.split("/")[-1] if "/" in source else source

        context_parts.append(
            f"[Source {i}: {source_name}, Page {page}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)
