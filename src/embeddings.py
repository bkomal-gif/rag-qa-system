"""
Embeddings and vector store: converts text chunks into vectors
and stores them in FAISS for fast similarity search.

This is the "memory" of our RAG system — it lets us find the most
relevant chunks for any given question.
"""

import os
import logging
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import Config

logger = logging.getLogger(__name__)


def get_embedding_model() -> OpenAIEmbeddings:
    """
    Initialize the embedding model.
    Uses OpenAI's text-embedding-3-small by default (~$0.02 per 1M tokens).
    """
    Config.validate()

    embeddings = OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        openai_api_key=Config.OPENAI_API_KEY,
    )

    logger.info(f"Initialized embedding model: {Config.EMBEDDING_MODEL}")
    return embeddings


def create_vector_store(
    chunks: List[Document],
    save_path: str = None,
) -> FAISS:
    """
    Create a FAISS vector store from document chunks.

    Steps:
    1. Generate embeddings for each chunk using OpenAI
    2. Build a FAISS index for fast similarity search
    3. Save the index to disk for later use

    Args:
        chunks: List of Document objects (from text_splitter)
        save_path: Directory to save the FAISS index

    Returns:
        FAISS vector store object
    """
    save_path = save_path or Config.INDEX_DIR
    embeddings = get_embedding_model()

    logger.info(f"Creating vector store from {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save to disk so we don't have to re-embed every time
    os.makedirs(save_path, exist_ok=True)
    vector_store.save_local(save_path)
    logger.info(f"Vector store saved to {save_path}")

    return vector_store


def load_vector_store(load_path: str = None) -> FAISS:
    """
    Load a previously saved FAISS vector store from disk.

    Args:
        load_path: Directory containing the saved FAISS index

    Returns:
        FAISS vector store object
    """
    load_path = load_path or Config.INDEX_DIR

    if not os.path.exists(load_path):
        raise FileNotFoundError(
            f"No vector store found at {load_path}. Run ingest.py first."
        )

    embeddings = get_embedding_model()
    vector_store = FAISS.load_local(
        load_path, embeddings, allow_dangerous_deserialization=True
    )

    logger.info(f"Loaded vector store from {load_path}")
    return vector_store


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Quick test with dummy chunks
    test_chunks = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "test.pdf", "page": 0},
        ),
        Document(
            page_content="RAG combines retrieval with generation for better answers.",
            metadata={"source": "test.pdf", "page": 1},
        ),
    ]

    vs = create_vector_store(test_chunks, save_path="/tmp/test_faiss")
    print(f"Vector store created with {vs.index.ntotal} vectors")

    # Test search
    results = vs.similarity_search("What is machine learning?", k=1)
    print(f"Search result: {results[0].page_content}")
