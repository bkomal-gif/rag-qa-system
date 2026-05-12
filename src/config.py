"""
Configuration management for the RAG pipeline.
Loads settings from .env file with sensible defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration for the RAG system."""

    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Models
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "4"))

    # Paths
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    INDEX_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "faiss_index"
    )

    @classmethod
    def validate(cls) -> bool:
        """Check that at least one API key is configured."""
        if not cls.OPENAI_API_KEY and not cls.ANTHROPIC_API_KEY:
            raise ValueError(
                "No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file."
            )
        return True
