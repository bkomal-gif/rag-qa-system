"""
Text splitting: breaks documents into chunks for embedding.

Why chunk? LLMs have context limits, and embedding models work better
on focused passages than entire pages. Chunking also enables precise
retrieval — we want to find the EXACT paragraph that answers a question,
not an entire 10-page section.
"""

import logging
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import Config

logger = logging.getLogger(__name__)


def split_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.

    Uses RecursiveCharacterTextSplitter which tries to split on:
    1. Paragraphs (\\n\\n)
    2. Sentences (. or \\n)
    3. Words (spaces)
    4. Characters (last resort)

    This preserves semantic coherence better than fixed-size splitting.

    Args:
        documents: List of Document objects from the loader
        chunk_size: Target size of each chunk in characters (default: 1000)
        chunk_overlap: Overlap between consecutive chunks (default: 200)

    Returns:
        List of smaller Document objects with preserved metadata
    """
    chunk_size = chunk_size or Config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    logger.info(
        f"Split {len(documents)} documents into {len(chunks)} chunks "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )

    return chunks


if __name__ == "__main__":
    # Quick test with dummy data
    logging.basicConfig(level=logging.INFO)

    test_docs = [
        Document(
            page_content="This is a long document. " * 200,
            metadata={"source": "test.pdf", "page": 0},
        )
    ]

    chunks = split_documents(test_docs, chunk_size=500, chunk_overlap=100)
    print(f"Input: 1 document, {len(test_docs[0].page_content)} chars")
    print(f"Output: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  Chunk {i}: {len(chunk.page_content)} chars")
