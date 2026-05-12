"""
Document loading: reads PDF files from the data directory.

This module handles the first step of the RAG pipeline — getting text
out of PDF documents into a format we can process.
"""

import os
import logging
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_single_pdf(file_path: str) -> List[Document]:
    """
    Load a single PDF file and return a list of Document objects.
    Each page becomes one Document with metadata (source, page number).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    logger.info(f"Loaded {len(pages)} pages from {os.path.basename(file_path)}")
    return pages


def load_all_pdfs(data_dir: str) -> List[Document]:
    """
    Load all PDF files from a directory.
    Returns a combined list of Document objects from all PDFs.
    """
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Data directory not found: {data_dir}")

    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    all_documents: List[Document] = []

    for pdf_file in sorted(pdf_files):
        file_path = os.path.join(data_dir, pdf_file)
        try:
            docs = load_single_pdf(file_path)
            all_documents.extend(docs)
        except Exception as e:
            logger.error(f"Failed to load {pdf_file}: {e}")
            continue

    logger.info(
        f"Total: loaded {len(all_documents)} pages from {len(pdf_files)} PDF(s)"
    )
    return all_documents


if __name__ == "__main__":
    # Quick test: run this file directly to check loading works
    from config import Config

    logging.basicConfig(level=logging.INFO)
    docs = load_all_pdfs(Config.DATA_DIR)
    for doc in docs[:3]:
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Page: {doc.metadata.get('page', '?')}")
        print(f"Content preview: {doc.page_content[:200]}...")
        print("---")
