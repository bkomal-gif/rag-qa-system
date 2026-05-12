"""
End-to-end RAG pipeline: ties all components together.

This is the main module you'll interact with. It provides two
high-level functions:
1. ingest_documents() — process PDFs and build the vector store
2. ask() — answer a question using the RAG pipeline
"""

import logging
from typing import Dict, List

from src.config import Config
from src.document_loader import load_all_pdfs
from src.text_splitter import split_documents
from src.embeddings import create_vector_store, load_vector_store
from src.retriever import retrieve_relevant_chunks, format_context
from src.llm_chain import generate_answer

logger = logging.getLogger(__name__)


def ingest_documents(data_dir: str = None) -> int:
    """
    Full ingestion pipeline: load PDFs → split → embed → store.

    Args:
        data_dir: Directory containing PDF files

    Returns:
        Number of chunks indexed
    """
    data_dir = data_dir or Config.DATA_DIR

    # Step 1: Load PDFs
    print("📄 Loading PDF documents...")
    documents = load_all_pdfs(data_dir)

    # Step 2: Split into chunks
    print("✂️  Splitting into chunks...")
    chunks = split_documents(documents)

    # Step 3: Create embeddings and vector store
    print("🧮 Creating embeddings and vector store...")
    vector_store = create_vector_store(chunks)

    print(f"✅ Done! Indexed {len(chunks)} chunks from {len(documents)} pages.")
    return len(chunks)


def ask(question: str, k: int = None) -> Dict:
    """
    Answer a question using the full RAG pipeline.

    Args:
        question: User's question
        k: Number of chunks to retrieve

    Returns:
        Dict with 'answer', 'sources', and 'context' keys
    """
    # Step 1: Load the vector store
    vector_store = load_vector_store()

    # Step 2: Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(vector_store, question, k=k)

    # Step 3: Format context
    context = format_context(relevant_chunks)

    # Step 4: Generate answer
    result = generate_answer(context, question)

    # Step 5: Extract source info
    sources = []
    for doc in relevant_chunks:
        source = doc.metadata.get("source", "unknown").split("/")[-1]
        page = doc.metadata.get("page", "?")
        sources.append({"file": source, "page": page})

    return {
        "answer": result["answer"],
        "sources": sources,
        "context": context,
        "question": question,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Interactive Q&A loop for testing
    print("🔍 RAG Q&A System")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        try:
            result = ask(question)
            print(f"\n📝 Answer: {result['answer']}")
            print(f"\n📚 Sources:")
            for s in result["sources"]:
                print(f"   - {s['file']}, Page {s['page']}")
            print()
        except Exception as e:
            print(f"❌ Error: {e}\n")
