"""
CLI script to index PDF documents into the vector store.

Usage:
    python ingest.py                    # Index all PDFs in data/
    python ingest.py --data-dir ./docs  # Index PDFs from a custom directory
"""

import argparse
import logging

from src.pipeline import ingest_documents


def main():
    parser = argparse.ArgumentParser(description="Index PDF documents for RAG")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing PDF files (default: data/)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        num_chunks = ingest_documents(data_dir=args.data_dir)
        print(f"\n🎉 Successfully indexed {num_chunks} chunks!")
        print("You can now run: streamlit run app.py")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("Place your PDF files in the data/ directory and try again.")
    except ValueError as e:
        print(f"\n❌ {e}")
        print("Copy .env.example to .env and add your API key.")


if __name__ == "__main__":
    main()
