# 🔍 RAG-Based Document Q&A System

A modular Retrieval-Augmented Generation (RAG) pipeline that answers questions over PDF documents with page-level source citations. Built with LangChain, FAISS, and the OpenAI API, with a Streamlit chat interface. The current index covers 7 peer-reviewed papers on singular optics (254 chunks).

## 🎯 What This Project Demonstrates

- **RAG Pipeline Architecture**: Document ingestion → chunking → embedding → vector storage → retrieval → LLM generation
- **Vector Database**: FAISS for similarity search over document embeddings
- **LLM API Integration**: OpenAI (GPT-4o-mini) API calls with structured, grounded prompts
- **Prompt Engineering**: Context-aware prompting with source attribution
- **NLP Fundamentals**: Text chunking, embeddings, semantic search
- **Production Practices**: Error handling, modular code, environment management, logging
- **Deployment**: Interactive Streamlit chat interface

## 🏗️ Architecture

```
PDF Documents
     │
     ▼
[Document Loader] ── PyPDF / PyMuPDF
     │
     ▼
[Text Splitter] ── RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
     │
     ▼
[Embedding Model] ── OpenAI text-embedding-3-small
     │
     ▼
[Vector Store] ── FAISS (local, no server needed)
     │
     ▼
[Retriever] ── Top-k similarity search (k=4)
     │
     ▼
[LLM + Prompt] ── GPT-4o-mini with context + question
     │
     ▼
[Streamlit UI] ── Chat interface with source display
```

## 📁 Project Structure

```
rag-qa-system/
├── data/                   # Place your PDF files here
│   └── *.pdf               # Research papers to index
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration and environment variables
│   ├── document_loader.py  # PDF loading and text extraction
│   ├── text_splitter.py    # Chunking strategies
│   ├── embeddings.py       # Embedding generation and vector store
│   ├── retriever.py        # Similarity search and retrieval
│   ├── llm_chain.py        # LLM integration and prompt templates
│   └── pipeline.py         # End-to-end RAG pipeline
├── app.py                  # Streamlit chat interface
├── ingest.py               # CLI script to index documents
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/bkomal-gif/rag-qa-system.git
cd rag-qa-system
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Add documents

Place your PDF files in the `data/` folder.

### 4. Index documents

```bash
python ingest.py
```

### 5. Run the chat interface

```bash
streamlit run app.py
```

## 🔑 API Key

You need an **OpenAI API key** (https://platform.openai.com/api-keys). Add it to `.env` locally, or to **Settings → Secrets** on Streamlit Cloud.

## 🐞 Issues Found and Fixed

- **Off-by-one citations:** PyPDF stores 0-based page indices, so every citation pointed one page too early. Citations now use the PDF's own page labels, verified against the source documents.
- **Startup crash:** a missing API key raised an unhandled error. The app now shows a clear setup message instead.

## 🧪 Experiments to Try

Once the basic pipeline works, try these to deepen your understanding:

1. **Chunk size ablation**: Try 500, 1000, 1500 — how does retrieval quality change?
2. **Different embedding models**: Compare OpenAI vs. free alternatives (sentence-transformers)
3. **Reranking**: Add a reranker after retrieval to improve relevance
4. **Multi-document**: Index 10+ papers and test cross-document questions
5. **Evaluation**: Build a small eval set (20 questions with expected answers) and measure accuracy

## 📝 License

MIT
