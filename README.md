# 🔍 RAG-Based Document Q&A System

A production-style Retrieval-Augmented Generation (RAG) pipeline that lets you chat with your PDF documents. Built with LangChain, FAISS, and OpenAI/Claude APIs, with a Streamlit frontend.

## 🎯 What This Project Demonstrates

- **RAG Pipeline Architecture**: Document ingestion → chunking → embedding → vector storage → retrieval → LLM generation
- **Vector Database**: FAISS for similarity search over document embeddings
- **LLM API Integration**: OpenAI (GPT-4o-mini) or Anthropic (Claude) API calls with structured prompts
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
[LLM + Prompt] ── GPT-4o-mini / Claude with context + question
     │
     ▼
[Streamlit UI] ── Chat interface with source display
```

## 📁 Project Structure

```
rag-qa-system/
├── data/                   # Place your PDF files here
│   └── sample.pdf
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration and environment variables
│   ├── document_loader.py  # PDF loading and text extraction
│   ├── text_splitter.py    # Chunking strategies
│   ├── embeddings.py       # Embedding generation and vector store
│   ├── retriever.py        # Similarity search and retrieval
│   ├── llm_chain.py        # LLM integration and prompt templates
│   └── pipeline.py         # End-to-end RAG pipeline
├── notebooks/
│   └── 01_exploration.ipynb  # Step-by-step walkthrough
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
git clone https://github.com/YOUR_USERNAME/rag-qa-system.git
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

## 🔑 API Keys

You need ONE of these:
- **OpenAI API key**: Get from https://platform.openai.com/api-keys (~$0.50 to run this entire project)
- **Anthropic API key**: Get from https://console.anthropic.com/ (alternative)

## 📚 Learning Roadmap

### Week 1: Core Pipeline
- [ ] Day 1-2: Read about RAG architecture, set up environment, get API key
- [ ] Day 3: Build document loader and text splitter
- [ ] Day 4: Generate embeddings and build FAISS index
- [ ] Day 5-7: Build retriever + LLM chain, get basic Q&A working

### Week 2: Polish and Deploy
- [ ] Day 8-9: Build Streamlit chat UI
- [ ] Day 10: Add source attribution and confidence display
- [ ] Day 11-12: Test with different documents, tune chunk size and k
- [ ] Day 13-14: Write documentation, push to GitHub, deploy Streamlit Cloud

## 🧪 Experiments to Try

Once the basic pipeline works, try these to deepen your understanding:

1. **Chunk size ablation**: Try 500, 1000, 1500 — how does retrieval quality change?
2. **Different embedding models**: Compare OpenAI vs. free alternatives (sentence-transformers)
3. **Reranking**: Add a reranker after retrieval to improve relevance
4. **Multi-document**: Index 10+ papers and test cross-document questions
5. **Evaluation**: Build a small eval set (20 questions with expected answers) and measure accuracy

## 📝 License

MIT
