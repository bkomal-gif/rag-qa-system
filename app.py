"""
Streamlit chat interface for the RAG Q&A system.

Run with: streamlit run app.py
"""

import streamlit as st
from src.pipeline import ask
from src.embeddings import load_vector_store


# ─── PAGE CONFIG ───
st.set_page_config(
    page_title="📚 Document Q&A",
    page_icon="🔍",
    layout="centered",
)

st.title("📚 RAG Document Q&A")
st.caption("Ask questions about your indexed documents. Answers are grounded in your PDFs, not general knowledge.")


# ─── CHECK VECTOR STORE EXISTS ───
@st.cache_resource
def check_index():
    """Check if documents have been indexed."""
    try:
        vs = load_vector_store()
        return True, vs.index.ntotal
    except FileNotFoundError:
        return False, 0


index_exists, num_vectors = check_index()

if not index_exists:
    st.error(
        "⚠️ No documents indexed yet. Run `python ingest.py` first to index your PDFs."
    )
    st.stop()

st.success(f"✅ {num_vectors} document chunks indexed and ready.")


# ─── SIDEBAR SETTINGS ───
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=4,
        help="More chunks = more context but potentially more noise.",
    )
    show_sources = st.checkbox("Show source documents", value=True)
    show_context = st.checkbox("Show raw context (debug)", value=False)

    st.divider()
    st.caption("Built with LangChain, FAISS, and OpenAI")


# ─── CHAT HISTORY ───
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources") and show_sources:
            with st.expander("📚 Sources"):
                for s in message["sources"]:
                    st.markdown(f"- **{s['file']}**, Page {s['page']}")


# ─── CHAT INPUT ───
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                result = ask(prompt, k=k)

                st.markdown(result["answer"])

                if show_sources and result["sources"]:
                    with st.expander("📚 Sources"):
                        for s in result["sources"]:
                            st.markdown(f"- **{s['file']}**, Page {s['page']}")

                if show_context:
                    with st.expander("🔧 Raw Context (Debug)"):
                        st.text(result["context"])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    }
                )

            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.caption("Make sure your .env file has a valid API key.")
