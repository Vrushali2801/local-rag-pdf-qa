import streamlit as st
st.write("DEBUG: UI file is running")
import os
import sys

# add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.write("Testing imports...")

try:
    from ingestion.extract_text import extract_text_from_pdf
    st.write("import 1 OK")
except Exception as e:
    st.write("import 1 FAILED:", e)

try:
    from chunking.chunk_text import chunk_text
    st.write("import 2 OK")
except Exception as e:
    st.write("import 2 FAILED:", e)

try:
    from embeddings.embed_chunks import embed_and_store
    st.write("import 3 OK")
except Exception as e:
    st.write("import 3 FAILED:", e)

try:
    from retrieval.hybrid_search import retrieve_context
    st.write("import 4 OK")
except Exception as e:
    st.write("import 4 FAILED:", e)

try:
    from llm.generate_answer import generate_answer
    st.write("import 5 OK")
except Exception as e:
    st.write("import 5 FAILED:", e)



UPLOAD_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


# --- Helper functions ---------------------------------------------------------

def process_pdf(uploaded_file):
    raw_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(raw_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text
    text = extract_text_from_pdf(raw_path)

    processed_path = os.path.join(PROCESSED_DIR, uploaded_file.name.replace(".pdf", ".txt"))
    with open(processed_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Chunk
    chunk_text(processed_path)

    # Re-embed into FAISS
    embed_and_store()


# --- UI Starts Here -----------------------------------------------------------

st.set_page_config(page_title="Local RAG App", layout="wide")

st.title("📄 Local CPU RAG System")
st.write("Upload your PDFs and ask questions using your local LLM.")

# --- PDF Upload Section --------------------------------------------------------

st.header("Upload Documents")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        process_pdf(uploaded_file)
    st.success("Document processed and indexed!")

st.divider()

# --- Q&A Section ---------------------------------------------------------------

st.header("Ask Your Documents")

query = st.text_input("Enter your question:")

if st.button("Search"):
    if not query.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Retrieving context..."):
            ret = retrieve_context(query)
        
        st.subheader("Retrieved Context")
        st.write(ret["context"])

        with st.spinner("Generating answer using Ollama..."):
            answer = generate_answer(ret["context"], query, model="phi3")

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for sid, _ in ret["sources"]:
            st.write(f"- {sid}")
