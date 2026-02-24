import streamlit as st
import os
import sys

# add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.extract_text import extract_text_from_pdf
from chunking.chunk_text import chunk_text
from embeddings.embed_chunks import embed_and_store
from retrieval.hybrid_search import retrieve_context
from llm.generate_answer import generate_answer
from config import RAW_DIR, PROCESSED_DIR, CHUNK_DIR, DEFAULT_MODEL

UPLOAD_DIR = RAW_DIR


# --- Helper functions ---------------------------------------------------------

def process_pdf(uploaded_file):
    """Process uploaded PDF: extract text, chunk, and re-index."""
    try:
        raw_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract text
        st.write("📄 Extracting text from PDF...")
        text = extract_text_from_pdf(raw_path)

        processed_path = os.path.join(PROCESSED_DIR, uploaded_file.name.replace(".pdf", ".txt"))
        with open(processed_path, "w", encoding="utf-8") as f:
            f.write(text)

        # Chunk text and save
        st.write("✂️ Chunking text...")
        chunks = chunk_text(text)
        chunk_filename = uploaded_file.name.replace(".pdf", "_chunks.txt")
        chunk_path = os.path.join(CHUNK_DIR, chunk_filename)
        
        with open(chunk_path, "w", encoding="utf-8") as f:
            for i, c in enumerate(chunks):
                f.write(f"--- chunk {i} ---\n")
                f.write(c + "\n\n")

        # Re-embed into FAISS
        st.write("🔢 Generating embeddings (this may take 30-60 seconds)...")
        embed_and_store()
        
        return True
    except Exception as e:
        st.error(f"Failed to process PDF: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False


# --- UI Starts Here -----------------------------------------------------------

st.set_page_config(page_title="Local RAG App", layout="wide")

st.title("📄 Local RAG System")
st.write("Upload your PDFs and ask questions using your local LLM.")

# Initialize session state
if 'last_uploaded' not in st.session_state:
    st.session_state.last_uploaded = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# --- PDF Upload Section --------------------------------------------------------

st.header("Upload Documents")

st.info("💡 **Tip:** All uploaded PDFs are indexed together. Use 'Clear All' to start fresh with a single document.")

# Show currently indexed documents
if os.path.exists(CHUNK_DIR):
    chunk_files = [f for f in os.listdir(CHUNK_DIR) if f.endswith("_chunks.txt")]
    if chunk_files:
        with st.expander(f"📚 Currently Indexed Documents ({len(chunk_files)})"):
            for f in chunk_files:
                st.write(f"• {f.replace('_chunks.txt', '')}")

# Option to clear all documents
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🗑️ Clear All", help="Remove all indexed documents"):
        # Clear chunks
        if os.path.exists(CHUNK_DIR):
            for f in os.listdir(CHUNK_DIR):
                if f.endswith("_chunks.txt") or f.endswith(".jsonl") or f == "chunks.txt":
                    os.remove(os.path.join(CHUNK_DIR, f))
        # Clear vector store
        from config import FAISS_INDEX_PATH, FAISS_META_PATH
        if os.path.exists(FAISS_INDEX_PATH):
            os.remove(FAISS_INDEX_PATH)
        if os.path.exists(FAISS_META_PATH):
            os.remove(FAISS_META_PATH)
        # Reset session state
        st.session_state.last_uploaded = None
        st.session_state.processing_complete = False
        st.success("✅ All documents cleared!")
        st.info("Upload a new PDF to get started.")

with col1:
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_uploader")

# Only process if file is new
if uploaded_file and uploaded_file.name != st.session_state.last_uploaded:
    with st.spinner("Processing PDF..."):
        success = process_pdf(uploaded_file)
    if success:
        st.session_state.last_uploaded = uploaded_file.name
        st.session_state.processing_complete = True
        st.success("✅ Document processed and indexed!")
        st.info("👇 Scroll down to ask questions!")
    else:
        st.session_state.last_uploaded = None
        st.session_state.processing_complete = False
elif uploaded_file and uploaded_file.name == st.session_state.last_uploaded:
    st.info(f"✅ Document '{uploaded_file.name}' is already indexed. You can ask questions below!")

st.divider()

# --- Q&A Section ---------------------------------------------------------------

st.header("Ask Your Documents")

query = st.text_input("Enter your question:")

if st.button("Search"):
    if not query.strip():
        st.warning("Please type a question.")
    else:
        try:
            with st.spinner("Retrieving context..."):
                ret = retrieve_context(query)
            
            st.subheader("Retrieved Context")
            st.write(ret["context"])

            with st.spinner("Generating answer using Ollama..."):
                answer = generate_answer(ret["context"], query, model=DEFAULT_MODEL)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources")
            for sid, _ in ret["sources"]:
                st.write(f"- {sid}")
        except FileNotFoundError:
            st.error("⚠️ No documents indexed yet. Please upload a PDF first.")
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
            st.info("Make sure Ollama is running: `ollama serve`")
