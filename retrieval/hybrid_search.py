import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import os
from config import FAISS_INDEX_PATH, FAISS_META_PATH, EMBEDDING_MODEL, TOP_K_RESULTS

DB_INDEX = FAISS_INDEX_PATH
DB_META = FAISS_META_PATH

model = SentenceTransformer(EMBEDDING_MODEL)


def load_faiss_db():
    """Load FAISS index and metadata."""
    if not os.path.exists(DB_INDEX) or not os.path.exists(DB_META):
        raise FileNotFoundError(
            "FAISS index not found. Please run embeddings/embed_chunks.py first."
        )
    
    try:
        index = faiss.read_index(DB_INDEX)
        with open(DB_META, "rb") as f:
            meta = pickle.load(f)
        return index, meta
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS database: {str(e)}")


def semantic_search(query, k=TOP_K_RESULTS):
    index, meta = load_faiss_db()

    query_vec = model.encode([query], convert_to_numpy=True)

    distances, positions = index.search(query_vec, k)

    results = []
    for pos in positions[0]:
        doc_id = meta["ids"][pos]
        text = meta["texts"][pos]
        results.append((doc_id, text))

    return results

def retrieve_context(query, k=TOP_K_RESULTS):
    """Retrieve relevant context for a query."""
    matches = semantic_search(query, k)
    # Join all chunks without truncation for better context
    context = "\n\n".join(text for _, text in matches)
    return {
        "context": context,
        "sources": matches
    }

if __name__ == "__main__":
    q = "what topics does this document talk about?"
    matches = semantic_search(q)
    print("\n--- Results ---\n")
    for mid, text in matches:
        print(mid)
        print(text[:150], "...")
        print()
