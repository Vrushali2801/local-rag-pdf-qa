import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import os

DB_INDEX = "vectorstore/faiss.index"
DB_META = "vectorstore/faiss_meta.pkl"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def load_faiss_db():
    index = faiss.read_index(DB_INDEX)
    with open(DB_META, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def semantic_search(query, k=3):
    index, meta = load_faiss_db()

    query_vec = model.encode([query], convert_to_numpy=True)

    distances, positions = index.search(query_vec, k)

    results = []
    for pos in positions[0]:
        doc_id = meta["ids"][pos]
        text = meta["texts"][pos]
        results.append((doc_id, text))

    return results

def retrieve_context(query, k=3):
    matches = semantic_search(query, k)
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
