import os
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import faiss

CHUNK_DIR = "data/chunks"
DB_INDEX = "vectorstore/faiss.index"
DB_META = "vectorstore/faiss_meta.pkl"


def load_chunks():
    texts = []
    ids = []
    for filename in os.listdir(CHUNK_DIR):
        if filename.endswith("_chunks.txt"):
            with open(os.path.join(CHUNK_DIR, filename), "r", encoding="utf-8") as f:
                content = f.read()
            pieces = content.split("--- chunk ")
            base = filename.replace("_chunks.txt", "")
            for p in pieces[1:]:
                header, *body = p.split("---", 1)
                idx = header.strip()
                text = "".join(body).strip()
                ids.append(f"{base}_{idx}")
                texts.append(text)
    return ids, texts


def embed_and_store():
    # REMOVE OLD INDEX SO NEW FILE TAKES EFFECT
    if os.path.exists(DB_INDEX):
        os.remove(DB_INDEX)
    if os.path.exists(DB_META):
        os.remove(DB_META)

    print("Loading model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Loading chunks...")
    ids, texts = load_chunks()

    print("Embedding...")
    embeddings = model.encode(texts, convert_to_numpy=True)

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    metadata = {"ids": ids, "texts": texts}

    print("Saving FAISS index...")
    faiss.write_index(index, DB_INDEX)
    with open(DB_META, "wb") as f:
        pickle.dump(metadata, f)

    print("Done.")


if __name__ == "__main__":
    embed_and_store()
