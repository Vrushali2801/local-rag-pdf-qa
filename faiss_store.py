import faiss
import numpy as np
import os
import pickle

DB_INDEX = "vectorstore/faiss.index"
DB_META = "vectorstore/faiss_meta.pkl"

def save_faiss(index, metadata):
    faiss.write_index(index, DB_INDEX)
    with open(DB_META, "wb") as f:
        pickle.dump(metadata, f)

def load_faiss():
    if not os.path.exists(DB_INDEX):
        return None, None
    index = faiss.read_index(DB_INDEX)
    with open(DB_META, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata
