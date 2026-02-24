"""
Configuration file for Local RAG system.
Centralizes all paths, parameters, and settings.
"""

import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directories
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CHUNK_DIR = os.path.join(PROJECT_ROOT, "data", "chunks")

# Vector store paths
VECTORSTORE_DIR = os.path.join(PROJECT_ROOT, "vectorstore")
FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss.index")
FAISS_META_PATH = os.path.join(VECTORSTORE_DIR, "faiss_meta.pkl")

# LLM settings
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "phi3"  # phi3 is smaller (2.2GB) and faster than llama3 (4.7GB)
LLM_TIMEOUT = 180  # seconds (3 minutes for slower machines)

# Prompt template path
PROMPT_TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "llm", "templates", "qa_prompt.jinja")

# Embedding settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking settings
CHUNK_SIZE = 1000  # characters per chunk (increased for better context)
CHUNK_OVERLAP = 200  # characters overlap

# Retrieval settings
TOP_K_RESULTS = 3  # number of chunks to retrieve

# OCR settings
OCR_DPI = 200

# Ensure directories exist
for directory in [RAW_DIR, PROCESSED_DIR, CHUNK_DIR, VECTORSTORE_DIR]:
    os.makedirs(directory, exist_ok=True)
