# Local RAG PDF Question Answering System

An end-to-end Retrieval-Augmented Generation (RAG) pipeline for querying PDF documents using semantic search and a locally hosted LLM.

## Overview
This project implements a complete RAG workflow that allows users to upload PDF documents, retrieve relevant content using vector similarity search, and generate answers grounded in the retrieved context.

The system is designed to be modular, making it easy to swap components such as the embedding model, vector store, or language model.

## Pipeline Architecture
1. **Document Ingestion** – Extracts text from PDF files (with OCR support)
2. **Text Chunking** – Splits documents into manageable chunks with overlap
3. **Embedding Generation** – Converts text chunks into dense vector representations
4. **Vector Retrieval** – Uses FAISS to perform semantic similarity search
5. **Answer Generation** – Feeds retrieved context to a local LLM for question answering

## Features
- 📄 PDF text extraction with OCR fallback for scanned documents
- 🔍 Semantic search using SentenceTransformers and FAISS
- 🤖 Local LLM integration via Ollama (no API keys required)
- 🎨 Interactive Streamlit UI
- 💻 Command-line interface for batch processing
- ⚙️ Centralized configuration for easy customization

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Tesseract OCR (for scanned PDFs)

### Installing Ollama

**Windows/Mac/Linux:**
```bash
# Visit https://ollama.ai/ and follow installation instructions
# Then pull the model:
ollama pull phi3
```

**Start Ollama server:**
```bash
ollama serve
```

### Installing Tesseract OCR

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

**Mac:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

## Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd local_rag
```

2. **Create and activate virtual environment:**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Streamlit UI (Recommended)

1. **Run the Streamlit app:**
```bash
streamlit run app/ui.py
```

2. **Upload a PDF** through the web interface
3. **Ask questions** about your documents

### Option 2: Command Line

1. **Place PDF files** in the `data/raw/` directory

2. **Extract text from PDFs:**
```bash
python ingestion/extract_text.py
```

3. **Chunk the text:**
```bash
python chunking/chunk_text.py
```

4. **Generate embeddings and build FAISS index:**
```bash
python embeddings/embed_chunks.py
```

5. **Ask questions:**
```bash
python main.py
```

## Project Structure

```
local_rag/
├── app/
│   └── ui.py                 # Streamlit web interface
├── chunking/
│   └── chunk_text.py         # Text chunking logic
├── data/
│   ├── raw/                  # Place PDF files here
│   ├── processed/            # Extracted text files
│   └── chunks/               # Chunked text with metadata
├── embeddings/
│   └── embed_chunks.py       # Generate embeddings and FAISS index
├── ingestion/
│   └── extract_text.py       # PDF text extraction with OCR
├── llm/
│   ├── generate_answer.py    # Ollama integration
│   └── templates/
│       └── qa_prompt.jinja   # Prompt template
├── retrieval/
│   └── hybrid_search.py      # FAISS semantic search
├── vectorstore/
│   ├── faiss.index           # FAISS index (generated)
│   └── faiss_meta.pkl        # Metadata (generated)
├── config.py                 # Centralized configuration
├── main.py                   # CLI entry point
├── requirements.txt          # Python dependencies
├── docker-compose.yml        # OpenSearch (optional)
└── README.md                 # This file
```

## Configuration

Customize settings in [config.py](config.py):

- **Chunk settings:** `CHUNK_SIZE`, `CHUNK_OVERLAP`
- **Embedding model:** `EMBEDDING_MODEL`
- **LLM model:** `DEFAULT_MODEL`
- **Retrieval:** `TOP_K_RESULTS`
- **Paths:** Data directories and vector store locations

## How It Works

1. **PDF Processing:** PDFs are converted to text using PyMuPDF, with Tesseract OCR as fallback for scanned pages
2. **Chunking:** Text is split into overlapping chunks to preserve context
3. **Embedding:** SentenceTransformers creates vector representations of each chunk
4. **Indexing:** FAISS stores embeddings for fast similarity search
5. **Retrieval:** User queries are embedded and matched against the index
6. **Generation:** Retrieved context is fed to Ollama to generate accurate answers

## Troubleshooting

### "Cannot connect to Ollama"
- Ensure Ollama is running: `ollama serve`
- Check if the model is installed: `ollama list`
- Pull the model if needed: `ollama pull phi3`

### "FAISS index not found"
- Run the embedding generation: `python embeddings/embed_chunks.py`
- Ensure PDFs are processed first

### "Tesseract not found"
- Install Tesseract OCR (see Prerequisites)
- Add Tesseract to your system PATH

## Key Learnings
- Implemented a full RAG architecture from scratch
- Gained hands-on experience with vector databases and semantic search
- Integrated LLMs with retrieved context for grounded responses
- Designed modular components for easy extensibility

## Future Improvements
- [ ] Support for more document formats (DOCX, TXT, etc.)
- [ ] Hybrid retrieval (keyword + vector search)
- [ ] Advanced chunking strategies (semantic chunking)
- [ ] Chat history and conversation memory
- [ ] Multi-document comparison queries
- [ ] Evaluation metrics and benchmarking

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


