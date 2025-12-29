# Local RAG PDF Question Answering System

An end-to-end Retrieval-Augmented Generation (RAG) pipeline for querying PDF documents using semantic search and a locally hosted LLM.

## Overview
This project implements a complete RAG workflow that allows users to upload PDF documents, retrieve relevant content using vector similarity search, and generate answers grounded in the retrieved context.

The system is designed to be modular, making it easy to swap components such as the embedding model, vector store, or language model.

## Pipeline Architecture
1. **Document Ingestion** – Extracts text from PDF files
2. **Text Chunking** – Splits documents into manageable chunks
3. **Embedding Generation** – Converts text chunks into dense vector representations
4. **Vector Retrieval** – Uses FAISS to perform semantic similarity search
5. **Answer Generation** – Feeds retrieved context to a local LLM for question answering


## How It Works
1. PDF files are processed and converted into clean text.
2. The text is split into chunks for efficient retrieval.
3. Each chunk is embedded using SentenceTransformers.
4. FAISS indexes the embeddings for fast similarity search.
5. Relevant chunks are retrieved based on user queries.
6. Retrieved context is passed to a local LLM to generate accurate answers.

## Key Learnings
- Implemented a full RAG architecture from scratch
- Gained hands-on experience with vector databases and semantic search
- Integrated LLMs with retrieved context for grounded responses
- Designed modular components for easy extensibility

## Future Improvements
- Support for multiple document sources
- Hybrid retrieval (keyword + vector search)
- Improved chunking strategies
- UI integration for interactive querying


