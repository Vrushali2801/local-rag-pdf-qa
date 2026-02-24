import os
from config import PROCESSED_DIR, CHUNK_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunk text with sentence-aware boundaries for better context."""
    chunks = []
    
    # Split into sentences (simple approach)
    sentences = text.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')
    
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence exceeds chunk size and we have content, save chunk
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap (last ~overlap chars)
            if len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += (" " if current_chunk else "") + sentence
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def process_all_text_files():
    os.makedirs(CHUNK_DIR, exist_ok=True)

    for filename in os.listdir(PROCESSED_DIR):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(PROCESSED_DIR, filename)
        print(f"Chunking {filename}...")

        with open(path, "r", encoding="utf8") as f:
            text = f.read()

        chunks = chunk_text(text)

        out_name = filename.replace(".txt", "_chunks.txt")
        out_path = os.path.join(CHUNK_DIR, out_name)

        with open(out_path, "w", encoding="utf8") as f:
            for i, c in enumerate(chunks):
                f.write(f"--- chunk {i} ---\n")
                f.write(c + "\n\n")

        print(f"Saved → {out_path}")


if __name__ == "__main__":
    process_all_text_files()
