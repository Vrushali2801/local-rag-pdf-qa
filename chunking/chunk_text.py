import os

PROCESSED_DIR = "data/processed"
CHUNK_DIR = "data/chunks"

CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # characters overlap to preserve context


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk.strip())
        start += chunk_size - overlap

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
