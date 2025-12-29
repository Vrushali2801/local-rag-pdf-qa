import json
from opensearchpy import OpenSearch

HOST = "http://localhost:9200"
INDEX_NAME = "rag_chunks"
EMB_PATH = "data/chunks/chunk_embeddings.jsonl"


def get_client():
    return OpenSearch(
        hosts=[HOST],
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False
    )


def create_index_if_needed(client):
    exists = client.indices.exists(index=INDEX_NAME)

    if exists:
        print(f"Index '{INDEX_NAME}' already exists")
        return

    body = {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 1536  # adjust if needed
                }
            }
        }
    }

    client.indices.create(index=INDEX_NAME, body=body)
    print(f"Created index '{INDEX_NAME}'")


def load_data():
    docs = []
    print(f"Loading embeddings from {EMB_PATH} ...")

    with open(EMB_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)

            docs.append({
                "chunk_id": f"chunk_{i}",
                "text": row["text"],
                "embedding": row["embedding"]
            })

    print(f"Loaded {len(docs)} documents")
    return docs


def upload_docs(client, docs):
    print("Uploading documents...")

    for i, doc in enumerate(docs):
        client.index(
            index=INDEX_NAME,
            id=doc["chunk_id"],
            body=doc
        )

    print("Upload complete.")


if __name__ == "__main__":
    client = get_client()
    create_index_if_needed(client)

    docs = load_data()
    upload_docs(client, docs)
