import chromadb

client = chromadb.PersistentClient(path="vectorstore/chroma_db")
col = client.get_collection("rag_collection", embedding_function=None)

print("Count:", col.count())
