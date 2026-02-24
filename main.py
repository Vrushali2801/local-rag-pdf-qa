from retrieval.hybrid_search import retrieve_context
from llm.generate_answer import generate_answer
from config import DEFAULT_MODEL


def ask_rag(query):
    """Ask a question using RAG pipeline."""
    print("\n### User Question ###")
    print(query)

    try:
        # Retrieve relevant text chunks
        ret = retrieve_context(query)
        context = ret["context"]

        print("\n### Retrieved Context (top chunks) ###")
        print(context[:500] + "..." if len(context) > 500 else context)

        # Generate answer using Ollama
        answer = generate_answer(context, query, model=DEFAULT_MODEL)

        print("\n### Final Answer ###\n")
        print(answer)

        print("\n### Chunks Used ###")
        for sid, _ in ret["sources"]:
            print("- ", sid)
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("Run: python embeddings/embed_chunks.py")
    except ConnectionError as e:
        print(f"\n❌ Error: {str(e)}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")


if __name__ == "__main__":
    user_q = "What topics does this document talk about?"
    ask_rag(user_q)
