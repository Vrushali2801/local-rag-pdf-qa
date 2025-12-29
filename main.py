from retrieval.hybrid_search import retrieve_context
from llm.generate_answer import generate_answer


def ask_rag(query):
    print("\n### User Question ###")
    print(query)

    # Retrieve relevant text chunks
    ret = retrieve_context(query)
    context = ret["context"]

    print("\n### Retrieved Context (top chunks) ###")
    print(context[:500] + "..." if len(context) > 500 else context)

    # Generate answer using Ollama
    answer = generate_answer(context, query, model="phi3")

    print("\n### Final Answer ###\n")
    print(answer)

    print("\n### Chunks Used ###")
    for sid, _ in ret["sources"]:
        print("- ", sid)


if __name__ == "__main__":
    user_q = "What topics does this document talk about?"
    ask_rag(user_q)
