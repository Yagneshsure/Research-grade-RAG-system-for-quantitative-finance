import os
from vector_store import load_faiss_index


def retrieve(query: str, k: int = 3):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_dir = os.path.join(project_root, "vector_store", "faiss_index")

    vectorstore = load_faiss_index(index_dir)
    return vectorstore.similarity_search(query, k=k)


if __name__ == "__main__":
    queries = [
        "What is volatility clustering?",
        "Explain market microstructure noise",
        "Why do momentum strategies decay over time?"
    ]

    for q in queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {q}")
        print("=" * 80)

        results = retrieve(q, k=3)

        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Source : {doc.metadata['source']}")
            print(f"Domain : {doc.metadata['domain']}")
            print(doc.page_content[:500])
