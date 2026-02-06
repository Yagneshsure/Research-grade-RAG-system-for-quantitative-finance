import os
import sys

# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from rag_pipeline import answer_query


if __name__ == "__main__":
    queries = [
        "What is volatility clustering?",
        "Explain market microstructure noise",
        "Why do momentum strategies decay over time?"
    ]

    for q in queries:
        print("\n" + "=" * 90)
        print(f"QUESTION: {q}")
        print("=" * 90)

        answer, sources = answer_query(q)

        print("\nANSWER:")
        print(answer)

        print("\nSOURCES USED:")
        seen = set()
        for i, doc in enumerate(sources, 1):
            meta = doc.metadata
            key = (meta["source"], meta["page"])
            if key in seen:
                continue
            seen.add(key)

            print(
                f"{i}. {meta['source']} "
                f"({meta['domain']}, page {meta['page']})"
            )
