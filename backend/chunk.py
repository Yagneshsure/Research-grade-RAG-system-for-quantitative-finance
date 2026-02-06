from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def chunk_documents(documents: List[Dict]) -> List[Document]:
    """
    Chunk cleaned documents into semantically meaningful pieces.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=[
            "\n\n",   # paragraph
            "\n",
            ". ",
            "; ",
            ", ",
            " "
        ]
    )

    langchain_docs = [
        Document(
            page_content=doc["page_content"],
            metadata=doc["metadata"]
        )
        for doc in documents
    ]

    chunks = splitter.split_documents(langchain_docs)

    print(f"Total chunks: {len(chunks)}")
    print(f"Average chunks per page: {len(chunks) / len(documents):.2f}")

    return chunks


if __name__ == "__main__":
    # Manual test
    from ingest import load_pdfs
    from preprocess import preprocess_documents
    import os

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "Data")

    raw_docs = load_pdfs(DATA_DIR)
    cleaned_docs = preprocess_documents(raw_docs)
    chunks = chunk_documents(cleaned_docs)

    print("\nSample chunk:")
    print(chunks[0].metadata)
    print(chunks[0].page_content[:500])
