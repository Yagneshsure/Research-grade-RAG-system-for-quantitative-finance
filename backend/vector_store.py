import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


def get_embedding_model():
    """
    Local, deterministic embedding model.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_faiss_index(
    documents: List[Document],
    index_dir: str
) -> FAISS:
    """
    Build and persist a FAISS index.
    """

    os.makedirs(index_dir, exist_ok=True)

    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(index_dir)

    return vectorstore


def load_faiss_index(index_dir: str) -> FAISS:
    """
    Load an existing FAISS index.
    """

    embeddings = get_embedding_model()

    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )
