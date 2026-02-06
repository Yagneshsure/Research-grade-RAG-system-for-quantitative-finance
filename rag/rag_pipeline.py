import os
import sys
from typing import List

# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from backend.vector_store import load_faiss_index
from transformers import pipeline


def load_llm():
    """
    Lightweight local LLM for grounded synthesis.
    """
    return pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        max_length=512,
        device=-1  # CPU
    )


def retrieve_documents(query: str, k: int = 4):
    """
    Retrieve top-k relevant chunks from FAISS.
    """
    index_dir = os.path.join(PROJECT_ROOT, "vector_store", "faiss_index")
    vectorstore = load_faiss_index(index_dir)
    return vectorstore.similarity_search(query, k=k)


def build_prompt(query: str, docs) -> str:
    """
    Build a strictly grounded prompt.
    """
    context_blocks = []

    for i, doc in enumerate(docs, 1):
        source = doc.metadata["source"]
        domain = doc.metadata["domain"]
        page = doc.metadata["page"]

        block = (
            f"[Source {i} | {domain} | {source} | page {page}]\n"
            f"{doc.page_content}"
        )
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a quantitative finance research assistant.

Answer the question ONLY using the information from the provided sources.
Do NOT use outside knowledge.
If the sources do not contain sufficient information, say:
"Insufficient evidence in the provided documents."

Question:
{query}

Sources:
{context}

Answer (concise, factual, no speculation):
"""
    return prompt.strip()


def answer_query(query: str):
    docs = retrieve_documents(query)
    prompt = build_prompt(query, docs)

    llm = load_llm()
    response = llm(prompt)[0]["generated_text"]

    return response, docs
