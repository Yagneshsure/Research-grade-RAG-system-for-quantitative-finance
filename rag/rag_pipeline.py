import os
import sys
from typing import List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from backend.vector_store import load_faiss_index
from langchain.schema import Document

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
)

INDEX_DIR = os.path.join(PROJECT_ROOT, "vector_store", "faiss_index")
TOP_K = 6
MAX_FACT_CHARS = 1200

MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"

# -------------------------
# Load model SAFELY on Windows
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto"
)

generator = TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=220,
    do_sample=False,
)

# -------------------------
# RAG logic
# -------------------------
def retrieve_docs(query: str) -> List[Document]:
    vs = load_faiss_index(INDEX_DIR)
    return vs.similarity_search(query, k=TOP_K)


def extract_facts(docs: List[Document]) -> str:
    text = " ".join(
        d.page_content.replace("\n", " ").strip()
        for d in docs
    )
    return text[:MAX_FACT_CHARS]


def build_prompt(question: str, facts: str) -> str:
    return f"""
You are a financial economics researcher.

TASK:
- Give a clear definition
- Explain the economic or statistical mechanism
- Do NOT quote text
- Do NOT mention authors or papers
- Write 4–6 sentences

FACTS:
{facts}

QUESTION:
{question}

ANSWER:
""".strip()


def answer_query(query: str) -> Tuple[str, List[Document]]:
    docs = retrieve_docs(query)
    facts = extract_facts(docs)
    prompt = build_prompt(query, facts)

    output = generator(prompt)[0]["generated_text"]
    answer = output.split("ANSWER:")[-1].strip()

    if len(answer.split()) < 15:
        answer = (
            "The concept is discussed in the literature, but the extracted "
            "material does not contain a clear explanatory passage."
        )

    return answer, docs
