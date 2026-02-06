import re
from typing import List, Dict


REFERENCE_PATTERNS = re.compile(
    r"(references|bibliography|acknowledg(e)?ments)",
    flags=re.IGNORECASE
)


def clean_text(text: str) -> str:
    """
    Light but effective PDF text cleaning.
    """

    # Fix broken newlines inside paragraphs
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Collapse excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove references / bibliography if present
    split_text = REFERENCE_PATTERNS.split(text)
    if split_text:
        text = split_text[0]

    return text.strip()


def preprocess_documents(documents: List[Dict]) -> List[Dict]:
    cleaned_docs = []

    for doc in documents:
        cleaned_text = clean_text(doc["page_content"])

        # Drop near-empty or junk pages
        if len(cleaned_text) < 120:
            continue

        doc["page_content"] = cleaned_text
        cleaned_docs.append(doc)

    print(f"Cleaned pages: {len(cleaned_docs)}")
    return cleaned_docs


if __name__ == "__main__":
    # Manual test
    from ingest import load_pdfs
    import os

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "Data")

    raw_docs = load_pdfs(DATA_DIR)
    cleaned_docs = preprocess_documents(raw_docs)

    print("\nSample cleaned document:")
    print(cleaned_docs[0]["metadata"])
    print(cleaned_docs[0]["page_content"][:500])
