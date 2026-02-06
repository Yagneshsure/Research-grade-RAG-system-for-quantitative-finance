import os
from pypdf import PdfReader
from typing import List, Dict


def load_pdfs(root_dir: str) -> List[Dict]:
    """
    Robust PDF ingestion.
    - Skips broken / hanging pages
    - Never blocks the full pipeline
    """

    if not os.path.exists(root_dir):
        raise ValueError(f"Data folder not found: {root_dir}")

    documents = []
    skipped_pages = 0

    for domain in os.listdir(root_dir):
        domain_path = os.path.join(root_dir, domain)
        if not os.path.isdir(domain_path):
            continue

        for filename in os.listdir(domain_path):
            if not filename.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(domain_path, filename)

            try:
                reader = PdfReader(pdf_path, strict=False)
            except Exception as e:
                print(f"[SKIP PDF] {filename} ({e})")
                continue

            for page_idx, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                except KeyboardInterrupt:
                    # user interruption: re-raise
                    raise
                except Exception:
                    skipped_pages += 1
                    continue

                if not text:
                    continue

                text = text.strip()
                if len(text) < 50:
                    continue

                documents.append({
                    "page_content": text,
                    "metadata": {
                        "source": filename,
                        "domain": domain,
                        "page": page_idx + 1
                    }
                })

    print(f"Loaded {len(documents)} raw pages")
    print(f"Skipped {skipped_pages} problematic pages")
    return documents


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "Data")

    docs = load_pdfs(DATA_DIR)

    if docs:
        print("Sample document:")
        print(docs[0]["metadata"])
        print(docs[0]["page_content"][:400])
