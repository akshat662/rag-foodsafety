"""Throwaway extraction-quality check. Not part of the pipeline."""

from pathlib import Path

import fitz  # PyMuPDF

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
        doc = fitz.open(pdf_path)

        page_texts = [page.get_text() for page in doc]
        full_text = "\n".join(page_texts)

        out_path = PROCESSED_DIR / f"{pdf_path.stem}.txt"
        out_path.write_text(full_text, encoding="utf-8")

        page_count = len(page_texts)
        total_chars = sum(len(t) for t in page_texts)
        sparse_pages = sum(1 for t in page_texts if len(t) < 100)

        print(f"{pdf_path.name}")
        print(f"  page count: {page_count}")
        print(f"  total extracted characters: {total_chars}")
        print(f"  pages with fewer than 100 characters: {sparse_pages}")

        doc.close()


if __name__ == "__main__":
    main()
