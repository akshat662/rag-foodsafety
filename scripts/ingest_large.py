"""SECONDARY EXPERIMENT -- ingest a ~4x larger FSSAI corpus into a new,
separate Chroma collection, to test whether primary-experiment recall@3
saturation (1.00 on the 3-chapter/124-chunk corpus) is a property of dense
retrieval or an artifact of a small corpus.

Reuses src.chunking.chunk_document UNCHANGED -- no chunking logic is
duplicated here. Reuses the same embedding model as the primary experiment
(imported from src.ingest.EMBEDDING_MODEL_NAME, not redefined) so the ONLY
variable versus the primary corpus is corpus size, per the experiment's
absolute constraints.

Does NOT touch data/processed/, the existing `fssai_regulations` Chroma
collection, or data/chroma/ -- writes to a brand-new collection
(fssai_large) in a brand-new persist directory (data/chroma_large/).

PDFs are read directly from data/raw/, which -- for this secondary
experiment -- holds all 9 chapters (the original 3: 2.1, 2.4, 2.9, plus 6
new ones: 2.2, 2.3, 2.5, 2.6, 2.7, 2.10). Page text is extracted directly
with PyMuPDF (the same library/method src/ingest.py's upstream extraction
step, scripts/extract_check.py, uses) into a true per-page list, rather
than round-tripping through data/processed/*.txt and re-splitting on a
page-footer regex that was reverse-engineered from only the original 3
PDFs' footer format and is not guaranteed to generalize to the 6 new ones.

Run with: python -m scripts.ingest_large
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
from chromadb import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.chunking import Chunk, chunk_document
from src.ingest import EMBEDDING_MODEL_NAME

RAW_DIR = Path("data/raw")
CHROMA_DIR = Path("data/chroma_large")
COLLECTION_NAME = "fssai_large"
QA_SET_PATH = Path("data/qa_set.json")


@dataclass(frozen=True)
class SourceDocument:
    """One source PDF and the chapter identity chunk_document() needs.

    Explicit mapping (not filename parsing) so a typo or an unexpected file
    in data/raw/ fails loudly at startup rather than silently mis-chunking
    or silently omitting a chapter.
    """

    filename: str
    chapter_id: str
    chapter_title: str


SOURCE_DOCUMENTS: tuple[SourceDocument, ...] = (
    SourceDocument("Chapter 2_1_Dairy_products_and_analogues.pdf", "2.1", "Dairy products and analogues"),
    SourceDocument("Chapter 2_2_Fats_oils and fat emulsions.pdf", "2.2", "Fats, oils and fat emulsions"),
    SourceDocument("Chapter 2_3_Fruit_Vegetable_products.pdf", "2.3", "Fruit and Vegetable products"),
    SourceDocument("Chapter 2_4_Cereals_and_Cereal_products.pdf", "2.4", "Cereals and Cereal products"),
    SourceDocument("Chapter 2_5_Meat and Meat products(1).pdf", "2.5", "Meat and Meat products"),
    SourceDocument("Chapter 2_6_Fish_and_Fish_products.pdf", "2.6", "Fish and Fish products"),
    SourceDocument("Chapter 2_7 (Sweets and Confectionary).pdf", "2.7", "Sweets and Confectionary"),
    SourceDocument(
        "Chapter 2_9_Salt_Spices_Condiments and related products.pdf",
        "2.9",
        "Salt, Spices, Condiments and related products",
    ),
    SourceDocument(
        "Chapter 2_10_BEVERAGES_Other than Dairy and Fruits Vegetables based.pdf",
        "2.10",
        "Beverages, Other than Dairy and Fruits & Vegetables based",
    ),
)


def _verify_source_documents_match_raw_dir() -> None:
    """Fail loudly if SOURCE_DOCUMENTS and data/raw/*.pdf have diverged --
    either a mapped file is missing, or an unmapped PDF is sitting in
    data/raw/ and would otherwise be silently excluded from the corpus.
    """
    mapped = {doc.filename for doc in SOURCE_DOCUMENTS}
    present = {p.name for p in RAW_DIR.glob("*.pdf")}

    missing = mapped - present
    if missing:
        raise SystemExit(f"FAIL: mapped source PDF(s) not found in {RAW_DIR}: {sorted(missing)}")

    unmapped = present - mapped
    if unmapped:
        raise SystemExit(
            f"FAIL: {RAW_DIR} contains PDF(s) with no SOURCE_DOCUMENTS entry -- "
            f"they would be silently excluded from the large corpus: {sorted(unmapped)}"
        )


def _extract_pages(pdf_path: Path) -> list[str]:
    """One text string per PDF page, in order -- the exact shape
    src.chunking.chunk_document expects for its `pages` argument.
    """
    doc = fitz.open(pdf_path)
    try:
        return [page.get_text() for page in doc]
    finally:
        doc.close()


def load_and_chunk_all() -> dict[str, list[Chunk]]:
    """Extract and chunk every source document. Returns chunks grouped by
    chapter_id so per-chapter counts can be reported.
    """
    chunks_by_chapter: dict[str, list[Chunk]] = {}
    for doc in SOURCE_DOCUMENTS:
        pages = _extract_pages(RAW_DIR / doc.filename)
        chunks_by_chapter[doc.chapter_id] = chunk_document(pages, doc.chapter_id, doc.chapter_title)
    return chunks_by_chapter


def _chunk_id(chunk: Chunk) -> str:
    """Deterministic chunk ID, same scheme as src.ingest._chunk_id, so
    re-running this script upserts in place instead of duplicating records.
    """
    return f"{chunk.chapter_id}:{chunk.clause_number}:{chunk.part_index}"


def _chunk_metadata(chunk: Chunk) -> dict[str, str | int]:
    """Same metadata shape as src.ingest._chunk_metadata."""
    return {
        "clause": chunk.clause_number,
        "chapter": chunk.chapter_id,
        "page": chunk.page_start if chunk.page_start is not None else -1,
        "heading": chunk.heading,
    }


def get_large_collection() -> Collection:
    """The new, separate persistent Chroma collection for the secondary
    experiment. Same embedding model as the primary experiment (imported,
    not redefined); distinct name and distinct persist directory so the
    primary `fssai_regulations` collection in data/chroma/ is never touched.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)


# ---------------------------------------------------------------------------
# Gold-clause coverage check.
# ---------------------------------------------------------------------------


def _clause_tuple(clause: str) -> tuple[int, ...]:
    return tuple(int(p) for p in clause.split("."))


def _stored_label_covers(gold: str, stored_label: str) -> bool:
    """True if `gold` (e.g. "2.4.4") is covered by `stored_label`, which is
    either an exact clause number ("2.4.4") or a merged-range label produced
    by src.chunking's short-clause merging ("2.4.4-2.4.5"), per the same
    _ClauseGroup.number_label format used by the primary collection (see
    e.g. "2.4.4-2.4.5" observed in the primary `fssai_regulations`
    collection's metadata for this exact clause).
    """
    if "-" not in stored_label:
        return gold == stored_label
    start, end = stored_label.split("-", 1)
    gold_t, start_t, end_t = _clause_tuple(gold), _clause_tuple(start), _clause_tuple(end)
    return start_t <= gold_t <= end_t


def _gold_clauses_from_qa_set() -> set[str]:
    import json

    questions = json.loads(QA_SET_PATH.read_text())
    gold: set[str] = set()
    for q in questions:
        gold.update(q["source_clauses"])
    return gold


def verify_gold_clauses_present(collection: Collection) -> None:
    """FAIL LOUDLY if any gold clause referenced by the frozen QA set is
    missing from the new collection's metadata -- data/qa_set.json itself
    is never read for anything but this read-only check, and is never
    modified.
    """
    records = collection.get(include=["metadatas"])
    stored_labels = [m["clause"] for m in records["metadatas"]]

    gold_clauses = _gold_clauses_from_qa_set()
    missing = sorted(
        gold for gold in gold_clauses if not any(_stored_label_covers(gold, label) for label in stored_labels)
    )
    if missing:
        raise SystemExit(
            "FAIL: the following gold clauses from data/qa_set.json are MISSING from the "
            f"'{COLLECTION_NAME}' collection's metadata: {missing}. Refusing to proceed -- "
            "the secondary experiment cannot be scored against ground truth without them."
        )

    print(f"Gold clause check: all {len(gold_clauses)} clauses referenced by data/qa_set.json are present.")


def ingest() -> None:
    _verify_source_documents_match_raw_dir()

    chunks_by_chapter = load_and_chunk_all()
    collection = get_large_collection()

    all_chunks = [c for chunks in chunks_by_chapter.values() for c in chunks]
    collection.upsert(
        ids=[_chunk_id(c) for c in all_chunks],
        documents=[c.text for c in all_chunks],
        metadatas=[_chunk_metadata(c) for c in all_chunks],
    )

    print("Ingestion summary (SECONDARY experiment -- large corpus)")
    print(f"  collection name  : {COLLECTION_NAME}")
    print(f"  persisted at     : {CHROMA_DIR.resolve()}")
    print(f"  embedding model  : {EMBEDDING_MODEL_NAME}")
    print(f"  source documents : {len(SOURCE_DOCUMENTS)}")
    print("  chunks per chapter:")
    for doc in SOURCE_DOCUMENTS:
        print(f"    {doc.chapter_id:<6} {doc.chapter_title:<55} {len(chunks_by_chapter[doc.chapter_id]):>4} chunks")
    print(f"  total chunks     : {len(all_chunks)}")
    print(f"  embeddings stored: {collection.count()}")
    print()

    verify_gold_clauses_present(collection)


if __name__ == "__main__":
    sys.exit(ingest())
