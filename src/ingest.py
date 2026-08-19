"""Ingest the FSSAI regulatory chapters into a persistent Chroma vector store.

Reads the extracted per-chapter text in data/processed/, chunks each with
src.chunking (chunking logic is not duplicated here), embeds chunks locally
with BAAI/bge-small-en-v1.5, and upserts them into a persistent Chroma
collection. Retrieval is implemented separately in src/retrieval/vector.py.

Run with: python -m src.ingest
"""

import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.chunking import Chunk, chunk_document

PROCESSED_DIR = Path("data/processed")
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "fssai_regulations"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class SourceDocument:
    """One extracted chapter file and the chapter identity chunk_document() needs."""

    filename: str
    chapter_id: str
    chapter_title: str


SOURCE_DOCUMENTS: tuple[SourceDocument, ...] = (
    SourceDocument(
        "Chapter 2_1_Dairy_products_and_analogues.txt", "2.1", "Dairy products and analogues"
    ),
    SourceDocument(
        "Chapter 2_4_Cereals_and_Cereal_products.txt", "2.4", "Cereals and Cereal products"
    ),
    SourceDocument(
        "Chapter 2_9_Salt_Spices_Condiments and related products.txt",
        "2.9",
        "Salt, Spices, Condiments and related products",
    ),
)

# Marks the start of each source page in the flattened extraction, e.g.
# "1 | V e r s i o n 4 ( 0 1 . 0 8 . 2 0 2 5 )". Splitting on it recovers the
# per-page list chunk_document() expects; this is extraction-format parsing,
# not chunking logic.
_PAGE_MARKER = re.compile(r"^\d+ \| V e r s i o n \d+ \(", re.MULTILINE)


def _load_pages(path: Path) -> list[str]:
    """Recover the original per-page text from a flattened data/processed/*.txt file."""
    full_text = path.read_text()
    starts = [m.start() for m in _PAGE_MARKER.finditer(full_text)]
    if not starts:
        return [full_text]
    if starts[0] != 0:
        starts = [0] + starts
    starts.append(len(full_text))
    return [full_text[starts[i] : starts[i + 1]] for i in range(len(starts) - 1)]


def _chunk_id(chunk: Chunk) -> str:
    """Deterministic chunk ID: the same chunk always maps to the same ID, so
    re-running ingestion upserts in place instead of duplicating records.
    """
    return f"{chunk.chapter_id}:{chunk.clause_number}:{chunk.part_index}"


def _chunk_metadata(chunk: Chunk) -> dict[str, str | int]:
    """The metadata every stored chunk must carry: clause, chapter, page, heading."""
    return {
        "clause": chunk.clause_number,
        "chapter": chunk.chapter_id,
        "page": chunk.page_start if chunk.page_start is not None else -1,
        "heading": chunk.heading,
    }


def load_and_chunk_all() -> list[Chunk]:
    """Load and chunk all three source documents.

    All chunking logic lives in src.chunking; this only supplies each
    document's pages and chapter identity.
    """
    chunks: list[Chunk] = []
    for doc in SOURCE_DOCUMENTS:
        pages = _load_pages(PROCESSED_DIR / doc.filename)
        chunks.extend(chunk_document(pages, doc.chapter_id, doc.chapter_title))
    return chunks


def get_collection() -> Collection:
    """The persistent Chroma collection, embedding documents locally with bge-small."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )


def ingest() -> None:
    """Chunk all source documents and upsert them into the persistent Chroma
    collection.

    Upsert (not add) by deterministic ID makes this idempotent: re-running
    ingestion overwrites existing records in place instead of duplicating
    them, so the collection's document count does not grow on repeat runs.
    """
    chunks = load_and_chunk_all()
    collection = get_collection()

    collection.upsert(
        ids=[_chunk_id(c) for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[_chunk_metadata(c) for c in chunks],
    )

    print("Ingestion summary")
    print(f"  source documents : {len(SOURCE_DOCUMENTS)}")
    print(f"  chunks           : {len(chunks)}")
    print(f"  embeddings       : {collection.count()}")
    print(f"  collection name  : {COLLECTION_NAME}")
    print(f"  persisted at     : {CHROMA_DIR.resolve()}")


if __name__ == "__main__":
    ingest()
