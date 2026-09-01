"""Dense vector retriever over the persisted Chroma collection.

Embeds the query locally with the same BAAI/bge-small-en-v1.5 model used at
ingestion time (via src.ingest.get_collection) and returns the top-k most
similar chunks. Read-only: never writes to the collection. BM25, fusion,
and reranking are implemented separately and share this module's Chunk
type and the retrieve(query, k) -> list[Chunk] contract.
"""

from dataclasses import dataclass

from chromadb import Collection

from src.config import RETRIEVAL
from src.ingest import get_collection


@dataclass(frozen=True)
class Chunk:
    """A retrieved chunk: stored text plus the metadata captured at ingestion.

    `score` is Chroma's raw distance for this match (lower = more similar);
    `None` if a caller constructs a Chunk without one.
    """

    text: str
    clause: str
    chapter: str
    page: int
    heading: str
    score: float | None = None


_collection: Collection | None = None


def _get_cached_collection() -> Collection:
    """The Chroma collection, loaded once per process and reused across calls
    so repeated queries don't reload the embedding model from disk each time.
    """
    global _collection
    if _collection is None:
        _collection = get_collection()
    return _collection


def retrieve(query: str, k: int = RETRIEVAL.k_final, collection: Collection | None = None) -> list[Chunk]:
    """Return the top-k chunks from `fssai_regulations` most similar to `query`.

    `query` is embedded locally by the collection's configured embedding
    function (BAAI/bge-small-en-v1.5, the same model src/ingest.py used to
    embed documents) — no external API call is made. `k` defaults to
    src.config.RETRIEVAL.k_final, the same final context size every
    ablation arm uses; callers pass a larger `k` explicitly when using this
    as a first-stage candidate pool (e.g. inside hybrid.retrieve).

    `collection` optionally overrides the default cached `fssai_regulations`
    collection with a different, already-constructed Collection — used only
    by the secondary corpus-scale experiment (scripts/recall_at_k_large.py)
    to query a separate collection. Every other caller leaves this unset
    and gets the original default-collection behavior unchanged.
    """
    target_collection = collection if collection is not None else _get_cached_collection()
    results = target_collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        Chunk(
            text=document,
            clause=metadata["clause"],
            chapter=metadata["chapter"],
            page=metadata["page"],
            heading=metadata["heading"],
            score=distance,
        )
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]
