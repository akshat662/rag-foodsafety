"""BM25 lexical retriever over the same chunks stored by src/ingest.py.

Tokenizes chunk text and queries the same deterministic way and ranks with
rank_bm25's BM25Okapi — no embeddings or Chroma similarity search are used
for ranking. Corpus chunks are read once from the persisted Chroma
collection (a plain fetch, not a similarity query), so this retriever
ranks over exactly the same records the dense retriever queries.
"""

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.ingest import get_collection
from src.retrieval.vector import Chunk

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Deterministic lowercase alphanumeric tokenizer, shared by corpus and queries."""
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass
class _Corpus:
    """Chunks plus the BM25 index built over their tokenized text."""

    chunks: list[Chunk]
    bm25: BM25Okapi


_corpus: _Corpus | None = None


def _load_corpus() -> _Corpus:
    """Fetch every stored chunk from Chroma (no similarity search) and build
    the BM25 index over their tokenized text.
    """
    collection = get_collection()
    records = collection.get(include=["documents", "metadatas"])

    chunks = [
        Chunk(
            text=document,
            clause=metadata["clause"],
            chapter=metadata["chapter"],
            page=metadata["page"],
            heading=metadata["heading"],
        )
        for document, metadata in zip(records["documents"], records["metadatas"])
    ]
    tokenized_corpus = [_tokenize(c.text) for c in chunks]
    return _Corpus(chunks=chunks, bm25=BM25Okapi(tokenized_corpus))


def _get_cached_corpus() -> _Corpus:
    """The chunk corpus and its BM25 index, loaded once per process and reused."""
    global _corpus
    if _corpus is None:
        _corpus = _load_corpus()
    return _corpus


def retrieve(query: str, k: int) -> list[Chunk]:
    """Return the top-k chunks ranked by BM25 lexical score for `query`.

    Same public interface as src.retrieval.vector.retrieve, so BM25 and
    dense retrieval results can be compared or fused directly.
    """
    corpus = _get_cached_corpus()
    scores = corpus.bm25.get_scores(_tokenize(query))

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    return [
        Chunk(
            text=corpus.chunks[i].text,
            clause=corpus.chunks[i].clause,
            chapter=corpus.chunks[i].chapter,
            page=corpus.chunks[i].page,
            heading=corpus.chunks[i].heading,
            score=float(scores[i]),
        )
        for i in ranked_indices
    ]
