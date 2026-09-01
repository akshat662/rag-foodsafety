"""BM25 lexical retriever over the same chunks stored by src/ingest.py.

Tokenizes chunk text and queries the same deterministic way and ranks with
rank_bm25's BM25Okapi — no embeddings or Chroma similarity search are used
for ranking. Corpus chunks are read once from the persisted Chroma
collection (a plain fetch, not a similarity query), so this retriever
ranks over exactly the same records the dense retriever queries.
"""

import re
from dataclasses import dataclass

from chromadb import Collection
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
_alt_corpora: dict[int, _Corpus] = {}


def _load_corpus(collection: Collection | None = None) -> _Corpus:
    """Fetch every stored chunk from Chroma (no similarity search) and build
    the BM25 index over their tokenized text.

    `collection` optionally overrides the default `fssai_regulations`
    collection — see retrieve()'s docstring.
    """
    source_collection = collection if collection is not None else get_collection()
    records = source_collection.get(include=["documents", "metadatas"])

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


def _get_cached_corpus(collection: Collection | None = None) -> _Corpus:
    """The chunk corpus and its BM25 index, loaded once per process and reused.

    An explicit `collection` is cached separately (keyed by object identity)
    from the default cache, so querying an alternate collection never
    evicts or is evicted by the default one.
    """
    if collection is None:
        global _corpus
        if _corpus is None:
            _corpus = _load_corpus()
        return _corpus

    key = id(collection)
    if key not in _alt_corpora:
        _alt_corpora[key] = _load_corpus(collection)
    return _alt_corpora[key]


def retrieve(query: str, k: int, collection: Collection | None = None) -> list[Chunk]:
    """Return the top-k chunks ranked by BM25 lexical score for `query`.

    Same public interface as src.retrieval.vector.retrieve, so BM25 and
    dense retrieval results can be compared or fused directly.

    `collection` optionally overrides the default cached `fssai_regulations`
    collection with a different, already-constructed Collection — used only
    by the secondary corpus-scale experiment (scripts/recall_at_k_large.py)
    to query a separate collection. Every other caller leaves this unset
    and gets the original default-collection behavior unchanged.
    """
    corpus = _get_cached_corpus(collection)
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
