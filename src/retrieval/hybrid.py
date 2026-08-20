"""Reciprocal Rank Fusion over the dense (vector) and lexical (BM25) retrievers.

Combines src.retrieval.vector.retrieve and src.retrieval.bm25.retrieve with
RRF — a rank-based fusion, not a score-based one — so the two retrievers'
incomparable score scales (Chroma distance vs. BM25's unbounded lexical
score) are never compared or combined directly. Reranking and generation
are implemented separately.
"""

from dataclasses import dataclass

from src.config import RETRIEVAL
from src.retrieval import bm25, vector
from src.retrieval.vector import Chunk


def _chunk_key(chunk: Chunk) -> str:
    """A deterministic identity for a chunk, used to deduplicate results that
    both retrievers surface.

    Retrieval-layer Chunks (src.retrieval.vector.Chunk) don't carry
    ingestion's chunk ID, so identity is derived from chapter + clause +
    page + exact stored text. The same underlying chunk always reproduces
    identical text; distinct chunks — including different parts of a split
    clause, which share a clause number but not their text — never collide.
    """
    return f"{chunk.chapter}:{chunk.clause}:{chunk.page}:{chunk.text}"


@dataclass
class _Fused:
    """One deduplicated chunk plus its accumulated RRF score."""

    chunk: Chunk
    rrf_score: float


def retrieve(
    query: str,
    k: int = RETRIEVAL.k_final,
    candidate_k: int = RETRIEVAL.candidate_k,
    rrf_k: int = RETRIEVAL.rrf_k,
) -> list[Chunk]:
    """Return the top-k chunks by Reciprocal Rank Fusion of dense + BM25 results.

    Each retriever supplies its own top-`candidate_k` ranked list (the same
    pool size for both dense and BM25). A chunk's RRF score is the sum of
    `1 / (rrf_k + rank)` over every list it appears in, using each
    retriever's 1-indexed rank only — never its raw score, so vector
    distances and BM25 scores are never compared or combined. Defaults come
    from src.config.RETRIEVAL, the single source of truth shared by every
    ablation arm.
    """
    ranked_lists = (
        vector.retrieve(query, candidate_k),
        bm25.retrieve(query, candidate_k),
    )

    fused: dict[str, _Fused] = {}
    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            key = _chunk_key(chunk)
            contribution = 1.0 / (rrf_k + rank)
            if key in fused:
                fused[key].rrf_score += contribution
            else:
                fused[key] = _Fused(chunk=chunk, rrf_score=contribution)

    top = sorted(fused.values(), key=lambda f: f.rrf_score, reverse=True)[:k]

    return [
        Chunk(
            text=f.chunk.text,
            clause=f.chunk.clause,
            chapter=f.chunk.chapter,
            page=f.chunk.page,
            heading=f.chunk.heading,
            score=f.rrf_score,
        )
        for f in top
    ]
