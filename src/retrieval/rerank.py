"""Local cross-encoder reranker over hybrid RRF candidates.

Scores each (query, chunk.text) pair directly with a local Hugging Face
cross-encoder — joint cross-attention over the pair, not just embedding
cosine similarity — and returns the top-k chunks by that score.
sentence-transformers is already a project dependency (it embeds
BAAI/bge-small-en-v1.5 in src/ingest.py) and also provides CrossEncoder,
so reranking needs no new dependency and makes no external API call.
"""

from sentence_transformers import CrossEncoder

from src.retrieval.vector import Chunk

# ~2ms/pair on CPU, well within budget for the small candidate pools this
# corpus produces; the same model choice as the project's original plan.
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K = 3

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    """The cross-encoder model, loaded once per process and reused across calls."""
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(query: str, candidates: list[Chunk], k: int = DEFAULT_TOP_K) -> list[Chunk]:
    """Rerank `candidates` by a cross-encoder relevance score for (query, chunk.text).

    Unlike embedding similarity, the cross-encoder attends over the query
    and each candidate's full text jointly in one forward pass, rather than
    comparing two independently-computed vectors. Deterministic: inference
    is a fixed forward pass with no sampling, so identical inputs always
    produce identical scores and ordering.
    """
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:k]

    return [
        Chunk(
            text=c.text,
            clause=c.clause,
            chapter=c.chapter,
            page=c.page,
            heading=c.heading,
            score=float(score),
        )
        for c, score in ranked
    ]
