"""FastAPI serving layer for the FSSAI RAG QA system.

Wraps the existing pipeline (src.generate, src.retrieval, eval.run_generation's
arm dispatch) with an HTTP interface. Deliberately does not reimplement any
retrieval or generation logic: retrieval-arm dispatch is reused from
eval.run_generation._retrieve (the same function Day 3's frozen ablation
used), and generation goes through src.generate.generate, which itself
routes through src.llm / the rate limiter, per this project's conventions.

Models and the Chroma collection are loaded once at startup (see the
`lifespan` context manager below) by exercising the real retrieval code
path once, not on each request.

Run with: uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from app.auth import require_api_key
from app.schemas import HealthResponse, LatencyBreakdown, ModelInfo, QueryRequest, QueryResponse
from eval.run_generation import _retrieve
from src.config import GENERATOR
from src.generate import generate
from src.ingest import EMBEDDING_MODEL_NAME, get_collection
from src.logging_utils import log_request
from src.rate_limiter import estimate_tokens
from src.retrieval import hybrid, rerank

# Public API arm names -> the internal A/B/C arms eval.run_generation._retrieve expects.
_ARM_MAP = {"dense": "A", "hybrid": "B", "hybrid_rerank": "C"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-up: exercise the real hybrid+rerank code path once so the Chroma
    # client, the local bge-small embedding model, the BM25 corpus index,
    # and the cross-encoder reranker are all loaded here, at startup --
    # not lazily on whichever request happens to arrive first.
    pool = hybrid.retrieve("warmup query for model preloading", k=3, candidate_k=3)
    rerank.rerank("warmup query for model preloading", pool, k=1)
    yield


app = FastAPI(title="FSSAI RAG QA API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    collection = get_collection()
    return HealthResponse(
        status="ok",
        chroma_chunk_count=collection.count(),
        models=ModelInfo(
            generator_provider=GENERATOR.provider,
            generator_model=GENERATOR.model,
            embedding_model=EMBEDDING_MODEL_NAME,
            reranker_model=rerank.MODEL_NAME,
        ),
    )


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(request: QueryRequest) -> QueryResponse:
    request_id = str(uuid.uuid4())
    internal_arm = _ARM_MAP[request.arm]

    t0 = time.monotonic()
    try:
        chunks = _retrieve(internal_arm, request.question)
    except Exception as exc:  # noqa: BLE001
        log_request(
            request_id=request_id,
            arm=request.arm,
            question=request.question,
            retrieved_clauses=[],
            abstained=None,
            latency_ms={"retrieval_ms": round((time.monotonic() - t0) * 1000, 1), "generation_ms": 0.0, "total_ms": round((time.monotonic() - t0) * 1000, 1)},
            tokens=None,
            status="error",
            error=f"retrieval: {exc!r}",
        )
        raise HTTPException(status_code=500, detail="retrieval failed") from exc
    t1 = time.monotonic()

    try:
        result = generate(request.question, chunks)
    except Exception as exc:  # noqa: BLE001
        t2 = time.monotonic()
        log_request(
            request_id=request_id,
            arm=request.arm,
            question=request.question,
            retrieved_clauses=[c.clause for c in chunks],
            abstained=None,
            latency_ms={
                "retrieval_ms": round((t1 - t0) * 1000, 1),
                "generation_ms": round((t2 - t1) * 1000, 1),
                "total_ms": round((t2 - t0) * 1000, 1),
            },
            tokens=None,
            status="error",
            error=f"generation: {exc!r}",
        )
        raise HTTPException(status_code=500, detail="generation failed") from exc
    t2 = time.monotonic()

    latency = LatencyBreakdown(
        retrieval_ms=round((t1 - t0) * 1000, 1),
        generation_ms=round((t2 - t1) * 1000, 1),
        total_ms=round((t2 - t0) * 1000, 1),
    )
    prompt_tokens = estimate_tokens(request.question) + sum(estimate_tokens(c.text) for c in chunks)
    output_tokens = estimate_tokens(result.answer)

    log_request(
        request_id=request_id,
        arm=request.arm,
        question=request.question,
        retrieved_clauses=[c.clause for c in chunks],
        abstained=result.abstained,
        latency_ms=latency.model_dump(),
        tokens={"prompt_estimated": prompt_tokens, "output_estimated": output_tokens},
        status="ok",
    )

    return QueryResponse(
        request_id=request_id,
        answer=result.answer,
        citations=result.citations,
        abstained=result.abstained,
        retrieved_clauses=[c.clause for c in chunks],
        latency=latency,
    )
