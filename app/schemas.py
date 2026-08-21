"""Request/response models for the FastAPI layer.

No retrieval or generation logic lives here -- these are pure data shapes
around src.generate.GeneratedAnswer and the retrieval arms already defined
in eval/run_generation.py.
"""

from typing import Literal

from pydantic import BaseModel, Field

Arm = Literal["dense", "hybrid", "hybrid_rerank"]


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    arm: Arm


class LatencyBreakdown(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    citations: list[str]
    abstained: bool
    retrieved_clauses: list[str]
    latency: LatencyBreakdown


class ModelInfo(BaseModel):
    generator_provider: str
    generator_model: str
    embedding_model: str
    reranker_model: str


class HealthResponse(BaseModel):
    status: str
    chroma_chunk_count: int
    models: ModelInfo
