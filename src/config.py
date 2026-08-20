"""Single source of truth for provider/model and retrieval configuration.

No API keys or secrets live here. Each provider's key is read from its
own environment variable at call time (e.g. GROQ_API_KEY, OPENAI_API_KEY).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimits:
    rpm: int
    tpm: int
    rpd: int
    tpd: int


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    rate_limits: RateLimits | None = None
    # Sampling temperature for generation calls. None leaves the provider's
    # own default in place (e.g. for a judge where determinism doesn't matter
    # the same way). Set explicitly to 0 for reproducible eval runs.
    temperature: float | None = None


# Generation.
# temperature=0 (lowest the Groq API allows) so the Day 3 ablation is
# reproducible: re-running generation for the same (question, arm) must
# yield the same answer, or run artifacts stop being comparable.
GENERATOR = ProviderConfig(
    provider="groq",
    model="openai/gpt-oss-20b",
    rate_limits=RateLimits(rpm=30, tpm=8000, rpd=1000, tpd=200000),
    temperature=0.0,
)

# RAGAS judging.
# Exact OpenAI judge model to be selected before Day 3.
# Rate limits intentionally left unset — OpenAI API account not yet configured.
JUDGE = ProviderConfig(
    provider="openai",
    model="TBD",
    rate_limits=None,
)


@dataclass(frozen=True)
class RetrievalConfig:
    """Ablation-arm retrieval knobs, shared by every arm so k_final is
    identical across dense-only, hybrid, and hybrid+rerank — per the Day 2
    methodological correction, only the retrieval/ranking method should
    differ between arms, never the final context size.
    """

    # Final number of context chunks returned by every arm.
    k_final: int = 3
    # First-stage candidate pool size for both dense and BM25 retrieval,
    # before RRF fusion (hybrid arms only).
    candidate_k: int = 20
    # RRF smoothing constant: RRF_score(d) = sum(1 / (rrf_k + rank(d))).
    rrf_k: int = 60


RETRIEVAL = RetrievalConfig()
