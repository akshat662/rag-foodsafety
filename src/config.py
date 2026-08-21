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
    # GPT-5-family reasoning effort ("minimal"/"low"/"medium"/"high"/...).
    # None for non-reasoning models (e.g. GENERATOR's Qwen on Groq), which
    # don't accept this parameter at all.
    reasoning_effort: str | None = None


# Generation.
# temperature=0 (lowest the Groq API allows) so the Day 3 ablation is
# reproducible: re-running generation for the same (question, arm) must
# yield the same answer, or run artifacts stop being comparable.
# Model is Qwen (not OpenAI's gpt-oss) so the generator and the eventual
# OpenAI RAGAS judge come from different model families — see decisions.md,
# 2026-08-20, "Reportable generator model: Qwen".
GENERATOR = ProviderConfig(
    provider="groq",
    model="qwen/qwen3.6-27b",
    rate_limits=RateLimits(rpm=30, tpm=8000, rpd=1000, tpd=200000),
    temperature=0.0,
)

# RAGAS judging.
# Selected over gpt-4o-mini on 2026-08-21 — see decisions.md for the full
# reasoning (cost difference vs. gpt-4o-mini was negligible against the
# project budget; the deciding factor was model lifecycle/reproducibility,
# not price). Pending pilot validation before treated as fully frozen.
# reasoning_effort=minimal: the judge task is structured NLI-style
# evaluation, not open-ended reasoning, so the lowest non-"none" effort
# tier is the appropriate choice; actual reasoning-token usage under this
# setting is unverified until the pilot runs (see eval/run_ragas.py).
# Rate limits intentionally left unset — OpenAI API account not yet configured.
JUDGE = ProviderConfig(
    provider="openai",
    model="gpt-5-mini",
    rate_limits=None,
    reasoning_effort="minimal",
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
