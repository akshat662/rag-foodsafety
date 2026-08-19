"""Single source of truth for provider/model configuration.

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


# Generation.
GENERATOR = ProviderConfig(
    provider="groq",
    model="openai/gpt-oss-20b",
    rate_limits=RateLimits(rpm=30, tpm=8000, rpd=1000, tpd=200000),
)

# RAGAS judging.
# Exact OpenAI judge model to be selected before Day 3.
# Rate limits intentionally left unset — OpenAI API account not yet configured.
JUDGE = ProviderConfig(
    provider="openai",
    model="TBD",
    rate_limits=None,
)
