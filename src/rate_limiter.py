"""Provider-agnostic token-bucket rate limiter with exponential backoff.

Wraps any callable (e.g. a provider SDK's completion function) without
importing that provider's SDK. Rate limits come from a ProviderConfig
in src/config.py, not from this module.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.config import ProviderConfig


class RateLimitError(Exception):
    """Raise this from wrapped calls to signal a 429 / provider rate-limit response."""


class RetryExhaustedError(Exception):
    """Raised when max_retries is exceeded while retrying a rate-limited call."""


class _TokenBucket:
    """Continuous-refill token bucket for a single limit dimension (requests or tokens).

    Not thread-safe: callers are expected to drive one RateLimiter from one thread.
    """

    def __init__(
        self,
        capacity: float,
        refill_per_second: float,
        *,
        time_fn: Callable[[], float],
    ):
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self._time_fn = time_fn
        self._tokens = capacity
        self._last_refill = time_fn()

    def _refill(self) -> None:
        now = self._time_fn()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_second)
        self._last_refill = now

    def time_until_available(self, amount: float) -> float:
        if amount > self.capacity:
            raise ValueError(f"requested amount {amount} exceeds bucket capacity {self.capacity}")
        self._refill()
        if self._tokens >= amount:
            return 0.0
        return (amount - self._tokens) / self.refill_per_second

    def consume(self, amount: float) -> None:
        self._refill()
        self._tokens = max(0.0, self._tokens - amount)


def estimate_tokens(text: str) -> int:
    """Rough, provider-agnostic token estimate (~4 chars/token). Pad upward for safety."""
    return max(1, len(text) // 4)


@dataclass
class RateLimiter:
    """Paces calls against a provider's RPM/TPM limits and retries 429s with backoff."""

    provider_config: ProviderConfig
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    sleep_fn: Callable[[float], None] = field(default=time.sleep, repr=False)
    time_fn: Callable[[], float] = field(default=time.monotonic, repr=False)
    random_fn: Callable[[], float] = field(default=random.random, repr=False)

    def __post_init__(self) -> None:
        limits = self.provider_config.rate_limits
        if limits is None:
            raise ValueError(
                f"no rate limits configured for provider={self.provider_config.provider!r} "
                f"model={self.provider_config.model!r}"
            )
        self._request_bucket = _TokenBucket(limits.rpm, limits.rpm / 60.0, time_fn=self.time_fn)
        self._token_bucket = _TokenBucket(limits.tpm, limits.tpm / 60.0, time_fn=self.time_fn)

    def _wait_for_capacity(self, estimated_tokens: int) -> None:
        while True:
            wait = max(
                self._request_bucket.time_until_available(1),
                self._token_bucket.time_until_available(estimated_tokens),
            )
            if wait <= 0:
                self._request_bucket.consume(1)
                self._token_bucket.consume(estimated_tokens)
                return
            self.sleep_fn(wait)

    def call(
        self,
        fn: Callable[..., Any],
        *,
        estimated_tokens: int,
        args: tuple = (),
        kwargs: dict | None = None,
        is_rate_limit_error: Callable[[Exception], bool] | None = None,
    ) -> Any:
        """Pace `fn(*args, **kwargs)` against RPM/TPM, retrying on rate-limit errors.

        `estimated_tokens` must be computed by the caller before calling this
        (e.g. via `estimate_tokens`) so pacing accounts for the request before
        it is sent, not after.
        """
        kwargs = kwargs or {}
        is_rate_limit_error = is_rate_limit_error or _default_is_rate_limit_error

        attempt = 0
        while True:
            self._wait_for_capacity(estimated_tokens)
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if not is_rate_limit_error(exc):
                    raise
                attempt += 1
                if attempt > self.max_retries:
                    raise RetryExhaustedError(
                        f"exceeded {self.max_retries} retries after repeated rate-limit errors"
                    ) from exc
                # Full jitter (AWS-style): uniform(0, delay) avoids retry synchronization
                # across callers without needing to track prior jitter draws.
                delay = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
                self.sleep_fn(self.random_fn() * delay)


def _default_is_rate_limit_error(exc: Exception) -> bool:
    """Detects 429s via our own RateLimitError or common `status_code` attribute
    shapes (used by requests/httpx-based SDKs) without importing any SDK.
    """
    if isinstance(exc, RateLimitError):
        return True
    if getattr(exc, "status_code", None) == 429:
        return True
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True
    return False
