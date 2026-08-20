"""Uniform LLM interface across providers.

get_llm(provider, model) is the only way callers should obtain a client.
Every provider SDK import lives inside this module (never elsewhere), every
request is paced through src/rate_limiter.py, and provider/model/rate-limit
configuration is resolved from src/config.py — never hard-coded here.
"""

import json
import os
from abc import ABC, abstractmethod

from src.config import GENERATOR, JUDGE, ProviderConfig
from src.rate_limiter import RateLimiter, estimate_tokens

_KNOWN_CONFIGS: tuple[ProviderConfig, ...] = (GENERATOR, JUDGE)


class LLMClient(ABC):
    """Uniform text/structured-generation interface, regardless of provider."""

    def __init__(self, provider_config: ProviderConfig):
        self.provider_config = provider_config
        self._rate_limiter = RateLimiter(provider_config)

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Generate free-form text for `prompt`."""

    @abstractmethod
    def generate_structured(
        self, prompt: str, schema: dict, *, system: str | None = None
    ) -> dict:
        """Generate output constrained to `schema` (a JSON-Schema-like dict), as a dict.

        Intended for generate.py's {answer, citations, abstained} payload.
        """


class GroqLLMClient(LLMClient):
    """Groq-backed LLMClient.

    The `groq` SDK is imported lazily inside `_get_client`, not at module
    load time, so this module (and get_llm) can be imported and tested
    without the SDK installed or any network access.
    """

    def __init__(self, provider_config: ProviderConfig):
        super().__init__(provider_config)
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set")
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import groq

            self._client = groq.Groq(api_key=self._api_key)
        return self._client

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        messages = _build_messages(prompt, system)

        def call():
            response = self._get_client().chat.completions.create(
                model=self.provider_config.model,
                messages=messages,
                **_temperature_kwargs(self.provider_config),
            )
            return response.choices[0].message.content

        return self._rate_limiter.call(call, estimated_tokens=_estimate_message_tokens(messages))

    def generate_structured(
        self, prompt: str, schema: dict, *, system: str | None = None
    ) -> dict:
        messages = _build_messages(prompt, system, schema=schema)

        def call():
            response = self._get_client().chat.completions.create(
                model=self.provider_config.model,
                messages=messages,
                response_format={"type": "json_object"},
                **_temperature_kwargs(self.provider_config),
            )
            return json.loads(response.choices[0].message.content)

        return self._rate_limiter.call(call, estimated_tokens=_estimate_message_tokens(messages))


class OpenAILLMClient(LLMClient):
    """Reserved for the OpenAI-backed RAGAS judge (src/config.py JUDGE).

    Not implemented yet: the OpenAI API account and exact judge model are
    still pending (see JUDGE.model = "TBD" in src/config.py). This stub
    exists so the provider registry and uniform interface don't need to
    change shape when OpenAI support is added.
    """

    def __init__(self, provider_config: ProviderConfig):
        raise NotImplementedError(
            "OpenAI provider is not implemented yet; JUDGE configuration is pending "
            "(src/config.py)"
        )

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        raise NotImplementedError

    def generate_structured(
        self, prompt: str, schema: dict, *, system: str | None = None
    ) -> dict:
        raise NotImplementedError


_PROVIDERS: dict[str, type[LLMClient]] = {
    "groq": GroqLLMClient,
    "openai": OpenAILLMClient,
}


def get_llm(provider: str, model: str) -> LLMClient:
    """Construct a uniform LLMClient for `provider`/`model`.

    `provider`/`model` must match a ProviderConfig already defined in
    src/config.py (GENERATOR or JUDGE) — that is where rate limits and
    model choices are centralized, not here.
    """
    if provider not in _PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; supported: {sorted(_PROVIDERS)}")

    provider_config = _resolve_provider_config(provider, model)
    return _PROVIDERS[provider](provider_config)


def _resolve_provider_config(provider: str, model: str) -> ProviderConfig:
    for cfg in _KNOWN_CONFIGS:
        if cfg.provider == provider and cfg.model == model:
            return cfg
    raise ValueError(
        f"no ProviderConfig for provider={provider!r} model={model!r}; "
        f"add one in src/config.py (GENERATOR/JUDGE)"
    )


def _build_messages(
    prompt: str, system: str | None, schema: dict | None = None
) -> list[dict[str, str]]:
    system_content = system or ""
    if schema is not None:
        instruction = (
            "Respond with a single JSON object matching this schema and no other text: "
            f"{json.dumps(schema)}"
        )
        system_content = f"{system_content}\n{instruction}".strip()

    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": prompt})
    return messages


def _estimate_message_tokens(messages: list[dict[str, str]]) -> int:
    return sum(estimate_tokens(m["content"]) for m in messages)


def _temperature_kwargs(provider_config: ProviderConfig) -> dict:
    """Only pass `temperature` when the config sets one explicitly, so a
    provider config that leaves it unset (e.g. the not-yet-implemented judge)
    keeps the provider's own default rather than us inventing one here.
    """
    if provider_config.temperature is None:
        return {}
    return {"temperature": provider_config.temperature}
