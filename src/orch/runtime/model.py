"""Meta Model API (muse-spark) as a Pydantic AI model, plus per-run cost.

The API is OpenAI-compatible (Chat Completions and Responses) at https://api.meta.ai/v1.
The key is read from MODEL_API_KEY (environment or a gitignored .env).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import SecretStr
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RunUsage
from pydantic_settings import BaseSettings, SettingsConfigDict

# Contributor mode is its own model id: cheaper, but Meta may train on prompts and outputs.
# Use "muse-spark-1.3" (standard) for anything a bank would actually send.
DEFAULT_MODEL = "muse-spark-1.3-contributor"
Api = Literal["chat", "responses"]


class ModelEnv(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    model_api_key: SecretStr | None = None
    model_base_url: str = "https://api.meta.ai/v1"


@dataclass(frozen=True)
class Price:
    """USD per million tokens. Reasoning tokens are billed as output."""

    input: float
    output: float
    cached_input: float


# Pay-as-you-go prices from dev.meta.ai/docs/pricing-rate-limits (Oct 2026). Rate limits are
# per team: standard 3,000 RPM / 4M TPM, contributor 150 RPM / 3M TPM.
PRICES: dict[str, Price] = {
    "muse-spark-1.3": Price(input=1.25, output=4.25, cached_input=0.15),
    "muse-spark-1.3-contributor": Price(input=0.10, output=0.20, cached_input=0.002),
}


def meta_model(name: str = DEFAULT_MODEL, *, api: Api = "chat") -> Model:
    env = ModelEnv()
    if env.model_api_key is None:
        raise RuntimeError("MODEL_API_KEY is not set (environment or .env)")
    provider = OpenAIProvider(
        base_url=env.model_base_url, api_key=env.model_api_key.get_secret_value()
    )
    if api == "responses":
        return OpenAIResponsesModel(name, provider=provider)
    return OpenAIChatModel(name, provider=provider)


def cost_usd(usage: RunUsage, model: str = DEFAULT_MODEL) -> float:
    """Cost of a run. input_tokens includes cache reads, output_tokens includes reasoning."""
    p = PRICES[model]
    cached = usage.cache_read_tokens
    fresh = max(usage.input_tokens - cached, 0)
    return (fresh * p.input + cached * p.cached_input + usage.output_tokens * p.output) / 1e6
