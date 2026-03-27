"""Framework configuration from environment variables (.env support).

Model resolution order
----------------------
1. Look up model ID in llm_providers.json  →  gets provider's baseUrl + apiKey
2. Fall back to raw LiteLLM model string   →  LiteLLM reads key from env by convention

See llm_providers.json.example for the provider config format.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).parent.parent / ".env", override=False)
    except ImportError:
        pass


def _load_providers() -> dict[str, Any]:
    """Load llm_providers.json from project root."""
    path = Path(__file__).parent.parent / "llm_providers.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


@dataclass(frozen=True)
class ResolvedModel:
    """Fully resolved model config ready to pass to LiteLLM.

    api_key is only populated for custom-baseUrl providers where LiteLLM cannot
    auto-discover the key from the environment. For standard providers (anthropic,
    openai, deepseek, gemini…) api_key is None — LiteLLM reads the key internally
    from the environment, so the key never appears in our kwargs or logs.
    """

    litellm_model: str       # e.g. "claude-sonnet-4-6" or "openai/moonshotai/kimi-k2.5"
    api_base: str | None     # custom endpoint; None for standard providers
    api_key: str | None      # only set when api_base is a custom URL
    max_tokens: int
    temperature: float
    # metadata (informational only, not passed to LiteLLM)
    provider_name: str | None = None
    model_name: str | None = None
    context_window: int | None = None


def resolve_model(model_id: str) -> ResolvedModel:
    """Resolve a model ID to a fully configured ResolvedModel.

    Args:
        model_id: Either a model ID from llm_providers.json (e.g. "moonshotai/kimi-k2.5")
                  or a raw LiteLLM model string (e.g. "claude-sonnet-4-6", "gpt-4o").

    Returns:
        ResolvedModel with all fields populated from provider config + env vars.

    Security note:
        api_key is populated only when api_base is a custom URL (non-standard provider).
        Standard providers (anthropic, openai, etc.) leave api_key=None so LiteLLM
        reads the key internally — the value never surfaces in Python kwargs or traces.
    """
    providers_cfg = _load_providers()
    providers: dict[str, Any] = providers_cfg.get("providers", {})

    # env-var overrides (always highest priority)
    env_max_tokens = os.environ.get("LITELLM_MAX_TOKENS")
    env_temperature = os.environ.get("LITELLM_TEMPERATURE")
    env_api_base = os.environ.get("LITELLM_API_BASE") or None

    # Search providers for matching model
    for provider_name, provider in providers.items():
        for model_cfg in provider.get("models", []):
            if model_cfg.get("id") != model_id:
                continue

            api_type = provider.get("api", "openai-completions")

            # Resolve baseUrl: env override > provider config
            api_base = env_api_base or provider.get("baseUrl") or None

            # Only pass api_key explicitly for custom-baseUrl providers.
            # Standard providers (anthropic / openai / deepseek…): LiteLLM reads
            # ANTHROPIC_API_KEY / OPENAI_API_KEY etc. internally — safer, never
            # leaks into our kwargs or log output.
            api_key: str | None = None
            if api_base:
                api_key = provider.get("apiKey") or None
                if not api_key and (env_name := provider.get("apiKeyEnv")):
                    api_key = os.environ.get(env_name) or None

            # LiteLLM model string
            if api_type == "anthropic":
                litellm_model = model_id
            else:
                # openai-completions with custom base: needs "openai/" prefix
                litellm_model = f"openai/{model_id}" if api_base else model_id

            return ResolvedModel(
                litellm_model=litellm_model,
                api_base=api_base,
                api_key=api_key,
                max_tokens=int(env_max_tokens or model_cfg.get("maxTokens", 4096)),
                temperature=float(env_temperature or 0.7),
                provider_name=provider_name,
                model_name=model_cfg.get("name"),
                context_window=model_cfg.get("contextWindow"),
            )

    # No provider match → treat model_id as raw LiteLLM string (no explicit key)
    return ResolvedModel(
        litellm_model=model_id,
        api_base=env_api_base,
        api_key=None,
        max_tokens=int(env_max_tokens or 4096),
        temperature=float(env_temperature or 0.7),
    )


_load_dotenv()

_DEFAULT_MODEL_ID: str = os.environ.get("LITELLM_MODEL", "claude-sonnet-4-6")
_DEFAULT: ResolvedModel = resolve_model(_DEFAULT_MODEL_ID)

# Module-level constants (used by agent.py and anywhere that needs the default model)
LLM_MODEL: str = _DEFAULT.litellm_model
LLM_API_BASE: str | None = _DEFAULT.api_base
LLM_API_KEY: str | None = _DEFAULT.api_key
LLM_MAX_TOKENS: int = _DEFAULT.max_tokens
LLM_TEMPERATURE: float = _DEFAULT.temperature

BASH_TOOL_DEFAULT_TIMEOUT: int = int(os.environ.get("BASH_TOOL_DEFAULT_TIMEOUT", "30"))
