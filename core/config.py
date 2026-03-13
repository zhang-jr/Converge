"""Framework configuration from environment variables (.env support)."""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load .env file if python-dotenv is installed.

    Uses override=False so existing environment variables take priority
    over values defined in the .env file.
    """
    try:
        from dotenv import load_dotenv

        env_file = Path(__file__).parent.parent / ".env"
        load_dotenv(env_file, override=False)
    except ImportError:
        pass  # python-dotenv is optional


_load_dotenv()

LLM_MODEL: str = os.environ.get("LITELLM_MODEL", "claude-sonnet-4-6")
LLM_API_BASE: str | None = os.environ.get("LITELLM_API_BASE") or None
LLM_MAX_TOKENS: int = int(os.environ.get("LITELLM_MAX_TOKENS", "4096"))
LLM_TEMPERATURE: float = float(os.environ.get("LITELLM_TEMPERATURE", "0.7"))
BASH_TOOL_DEFAULT_TIMEOUT: int = int(os.environ.get("BASH_TOOL_DEFAULT_TIMEOUT", "30"))
