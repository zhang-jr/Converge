"""WebSearchTool: abstract base + TavilySearchTool implementation."""

from __future__ import annotations

import os
from abc import abstractmethod
from typing import Any, Literal

from tools.base import ReadOnlyTool, ToolResult

try:
    import httpx
except ImportError as e:
    raise ImportError(
        "httpx is required for web search tools. "
        "Install with: pip install 'agent-framework[web]'"
    ) from e


class WebSearchTool(ReadOnlyTool):
    """Abstract base for web search tools.

    Subclasses implement _search() with their specific backend.
    All search tools share the same parameter schema and result format.

    Result format (list of dicts):
        [{"title": str, "url": str, "content": str, "score": float | None}]
    """

    @property
    def side_effects(self) -> list[str]:
        return ["network_request"]

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            "required": ["query"],
        }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute a web search.

        Args:
            params: Must contain 'query'. Optional: 'max_results'.

        Returns:
            ToolResult with output = list of result dicts.
        """
        query: str = params.get("query", "")
        if not query:
            return ToolResult(success=False, error="'query' parameter is required")

        max_results: int = min(int(params.get("max_results", 5)), 20)
        return await self._search(query=query, max_results=max_results, params=params)

    @abstractmethod
    async def _search(
        self, query: str, max_results: int, params: dict[str, Any]
    ) -> ToolResult:
        """Backend-specific search implementation.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return.
            params: Full params dict for backend-specific options.

        Returns:
            ToolResult with output = list[dict] results.
        """
        ...


class TavilySearchTool(WebSearchTool):
    """Web search via the Tavily API.

    Requires a Tavily API key — set via constructor or TAVILY_API_KEY env var.
    Sign up at https://tavily.com to get a free API key.

    Extra parameters:
        search_depth: "basic" (default, faster) or "advanced" (deeper, slower).
        include_domains: list of domains to restrict search to.
        exclude_domains: list of domains to exclude from results.
    """

    _API_URL = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Args:
            api_key: Tavily API key. Falls back to TAVILY_API_KEY env var.
            timeout: HTTP request timeout in seconds.
        """
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web using Tavily. "
            "Returns titles, URLs, and content snippets ranked by relevance."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        base = super().parameters_schema
        base["properties"].update({
            "search_depth": {
                "type": "string",
                "enum": ["basic", "advanced"],
                "description": "'basic' is faster; 'advanced' gives deeper results (default: basic)",
            },
            "include_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Restrict results to these domains",
            },
            "exclude_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exclude these domains from results",
            },
        })
        return base

    async def _search(
        self, query: str, max_results: int, params: dict[str, Any]
    ) -> ToolResult:
        if not self._api_key:
            return ToolResult(
                success=False,
                error=(
                    "No Tavily API key configured. "
                    "Pass api_key= to TavilySearchTool() or set TAVILY_API_KEY env var."
                ),
            )

        search_depth: Literal["basic", "advanced"] = params.get("search_depth", "basic")  # type: ignore[assignment]
        payload: dict[str, Any] = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
        }
        if include := params.get("include_domains"):
            payload["include_domains"] = include
        if exclude := params.get("exclude_domains"):
            payload["exclude_domains"] = exclude

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._API_URL, json=payload)
                response.raise_for_status()
                data = response.json()

            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score"),
                }
                for r in data.get("results", [])
            ]

            return ToolResult(
                success=True,
                output=results,
                metadata={
                    "query": query,
                    "result_count": len(results),
                    "search_depth": search_depth,
                    "response_time_ms": data.get("response_time"),
                },
            )

        except httpx.HTTPStatusError as e:
            # Tavily returns 401 for bad key, 429 for rate limit
            return ToolResult(
                success=False,
                error=f"Tavily API error HTTP {e.response.status_code}: {e.response.text}",
                metadata={"query": query},
            )
        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                error=f"Tavily request timed out after {self._timeout}s",
                metadata={"query": query},
            )
        except httpx.RequestError as e:
            return ToolResult(success=False, error=f"Network error: {e}", metadata={"query": query})
        except (KeyError, ValueError) as e:
            return ToolResult(success=False, error=f"Unexpected Tavily response format: {e}", metadata={"query": query})
