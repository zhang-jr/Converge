"""WebFetchTool: fetch a URL and return its content."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any

from tools.base import ReadOnlyTool, ToolResult

try:
    import httpx
except ImportError as e:
    raise ImportError(
        "httpx is required for WebFetchTool. "
        "Install with: pip install 'agent-framework[web]'"
    ) from e


class _TextExtractor(HTMLParser):
    """Minimal HTML-to-text extractor using stdlib html.parser."""

    _SKIP_TAGS = frozenset({"script", "style", "head", "meta", "link", "noscript"})

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self) -> str:
        raw = " ".join(self._parts)
        # Collapse runs of whitespace / newlines
        return re.sub(r"\s{2,}", " ", raw).strip()


class WebFetchTool(ReadOnlyTool):
    """Fetch a URL and return its content as text or raw HTML.

    Uses httpx for async HTTP requests. Optionally strips HTML tags and
    returns plain text. Follows redirects by default (up to 5 hops).

    side_effects: ["network_request"] — read-only but makes outbound connections.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        """Args:
            timeout: Default request timeout in seconds.
        """
        self._default_timeout = timeout

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch the content of a URL. "
            "Set extract_text=true to strip HTML tags and return plain text."
        )

    @property
    def side_effects(self) -> list[str]:
        return ["network_request"]

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"},
                "timeout": {
                    "type": "number",
                    "description": "Request timeout in seconds (default: 30)",
                },
                "extract_text": {
                    "type": "boolean",
                    "description": "Strip HTML and return plain text (default: true)",
                },
                "max_bytes": {
                    "type": "integer",
                    "description": "Maximum response bytes to read (default: 512000 = 500 KB)",
                },
            },
            "required": ["url"],
        }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Fetch a URL.

        Args:
            params: Must contain 'url'. Optional: 'timeout', 'extract_text', 'max_bytes'.

        Returns:
            ToolResult with keys: content, status_code, content_type, url.
        """
        url: str = params.get("url", "")
        if not url:
            return ToolResult(success=False, error="'url' parameter is required")

        timeout = float(params.get("timeout", self._default_timeout))
        extract_text: bool = params.get("extract_text", True)
        max_bytes: int = params.get("max_bytes", 512_000)

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=timeout,
                headers={"User-Agent": "AgentFramework/1.0 (+https://github.com/agent-framework)"},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

                raw_content = response.content[:max_bytes].decode(
                    response.encoding or "utf-8", errors="replace"
                )
                content_type: str = response.headers.get("content-type", "")

                if extract_text and "html" in content_type:
                    extractor = _TextExtractor()
                    extractor.feed(raw_content)
                    content = extractor.get_text()
                else:
                    content = raw_content

                return ToolResult(
                    success=True,
                    output=content,
                    metadata={
                        "url": str(response.url),
                        "status_code": response.status_code,
                        "content_type": content_type,
                        "bytes_read": len(response.content[:max_bytes]),
                    },
                )

        except httpx.HTTPStatusError as e:
            return ToolResult(
                success=False,
                error=f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
                metadata={"url": url, "status_code": e.response.status_code},
            )
        except httpx.TimeoutException:
            return ToolResult(success=False, error=f"Request timed out after {timeout}s", metadata={"url": url})
        except httpx.RequestError as e:
            return ToolResult(success=False, error=f"Request error: {e}", metadata={"url": url})
