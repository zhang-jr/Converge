"""Tests for WebFetchTool and TavilySearchTool.

All HTTP calls are mocked — no real network requests.
Uses unittest.mock to patch httpx.AsyncClient.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.web.fetch_tool import WebFetchTool, _TextExtractor
from tools.web.search_tool import TavilySearchTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    *,
    status_code: int = 200,
    text: str = "",
    content_type: str = "text/html; charset=utf-8",
    encoding: str = "utf-8",
    url: str = "https://example.com",
) -> MagicMock:
    """Build a fake httpx.Response mock."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.url = url
    resp.encoding = encoding
    resp.headers = {"content-type": content_type}
    resp.content = text.encode(encoding)
    resp.text = text
    resp.raise_for_status = MagicMock()  # does nothing by default
    return resp


def _make_http_status_error(status_code: int, reason: str = "Error") -> MagicMock:
    import httpx

    resp = MagicMock()
    resp.status_code = status_code
    resp.reason_phrase = reason
    resp.text = reason
    return httpx.HTTPStatusError(reason, request=MagicMock(), response=resp)


def _mock_client(response: MagicMock) -> MagicMock:
    """Return a mock AsyncClient context manager that yields a client."""
    client = AsyncMock()
    client.get = AsyncMock(return_value=response)
    client.post = AsyncMock(return_value=response)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm, client


# =============================================================================
# _TextExtractor
# =============================================================================


class TestTextExtractor:
    def test_strips_tags(self) -> None:
        extractor = _TextExtractor()
        extractor.feed("<p>Hello <b>world</b></p>")
        assert extractor.get_text() == "Hello world"

    def test_skips_script_content(self) -> None:
        extractor = _TextExtractor()
        extractor.feed("<p>visible</p><script>alert('x')</script><p>also visible</p>")
        text = extractor.get_text()
        assert "visible" in text
        assert "alert" not in text

    def test_skips_style_content(self) -> None:
        extractor = _TextExtractor()
        extractor.feed("<style>.foo { color: red; }</style><p>content</p>")
        text = extractor.get_text()
        assert "content" in text
        assert "color" not in text

    def test_collapses_whitespace(self) -> None:
        extractor = _TextExtractor()
        extractor.feed("<p>  lots   of   spaces  </p>")
        assert "  " not in extractor.get_text()

    def test_empty_input(self) -> None:
        extractor = _TextExtractor()
        extractor.feed("")
        assert extractor.get_text() == ""


# =============================================================================
# WebFetchTool
# =============================================================================


class TestWebFetchTool:
    @pytest.fixture
    def tool(self) -> WebFetchTool:
        return WebFetchTool()

    async def test_fetch_html_extracts_text(self, tool: WebFetchTool) -> None:
        html = "<html><body><h1>Title</h1><p>Paragraph text.</p></body></html>"
        resp = _make_response(text=html, content_type="text/html")
        cm, _ = _mock_client(resp)

        with patch("tools.web.fetch_tool.httpx.AsyncClient", return_value=cm):
            result = await tool.execute({"url": "https://example.com", "extract_text": True})

        assert result.success
        assert "Title" in result.output
        assert "Paragraph text" in result.output
        assert "<h1>" not in result.output

    async def test_fetch_raw_html_when_extract_false(self, tool: WebFetchTool) -> None:
        html = "<p>raw</p>"
        resp = _make_response(text=html, content_type="text/html")
        cm, _ = _mock_client(resp)

        with patch("tools.web.fetch_tool.httpx.AsyncClient", return_value=cm):
            result = await tool.execute({"url": "https://example.com", "extract_text": False})

        assert result.success
        assert "<p>" in result.output

    async def test_non_html_content_returned_raw(self, tool: WebFetchTool) -> None:
        data = '{"key": "value"}'
        resp = _make_response(text=data, content_type="application/json")
        cm, _ = _mock_client(resp)

        with patch("tools.web.fetch_tool.httpx.AsyncClient", return_value=cm):
            result = await tool.execute({"url": "https://api.example.com/data", "extract_text": True})

        assert result.success
        assert result.output == data

    async def test_missing_url_param(self, tool: WebFetchTool) -> None:
        result = await tool.execute({})
        assert not result.success
        assert "url" in result.error

    async def test_http_error_returns_failure(self, tool: WebFetchTool) -> None:
        import httpx

        cm, client = _mock_client(MagicMock())
        client.get.side_effect = _make_http_status_error(404, "Not Found")

        with patch("tools.web.fetch_tool.httpx.AsyncClient", return_value=cm):
            result = await tool.execute({"url": "https://example.com/missing"})

        assert not result.success
        assert "404" in result.error
        assert result.metadata["status_code"] == 404

    async def test_timeout_returns_failure(self, tool: WebFetchTool) -> None:
        import httpx

        cm, client = _mock_client(MagicMock())
        client.get.side_effect = httpx.TimeoutException("timed out")

        with patch("tools.web.fetch_tool.httpx.AsyncClient", return_value=cm):
            result = await tool.execute({"url": "https://example.com", "timeout": 5})

        assert not result.success
        assert "timed out" in result.error.lower()

    async def test_request_error_returns_failure(self, tool: WebFetchTool) -> None:
        import httpx

        cm, client = _mock_client(MagicMock())
        client.get.side_effect = httpx.RequestError("connection refused")

        with patch("tools.web.fetch_tool.httpx.AsyncClient", return_value=cm):
            result = await tool.execute({"url": "https://unreachable.example.com"})

        assert not result.success
        assert "connection refused" in result.error

    async def test_metadata_contains_status_and_url(self, tool: WebFetchTool) -> None:
        resp = _make_response(text="<p>ok</p>", url="https://example.com/page")
        cm, _ = _mock_client(resp)

        with patch("tools.web.fetch_tool.httpx.AsyncClient", return_value=cm):
            result = await tool.execute({"url": "https://example.com/page"})

        assert result.success
        assert result.metadata["status_code"] == 200
        assert "example.com" in result.metadata["url"]

    async def test_max_bytes_limits_content(self, tool: WebFetchTool) -> None:
        long_text = "A" * 10_000
        resp = _make_response(text=long_text, content_type="text/plain")
        cm, _ = _mock_client(resp)

        with patch("tools.web.fetch_tool.httpx.AsyncClient", return_value=cm):
            result = await tool.execute({"url": "https://example.com", "max_bytes": 100, "extract_text": False})

        assert result.success
        assert result.metadata["bytes_read"] == 100

    def test_risk_level_is_low(self, tool: WebFetchTool) -> None:
        assert tool.risk_level == "low"
        assert tool.idempotent is True
        assert tool.reversible is True

    def test_side_effects_declared(self, tool: WebFetchTool) -> None:
        assert "network_request" in tool.side_effects

    def test_tool_name(self, tool: WebFetchTool) -> None:
        assert tool.name == "web_fetch"


# =============================================================================
# TavilySearchTool
# =============================================================================


_TAVILY_SUCCESS = {
    "results": [
        {"title": "Result One", "url": "https://one.com", "content": "snippet one", "score": 0.95},
        {"title": "Result Two", "url": "https://two.com", "content": "snippet two", "score": 0.80},
    ],
    "response_time": 420,
}


class TestTavilySearchTool:
    @pytest.fixture
    def tool(self) -> TavilySearchTool:
        return TavilySearchTool(api_key="tvly-test-key")

    async def test_successful_search(self, tool: TavilySearchTool) -> None:
        resp = _make_response(
            text=json.dumps(_TAVILY_SUCCESS), content_type="application/json"
        )
        resp.json = MagicMock(return_value=_TAVILY_SUCCESS)
        resp.raise_for_status = MagicMock()
        cm, _ = _mock_client(resp)

        with patch("tools.web.search_tool.httpx.AsyncClient", return_value=cm):
            result = await tool.execute({"query": "python asyncio"})

        assert result.success
        assert len(result.output) == 2
        assert result.output[0]["title"] == "Result One"
        assert result.output[0]["score"] == 0.95
        assert result.metadata["result_count"] == 2
        assert result.metadata["query"] == "python asyncio"

    async def test_missing_query_param(self, tool: TavilySearchTool) -> None:
        result = await tool.execute({})
        assert not result.success
        assert "query" in result.error

    async def test_no_api_key_returns_error(self) -> None:
        tool = TavilySearchTool(api_key="")
        result = await tool.execute({"query": "test"})
        assert not result.success
        assert "API key" in result.error

    async def test_no_api_key_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-from-env")
        tool = TavilySearchTool()
        assert tool._api_key == "tvly-from-env"

    async def test_http_401_returns_error(self, tool: TavilySearchTool) -> None:
        cm, client = _mock_client(MagicMock())
        client.post.side_effect = _make_http_status_error(401, "Unauthorized")

        with patch("tools.web.search_tool.httpx.AsyncClient", return_value=cm):
            result = await tool.execute({"query": "test"})

        assert not result.success
        assert "401" in result.error

    async def test_http_429_returns_error(self, tool: TavilySearchTool) -> None:
        cm, client = _mock_client(MagicMock())
        client.post.side_effect = _make_http_status_error(429, "Too Many Requests")

        with patch("tools.web.search_tool.httpx.AsyncClient", return_value=cm):
            result = await tool.execute({"query": "test"})

        assert not result.success
        assert "429" in result.error

    async def test_timeout_returns_error(self, tool: TavilySearchTool) -> None:
        import httpx

        cm, client = _mock_client(MagicMock())
        client.post.side_effect = httpx.TimeoutException("timed out")

        with patch("tools.web.search_tool.httpx.AsyncClient", return_value=cm):
            result = await tool.execute({"query": "test"})

        assert not result.success
        assert "timed out" in result.error.lower()

    async def test_max_results_capped_at_20(self, tool: TavilySearchTool) -> None:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={"results": [], "response_time": 0})
        cm, client = _mock_client(resp)

        with patch("tools.web.search_tool.httpx.AsyncClient", return_value=cm):
            await tool.execute({"query": "test", "max_results": 999})

        call_kwargs = client.post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
        assert payload["max_results"] == 20

    async def test_search_depth_forwarded(self, tool: TavilySearchTool) -> None:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={"results": [], "response_time": 0})
        cm, client = _mock_client(resp)

        with patch("tools.web.search_tool.httpx.AsyncClient", return_value=cm):
            await tool.execute({"query": "test", "search_depth": "advanced"})

        payload = client.post.call_args[1]["json"]
        assert payload["search_depth"] == "advanced"

    async def test_include_domains_forwarded(self, tool: TavilySearchTool) -> None:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={"results": [], "response_time": 0})
        cm, client = _mock_client(resp)

        with patch("tools.web.search_tool.httpx.AsyncClient", return_value=cm):
            await tool.execute({"query": "test", "include_domains": ["github.com"]})

        payload = client.post.call_args[1]["json"]
        assert payload["include_domains"] == ["github.com"]

    async def test_exclude_domains_forwarded(self, tool: TavilySearchTool) -> None:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={"results": [], "response_time": 0})
        cm, client = _mock_client(resp)

        with patch("tools.web.search_tool.httpx.AsyncClient", return_value=cm):
            await tool.execute({"query": "test", "exclude_domains": ["spam.com"]})

        payload = client.post.call_args[1]["json"]
        assert payload["exclude_domains"] == ["spam.com"]

    async def test_empty_results_list(self, tool: TavilySearchTool) -> None:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={"results": [], "response_time": 10})
        cm, _ = _mock_client(resp)

        with patch("tools.web.search_tool.httpx.AsyncClient", return_value=cm):
            result = await tool.execute({"query": "very obscure query xyzzy"})

        assert result.success
        assert result.output == []
        assert result.metadata["result_count"] == 0

    def test_risk_level_is_low(self, tool: TavilySearchTool) -> None:
        assert tool.risk_level == "low"
        assert tool.idempotent is True
        assert tool.reversible is True

    def test_side_effects_declared(self, tool: TavilySearchTool) -> None:
        assert "network_request" in tool.side_effects

    def test_tool_name(self, tool: TavilySearchTool) -> None:
        assert tool.name == "web_search"
