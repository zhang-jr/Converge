"""Web Research Agent example.

Demonstrates an Agent that combines web search, web fetching, and file I/O
to perform real research tasks. All network calls require live API keys.

Prerequisites:
  .env 中需要设置：
    ANTHROPIC_API_KEY=your_key   # LLM 调用
    TAVILY_API_KEY=your_key      # Web 搜索（https://tavily.com 免费注册）

  安装依赖：
    pip install 'agent-framework[web]'   # httpx
    pip install litellm python-dotenv

Scenarios:
  1. Research & Save  ── 搜索主题，抓取最相关页面，将摘要写入本地文件
  2. Fetch & Extract  ── 直接抓取指定 URL，提取纯文本，追加到报告
  3. Compare & Report ── 搜索两个技术方案，综合多页内容，生成对比报告
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.declarative import AgentFramework
from core.state.models import AgentConfig
from tools.code.file_tools import ReadFileTool, WriteFileTool
from tools.registry import ToolRegistry
from tools.web.fetch_tool import WebFetchTool
from tools.web.search_tool import TavilySearchTool


# ---------------------------------------------------------------------------
# Shared output directory (temp, cleaned up after demo)
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(tempfile.gettempdir()) / "agent_web_research"
OUTPUT_DIR.mkdir(exist_ok=True)


def _registry(*tool_names: str) -> tuple[ToolRegistry, list[str]]:
    """Build a ToolRegistry with the requested tools."""
    all_tools = {
        "web_search": TavilySearchTool(),   # reads TAVILY_API_KEY from env
        "web_fetch":  WebFetchTool(),
        "write_file": WriteFileTool(),
        "read_file":  ReadFileTool(),
    }
    reg = ToolRegistry()
    names: list[str] = []
    for n in tool_names:
        reg.register(all_tools[n])
        names.append(n)
    return reg, names


def _print_steps(result) -> None:  # type: ignore[no-untyped-def]
    for step in result.steps:
        tool_info = f" [{step.tool_name}]" if step.tool_name else ""
        sys.stdout.write(
            f"    step {step.step_number:02d}{tool_info}: "
            f"{step.action[:80]}...\n" if len(step.action) > 80
            else f"    step {step.step_number:02d}{tool_info}: {step.action}\n"
        )
    sys.stdout.write(f"  → status={result.status}  steps={result.total_steps}\n")


# ---------------------------------------------------------------------------
# Scenario 1: Research & Save
# Agent searches for a topic, fetches the top result, writes a summary file.
# ---------------------------------------------------------------------------

async def scenario_research_and_save() -> None:
    """Search → fetch top result → write summary to file."""
    out_file = OUTPUT_DIR / "asyncio_summary.md"

    reg, tools = _registry("web_search", "web_fetch", "write_file")
    config = AgentConfig(
        agent_id="web-agent-1",
        tools=tools,
        safety_max_steps=8,
    )
    reg.grant_permissions("web-agent-1", tools)

    fw = AgentFramework(db_path=":memory:")
    fw._tool_registry = reg

    goal = (
        f"Research 'Python asyncio best practices 2024':\n"
        f"1. Use web_search to find the top 3 results.\n"
        f"2. Use web_fetch (extract_text=true) to read the most relevant page.\n"
        f"3. Write a concise Markdown summary (key points only, ≤300 words) "
        f"   to '{out_file}'.\n"
        f"The file must exist when you are done."
    )

    result = await fw.run(goal=goal, agent_config=config)
    _print_steps(result)

    if out_file.exists():
        preview = out_file.read_text(encoding="utf-8")[:400]
        sys.stdout.write(f"\n  File preview ({out_file.name}):\n")
        for line in preview.splitlines():
            sys.stdout.write(f"    {line}\n")
        sys.stdout.write("    ...\n" if len(out_file.read_text()) > 400 else "")


# ---------------------------------------------------------------------------
# Scenario 2: Fetch & Extract
# Agent fetches a specific URL and appends structured data to a report.
# ---------------------------------------------------------------------------

async def scenario_fetch_and_extract() -> None:
    """Fetch a known URL → extract key info → append to report file."""
    report_file = OUTPUT_DIR / "tech_report.md"

    reg, tools = _registry("web_fetch", "write_file", "read_file")
    config = AgentConfig(
        agent_id="web-agent-2",
        tools=tools,
        safety_max_steps=6,
    )
    reg.grant_permissions("web-agent-2", tools)

    fw = AgentFramework(db_path=":memory:")
    fw._tool_registry = reg

    goal = (
        f"1. Use web_fetch (extract_text=true, max_bytes=80000) to fetch "
        f"   'https://docs.python.org/3/whatsnew/3.12.html'.\n"
        f"2. From the fetched content, identify the 5 most important new features "
        f"   in Python 3.12.\n"
        f"3. Write them as a numbered Markdown list to '{report_file}'.\n"
        f"   Start the file with '# Python 3.12 Key Features\\n\\n'."
    )

    result = await fw.run(goal=goal, agent_config=config)
    _print_steps(result)

    if report_file.exists():
        sys.stdout.write(f"\n  File content ({report_file.name}):\n")
        for line in report_file.read_text(encoding="utf-8").splitlines():
            sys.stdout.write(f"    {line}\n")


# ---------------------------------------------------------------------------
# Scenario 3: Compare & Report
# Agent searches two competing technologies and produces a comparison report.
# ---------------------------------------------------------------------------

async def scenario_compare_and_report() -> None:
    """Search two topics → fetch pages → write comparison report."""
    report_file = OUTPUT_DIR / "framework_comparison.md"

    reg, tools = _registry("web_search", "web_fetch", "write_file", "read_file")
    config = AgentConfig(
        agent_id="web-agent-3",
        tools=tools,
        safety_max_steps=12,
    )
    reg.grant_permissions("web-agent-3", tools)

    fw = AgentFramework(db_path=":memory:")
    fw._tool_registry = reg

    goal = (
        f"Compare FastAPI and Django for building REST APIs in 2024:\n"
        f"1. Use web_search (max_results=3) to search 'FastAPI REST API performance 2024'.\n"
        f"2. Use web_search (max_results=3) to search 'Django REST framework performance 2024'.\n"
        f"3. Fetch one relevant page for each (web_fetch, extract_text=true, max_bytes=60000).\n"
        f"4. Write a structured Markdown comparison to '{report_file}' with sections:\n"
        f"   ## Performance, ## Developer Experience, ## Ecosystem, ## Verdict\n"
        f"   Keep each section ≤100 words. Be objective."
    )

    result = await fw.run(goal=goal, agent_config=config)
    _print_steps(result)

    if report_file.exists():
        preview = report_file.read_text(encoding="utf-8")[:600]
        sys.stdout.write(f"\n  File preview ({report_file.name}):\n")
        for line in preview.splitlines():
            sys.stdout.write(f"    {line}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    sys.stdout.write("Web Research Agent Demo\n")
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write("Required env vars: ANTHROPIC_API_KEY, TAVILY_API_KEY\n")
    sys.stdout.write(f"Output directory : {OUTPUT_DIR}\n\n")

    sys.stdout.write("--- Scenario 1: Research & Save (search → fetch → write) ---\n")
    await scenario_research_and_save()

    sys.stdout.write("\n--- Scenario 2: Fetch & Extract (fetch docs → write report) ---\n")
    await scenario_fetch_and_extract()

    sys.stdout.write("\n--- Scenario 3: Compare & Report (multi-search → compare) ---\n")
    await scenario_compare_and_report()

    sys.stdout.write(f"\nAll output files written to: {OUTPUT_DIR}\n")
    for f in sorted(OUTPUT_DIR.iterdir()):
        sys.stdout.write(f"  {f.name}  ({f.stat().st_size} bytes)\n")


if __name__ == "__main__":
    asyncio.run(main())
