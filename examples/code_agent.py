"""Code Agent example (Phase 4).

Demonstrates a real Agent driven by LiteLLM that can:
1. Read a file using ReadFileTool
2. Run shell commands using BashTool
3. Edit files using EditFileTool

Prerequisites:
  - Copy .env.example to .env and set ANTHROPIC_API_KEY
  - pip install litellm python-dotenv
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.declarative import AgentFramework
from core.state.models import AgentConfig
from core.state.sqlite_store import SQLiteStateStore
from tools.code.file_tools import EditFileTool, ReadFileTool, WriteFileTool
from tools.code.search_tools import GlobTool, GrepTool
from tools.code.shell_tools import BashTool
from tools.registry import ToolRegistry


async def demo_read_and_run_tests(state_store: SQLiteStateStore) -> None:
    """Scenario 1: Agent reads a file and runs tests."""
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(BashTool())

    config = AgentConfig(
        agent_id="code-agent-1",
        tools=["read_file", "bash"],
        safety_max_steps=5,
    )

    registry.grant_permissions("code-agent-1", ["read_file", "bash"])

    framework = AgentFramework(state_store=state_store)
    framework._tool_registry = registry
    result = await framework.run(
        goal="Read the file 'examples/code_agent.py' and report its line count using bash (wc -l).",
        agent_config=config,
    )

    for step in result.steps:
        sys.stdout.write(
            f"  Step {step.step_number}: {step.action} "
            f"(tokens={step.llm_tokens_used}, latency={step.llm_latency_ms:.1f}ms)\n"
        )
    sys.stdout.write(f"  Status: {result.status}, Steps: {result.total_steps}\n")


async def demo_write_and_edit(state_store: SQLiteStateStore) -> None:
    """Scenario 2: Agent writes a file then edits it."""
    registry = ToolRegistry()
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(ReadFileTool())

    config = AgentConfig(
        agent_id="code-agent-2",
        tools=["write_file", "edit_file", "read_file"],
        safety_max_steps=5,
    )

    registry.grant_permissions("code-agent-2", ["write_file", "edit_file", "read_file"])

    framework = AgentFramework(state_store=state_store)
    framework._tool_registry = registry
    result = await framework.run(
        goal=(
            "Create a file at '/tmp/phase4_demo.txt' with content 'version = 1', "
            "then edit it to change 'version = 1' to 'version = 2', "
            "then read it back to confirm the change."
        ),
        agent_config=config,
    )

    sys.stdout.write(f"  Status: {result.status}, Steps: {result.total_steps}\n")


async def demo_search_codebase(state_store: SQLiteStateStore) -> None:
    """Scenario 3: Agent searches the codebase."""
    registry = ToolRegistry()
    registry.register(GlobTool())
    registry.register(GrepTool())

    config = AgentConfig(
        agent_id="code-agent-3",
        tools=["glob", "grep"],
        safety_max_steps=3,
    )

    registry.grant_permissions("code-agent-3", ["glob", "grep"])

    framework = AgentFramework(state_store=state_store)
    framework._tool_registry = registry
    result = await framework.run(
        goal="Find all Python test files in the 'tests/' directory and list them.",
        agent_config=config,
    )

    sys.stdout.write(f"  Status: {result.status}, Steps: {result.total_steps}\n")


async def main() -> None:
    sys.stdout.write("Phase 4: Code Agent Demo\n")
    sys.stdout.write("=" * 50 + "\n")
    sys.stdout.write("NOTE: Set ANTHROPIC_API_KEY in .env to use real LLM.\n")
    sys.stdout.write("      Without it, the agent uses mock LLM responses.\n\n")

    state_store = SQLiteStateStore(":memory:")
    await state_store.initialize()

    sys.stdout.write("--- Scenario 1: Read file + run tests ---\n")
    await demo_read_and_run_tests(state_store)

    sys.stdout.write("\n--- Scenario 2: Write + Edit file ---\n")
    await demo_write_and_edit(state_store)

    sys.stdout.write("\n--- Scenario 3: Glob + Grep search ---\n")
    await demo_search_codebase(state_store)

    sys.stdout.write("\nPhase 4 demo complete.\n")


if __name__ == "__main__":
    asyncio.run(main())
