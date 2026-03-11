"""Unit tests for HumanInterventionHandler implementations."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from core.runtime.human_intervention import (
    CallbackHumanInterventionHandler,
    CLIHumanInterventionHandler,
)
from core.state.models import DesiredState, HumanDecision, LoopContext


def _ctx(agent_id: str = "test-agent", step: int = 1) -> LoopContext:
    return LoopContext(
        desired_state=DesiredState(goal="test goal"),
        current_step=step,
        agent_id=agent_id,
    )


class TestCallbackHumanInterventionHandler:
    """Tests for CallbackHumanInterventionHandler."""

    async def test_approve_via_callback(self) -> None:
        async def approve(reason, ctx, pending):
            return HumanDecision(approved=True, feedback="looks good")

        handler = CallbackHumanInterventionHandler(approve)
        decision = await handler.request_approval("Low confidence", _ctx())
        assert decision.approved is True
        assert decision.feedback == "looks good"

    async def test_reject_via_callback(self) -> None:
        async def reject(reason, ctx, pending):
            return HumanDecision(approved=False, feedback="too risky")

        handler = CallbackHumanInterventionHandler(reject)
        decision = await handler.request_approval("High risk tool", _ctx())
        assert decision.approved is False
        assert decision.feedback == "too risky"

    async def test_callback_receives_correct_args(self) -> None:
        received: dict = {}

        async def capture(reason, ctx, pending):
            received["reason"] = reason
            received["agent_id"] = ctx.agent_id
            received["pending"] = pending
            return HumanDecision(approved=True)

        handler = CallbackHumanInterventionHandler(capture)
        ctx = _ctx(agent_id="my-agent")
        pending = {"tool": "delete_file", "params": {"path": "/etc/passwd"}}
        await handler.request_approval("Dangerous operation", ctx, pending)

        assert received["reason"] == "Dangerous operation"
        assert received["agent_id"] == "my-agent"
        assert received["pending"] == pending

    async def test_pending_action_none_by_default(self) -> None:
        received_pending: list = []

        async def capture(reason, ctx, pending):
            received_pending.append(pending)
            return HumanDecision(approved=True)

        handler = CallbackHumanInterventionHandler(capture)
        await handler.request_approval("reason", _ctx())
        assert received_pending[0] is None


class TestCLIHumanInterventionHandler:
    """Tests for CLIHumanInterventionHandler using mocked input()."""

    async def test_cli_approve_y(self) -> None:
        handler = CLIHumanInterventionHandler()
        with patch("builtins.input", return_value="y"):
            decision = await handler.request_approval("needs approval", _ctx())
        assert decision.approved is True
        assert decision.decision_by == "cli_human"

    async def test_cli_approve_yes(self) -> None:
        handler = CLIHumanInterventionHandler()
        with patch("builtins.input", return_value="yes"):
            decision = await handler.request_approval("needs approval", _ctx())
        assert decision.approved is True

    async def test_cli_reject_n(self) -> None:
        handler = CLIHumanInterventionHandler()
        # First input: "n", second: feedback
        with patch("builtins.input", side_effect=["n", "not safe"]):
            decision = await handler.request_approval("needs approval", _ctx())
        assert decision.approved is False
        assert decision.feedback == "not safe"

    async def test_cli_reject_empty_input(self) -> None:
        """Empty input (just Enter) should default to reject."""
        handler = CLIHumanInterventionHandler()
        with patch("builtins.input", side_effect=["", ""]):
            decision = await handler.request_approval("needs approval", _ctx())
        assert decision.approved is False

    async def test_cli_displays_pending_action(self, capsys) -> None:
        handler = CLIHumanInterventionHandler(prompt_prefix="[TEST]")
        with patch("builtins.input", return_value="y"):
            await handler.request_approval(
                "High risk",
                _ctx(),
                pending_action={"tool": "nuke_db"},
            )
        captured = capsys.readouterr()
        assert "nuke_db" in captured.out
        assert "[TEST]" in captured.out


class TestHighRiskToolGate:
    """Integration tests: high-risk tool gate in ReconcileLoop._execute_tool."""

    async def test_high_risk_tool_requires_approval(self) -> None:
        """_execute_tool should call on_human_intervention_needed for high-risk tools."""
        from core.state.models import DesiredState
        from core.state.sqlite_store import SQLiteStateStore
        from core.runtime.reconcile_loop import ReconcileLoop
        from tools.base import ToolBase, ToolResult

        class DangerousTool(ToolBase):
            @property
            def name(self): return "dangerous"
            @property
            def description(self): return "danger"
            @property
            def side_effects(self): return ["deletes_all"]
            @property
            def reversible(self): return False
            @property
            def risk_level(self): return "high"
            @property
            def idempotent(self): return False
            async def execute(self, params): return ToolResult(success=True, output="done")

        approved_flag: list[bool] = []

        async def reject_handler(reason, ctx, pending):
            approved_flag.append(False)
            return HumanDecision(approved=False, feedback="rejected")

        handler = CallbackHumanInterventionHandler(reject_handler)

        store = SQLiteStateStore(":memory:")
        loop = ReconcileLoop.__new__(ReconcileLoop)
        ReconcileLoop.__init__(
            loop,
            state_store=store,
            human_intervention_handler=handler,
        )

        context = LoopContext(
            desired_state=DesiredState(goal="test"),
            current_step=1,
            agent_id="test",
        )

        tool = DangerousTool()
        result = await loop._execute_tool(tool, {}, context=context)

        assert result.success is False
        assert "rejected" in result.error
        assert approved_flag == [False]  # handler was called once

    async def test_low_risk_tool_skips_approval(self) -> None:
        """_execute_tool should NOT call intervention for low-risk tools."""
        from core.state.models import DesiredState
        from core.state.sqlite_store import SQLiteStateStore
        from core.runtime.reconcile_loop import ReconcileLoop
        from tools.base import ReadOnlyTool, ToolResult

        class SafeTool(ReadOnlyTool):
            @property
            def name(self): return "safe"
            @property
            def description(self): return "safe op"
            async def execute(self, params): return ToolResult(success=True, output="ok")

        calls: list[str] = []

        async def track_handler(reason, ctx, pending):
            calls.append(reason)
            return HumanDecision(approved=True)

        handler = CallbackHumanInterventionHandler(track_handler)
        store = SQLiteStateStore(":memory:")
        loop = ReconcileLoop.__new__(ReconcileLoop)
        ReconcileLoop.__init__(
            loop,
            state_store=store,
            human_intervention_handler=handler,
        )

        context = LoopContext(
            desired_state=DesiredState(goal="test"),
            current_step=1,
            agent_id="test",
        )

        tool = SafeTool()
        result = await loop._execute_tool(tool, {}, context=context)
        assert result.success is True
        assert calls == []  # handler was NOT called
