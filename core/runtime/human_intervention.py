"""Human-in-the-Loop intervention handlers.

Provides an ABC and two concrete implementations:
- CLIHumanInterventionHandler: non-blocking terminal prompts via asyncio.to_thread
- CallbackHumanInterventionHandler: programmatic integration for tests / webhooks

Integration:
    ReconcileLoop accepts an optional HumanInterventionHandler. When provided,
    on_human_intervention_needed() delegates to the handler instead of
    auto-approving. WorkflowController also accepts a handler and delegates
    on_approval_required() to it.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from core.state.models import HumanDecision

if TYPE_CHECKING:
    from core.state.models import LoopContext


class HumanInterventionHandler(ABC):
    """Abstract base for human-in-the-loop handlers.

    Implementors receive structured context about the pending action and must
    return a HumanDecision indicating whether to approve or reject continuation.
    """

    @abstractmethod
    async def request_approval(
        self,
        reason: str,
        context: LoopContext,
        pending_action: dict[str, Any] | None = None,
    ) -> HumanDecision:
        """Request a human approval decision.

        Args:
            reason: Why intervention is needed (shown to the human).
            context: Current loop context (step, history, desired_state).
            pending_action: Optional description of the action awaiting approval.

        Returns:
            HumanDecision with approved=True/False and optional feedback.
        """
        ...


class CLIHumanInterventionHandler(HumanInterventionHandler):
    """Interactive terminal handler.

    Uses ``asyncio.to_thread(input, ...)`` so the event loop is never blocked.
    Displays reason and pending_action, then waits for a y/n response.

    Args:
        prompt_prefix: Prefix shown before each prompt line (default "[HUMAN]").
    """

    def __init__(self, prompt_prefix: str = "[HUMAN]") -> None:
        self._prefix = prompt_prefix

    async def request_approval(
        self,
        reason: str,
        context: LoopContext,
        pending_action: dict[str, Any] | None = None,
    ) -> HumanDecision:
        """Show a CLI prompt and collect approval.

        Args:
            reason: Why intervention is needed.
            context: Current loop context.
            pending_action: Optional action details.

        Returns:
            HumanDecision based on user input.
        """
        import asyncio

        def _prompt() -> tuple[bool, str]:
            print(f"\n{self._prefix} Human Approval Required")
            print(f"{self._prefix} Agent: {context.agent_id}  Step: {context.current_step}")
            print(f"{self._prefix} Reason: {reason}")
            if pending_action:
                print(
                    f"{self._prefix} Pending action: "
                    + json.dumps(pending_action, default=str, indent=2)
                )
            raw = input(f"{self._prefix} Approve? [y/N]: ").strip().lower()
            approved = raw in ("y", "yes")
            feedback = ""
            if not approved:
                feedback = input(f"{self._prefix} Feedback (optional): ").strip()
            return approved, feedback

        approved, feedback = await asyncio.to_thread(_prompt)
        return HumanDecision(
            approved=approved,
            feedback=feedback,
            decision_by="cli_human",
        )


class CallbackHumanInterventionHandler(HumanInterventionHandler):
    """Programmatic handler backed by an async callback.

    Useful for:
    - Unit tests that need to control decisions deterministically.
    - Webhook-based approval flows (wrap the webhook in an async function).
    - UI backends that serve an approval dialog and await the response.

    Args:
        callback: An async callable with signature::

            async def callback(
                reason: str,
                context: LoopContext,
                pending_action: dict | None,
            ) -> HumanDecision: ...
    """

    def __init__(
        self,
        callback: Callable[
            [str, Any, dict[str, Any] | None],
            Awaitable[HumanDecision],
        ],
    ) -> None:
        self._callback = callback

    async def request_approval(
        self,
        reason: str,
        context: LoopContext,
        pending_action: dict[str, Any] | None = None,
    ) -> HumanDecision:
        """Delegate to the callback function.

        Args:
            reason: Why intervention is needed.
            context: Current loop context.
            pending_action: Optional action details.

        Returns:
            HumanDecision returned by the callback.
        """
        return await self._callback(reason, context, pending_action)
