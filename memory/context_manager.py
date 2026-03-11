"""Context Window Manager.

Manages the LLM context window by tracking token usage and evicting
old messages when approaching the limit. Evicted content is optionally
sent to episodic memory to avoid total information loss (ADR-003).

Eviction strategy (ADR-003):
  - system prompt: pinned, never evicted
  - current tool results: kept for the current step
  - recent history: kept up to budget
  - old history: evicted LRU-first, summarized into episodic memory
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from memory.episodic import EpisodicMemory


class ContextMessage(BaseModel):
    """A single message in the context window.

    Attributes:
        role: Message role: system / user / assistant / tool.
        content: Message content.
        token_count: Estimated token count (auto-calculated if 0).
        step_number: Reconcile loop step this message came from.
        pinned: If True, this message is never evicted.
    """

    role: str
    content: str
    token_count: int = 0
    step_number: int = 0
    pinned: bool = False

    def model_post_init(self, __context: Any) -> None:
        # Rough estimate: 1 token ≈ 4 characters
        if self.token_count == 0:
            object.__setattr__(self, "token_count", max(1, len(self.content) // 4))


class ContextWindowManager:
    """Manages the LLM context window with budget-aware eviction.

    Maintains an ordered list of messages that fit within the token budget.
    When new messages push over the limit, the oldest unpinned messages are
    evicted and optionally archived to episodic memory.

    Usage:
        manager = ContextWindowManager(max_tokens=8192)
        manager.add_system("You are a helpful agent.")

        for step in reconcile_loop:
            manager.add_user(f"Goal: {goal}", step=step)
            response = await llm.call(manager.get_messages())
            manager.add_assistant(response.content, step=step)

        stats = manager.get_stats()
    """

    def __init__(
        self,
        max_tokens: int = 8192,
        reserve_tokens: int = 2048,
        episodic_memory: EpisodicMemory | None = None,
    ) -> None:
        """Initialize the context window manager.

        Args:
            max_tokens: Maximum tokens for the context window.
            reserve_tokens: Tokens reserved for the model's response.
            episodic_memory: Optional episodic memory for archiving evicted content.
        """
        self._max_tokens = max_tokens
        self._reserve_tokens = reserve_tokens
        self._episodic_memory = episodic_memory
        self._messages: list[ContextMessage] = []

    @property
    def budget(self) -> int:
        """Effective token budget for stored messages."""
        return self._max_tokens - self._reserve_tokens

    @property
    def used_tokens(self) -> int:
        """Total tokens currently occupied by stored messages."""
        return sum(m.token_count for m in self._messages)

    @property
    def available_tokens(self) -> int:
        """Tokens available for new messages without triggering eviction."""
        return self.budget - self.used_tokens

    def add_system(self, content: str) -> None:
        """Add or replace the system prompt (always pinned).

        Only one system message is kept. Adding a new one replaces the old one.

        Args:
            content: System prompt content.
        """
        msg = ContextMessage(role="system", content=content, pinned=True)
        self._messages = [m for m in self._messages if m.role != "system"]
        self._messages.insert(0, msg)

    def add_user(self, content: str, step: int = 0) -> None:
        """Add a user message.

        Args:
            content: Message content.
            step: Reconcile loop step number.
        """
        self._add(ContextMessage(role="user", content=content, step_number=step))

    def add_assistant(self, content: str, step: int = 0) -> None:
        """Add an assistant message.

        Args:
            content: Message content.
            step: Reconcile loop step number.
        """
        self._add(ContextMessage(role="assistant", content=content, step_number=step))

    def add_tool_result(self, content: str, step: int = 0) -> None:
        """Add a tool result message.

        Args:
            content: Tool result content.
            step: Reconcile loop step number.
        """
        self._add(ContextMessage(role="tool", content=content, step_number=step))

    def _add(self, message: ContextMessage) -> None:
        """Append a message and evict if over budget."""
        self._messages.append(message)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Evict oldest unpinned messages until within budget."""
        while self.used_tokens > self.budget:
            evict_idx: int | None = None
            for i, msg in enumerate(self._messages):
                if not msg.pinned:
                    evict_idx = i
                    break

            if evict_idx is None:
                # All messages are pinned; cannot evict any further
                break

            evicted = self._messages.pop(evict_idx)
            self._archive_eviction(evicted)

    def _archive_eviction(self, evicted: ContextMessage) -> None:
        """Archive evicted message to episodic memory (fire-and-forget)."""
        if self._episodic_memory is None:
            return
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(
                    self._episodic_memory.store_eviction(
                        evicted_content=evicted.content,
                        role=evicted.role,
                        step=evicted.step_number,
                    )
                )
        except RuntimeError:
            pass  # No event loop available; skip archiving

    def get_messages(self) -> list[dict[str, str]]:
        """Get all current messages in LLM API format.

        Returns:
            List of {"role": ..., "content": ...} dicts.
        """
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def get_stats(self) -> dict[str, int]:
        """Get token usage statistics.

        Returns:
            Dict with max_tokens, reserve_tokens, used_tokens, available_tokens,
            and message_count.
        """
        return {
            "max_tokens": self._max_tokens,
            "reserve_tokens": self._reserve_tokens,
            "budget": self.budget,
            "used_tokens": self.used_tokens,
            "available_tokens": self.available_tokens,
            "message_count": len(self._messages),
        }

    def clear_history(self, keep_system: bool = True) -> None:
        """Clear conversation history.

        Args:
            keep_system: If True, the system prompt is retained.
        """
        if keep_system:
            self._messages = [m for m in self._messages if m.role == "system"]
        else:
            self._messages.clear()

    def message_count(self) -> int:
        """Return number of messages currently in the window."""
        return len(self._messages)
