"""AgentScratchpad — lightweight in-run notepad for agents.

Single-run lifecycle: the scratchpad is reset at the start of each Agent.run()
call and discarded when the run completes. Analogous to TodoWrite in Claude Code:
a fast, ephemeral scratch space with no TTL, no LRU, and no persistence.

Unlike WorkingMemory (which has TTL, LRU, and tag-based retrieval designed for
cross-step knowledge), AgentScratchpad is intentionally minimal — a dict with a
clean API so agents can jot down notes, intermediate results, or task status
without polluting the StateStore.
"""

from __future__ import annotations

from typing import Any


class AgentScratchpad:
    """Lightweight per-run scratchpad for agents.

    Key semantics:
    - Values can be any Python object (not serialized or validated).
    - No TTL, no LRU, no size limit — caller is responsible for hygiene.
    - Cleared automatically at the start of each ``Agent.run()`` call.

    Usage::

        scratchpad = AgentScratchpad()

        scratchpad.set("todo_items", ["read file", "analyse output"])
        scratchpad.set("current_step", 1)

        items = scratchpad.get("todo_items")   # ["read file", "analyse output"]
        scratchpad.delete("current_step")
        scratchpad.clear()
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Store a value under ``key``.

        Args:
            key: Identifier for the note.
            value: Any Python value to store.
        """
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve the value for ``key``, returning ``default`` if absent.

        Args:
            key: Identifier to look up.
            default: Fallback value when key is not found.

        Returns:
            Stored value or ``default``.
        """
        return self._data.get(key, default)

    def delete(self, key: str) -> bool:
        """Remove a key from the scratchpad.

        Args:
            key: Identifier to remove.

        Returns:
            True if the key existed and was removed, False if not found.
        """
        if key in self._data:
            del self._data[key]
            return True
        return False

    def clear(self) -> None:
        """Remove all entries. Called automatically by Agent.run()."""
        self._data.clear()

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def keys(self) -> list[str]:
        """Return all current keys."""
        return list(self._data.keys())

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow copy of all key-value pairs."""
        return dict(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"AgentScratchpad({len(self._data)} items)"
