"""Working memory for in-progress agent execution.

Working memory is a fast, ephemeral scratch space for agents during task
execution. Unlike StateStore it has:
- No persistence (lives only as long as the object)
- No versioning or optimistic locking
- Optional per-entry TTL
- Soft capacity limit with LRU eviction

Analogous to CPU L1 cache: fast, small, temporary.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

from pydantic import BaseModel, Field

# Private sentinel used by has() to distinguish "not found" from "stored None"
_SENTINEL = object()


class MemoryEntry(BaseModel):
    """A single working memory entry.

    Attributes:
        key: Entry key.
        value: Stored value.
        created_at: Unix timestamp of creation.
        ttl_seconds: TTL in seconds (-1.0 means no expiry).
        tags: Optional tags for grouped retrieval.
    """

    key: str
    value: Any
    created_at: float = Field(default_factory=time.time)
    ttl_seconds: float = -1.0
    tags: list[str] = Field(default_factory=list)

    def is_expired(self) -> bool:
        """Check if this entry has exceeded its TTL."""
        if self.ttl_seconds < 0:
            return False
        return time.time() - self.created_at > self.ttl_seconds


class WorkingMemory:
    """Ephemeral in-memory store for current task context.

    Provides fast key-value storage scoped to a single task or reconcile run.
    LRU eviction kicks in when max_entries is exceeded.

    Usage:
        memory = WorkingMemory(max_entries=100)
        memory.set("current_plan", {"steps": [...]})
        memory.set("temp_result", data, ttl_seconds=30.0)

        plan = memory.get("current_plan")
        assert memory.has("current_plan")

        memory.clear()
    """

    def __init__(self, max_entries: int = 200) -> None:
        """Initialize working memory.

        Args:
            max_entries: Maximum entries before LRU eviction begins.
        """
        self._entries: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._max_entries = max_entries

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: float = -1.0,
        tags: list[str] | None = None,
    ) -> None:
        """Store a value in working memory.

        Overwrites any existing entry with the same key.
        Evicts the least-recently-used entry if over capacity.

        Args:
            key: Entry key.
            value: Value to store (any Python object).
            ttl_seconds: TTL in seconds (-1.0 = never expire).
            tags: Optional tags for grouped retrieval.
        """
        # Remove existing entry to reset its LRU position
        self._entries.pop(key, None)

        entry = MemoryEntry(
            key=key,
            value=value,
            ttl_seconds=ttl_seconds,
            tags=tags or [],
        )
        self._entries[key] = entry

        # Evict LRU entries when over limit
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from working memory.

        Moves the entry to the MRU position on access.
        Returns default if key is absent or entry has expired.

        Args:
            key: Entry key.
            default: Value to return if not found or expired.

        Returns:
            Stored value or default.
        """
        entry = self._entries.get(key)
        if entry is None:
            return default

        if entry.is_expired():
            del self._entries[key]
            return default

        self._entries.move_to_end(key)
        return entry.value

    def delete(self, key: str) -> bool:
        """Delete an entry.

        Args:
            key: Entry key.

        Returns:
            True if deleted, False if key was not present.
        """
        if key in self._entries:
            del self._entries[key]
            return True
        return False

    def has(self, key: str) -> bool:
        """Check if a key exists and has not expired.

        Args:
            key: Entry key.

        Returns:
            True if the key is present and not expired.
        """
        return self.get(key, _SENTINEL) is not _SENTINEL

    def get_by_tag(self, tag: str) -> dict[str, Any]:
        """Get all non-expired entries with a specific tag.

        Args:
            tag: Tag to filter by.

        Returns:
            Dict of key -> value for all matching, non-expired entries.
        """
        result: dict[str, Any] = {}
        expired: list[str] = []

        for key, entry in self._entries.items():
            if entry.is_expired():
                expired.append(key)
            elif tag in entry.tags:
                result[key] = entry.value

        for k in expired:
            del self._entries[k]

        return result

    def update(self, key: str, value: Any) -> bool:
        """Update an existing entry's value without changing its TTL.

        Args:
            key: Entry key.
            value: New value.

        Returns:
            True if updated, False if key was not present.
        """
        entry = self._entries.get(key)
        if entry is None or entry.is_expired():
            return False
        entry.value = value
        self._entries.move_to_end(key)
        return True

    def clear(self) -> None:
        """Clear all working memory entries."""
        self._entries.clear()

    def prune_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        expired = [k for k, v in self._entries.items() if v.is_expired()]
        for k in expired:
            del self._entries[k]
        return len(expired)

    def size(self) -> int:
        """Return current number of entries (may include expired)."""
        return len(self._entries)

    def snapshot(self) -> dict[str, Any]:
        """Return a dict snapshot of all non-expired entries.

        Returns:
            Dict of key -> value for all live entries.
        """
        return {k: e.value for k, e in self._entries.items() if not e.is_expired()}
