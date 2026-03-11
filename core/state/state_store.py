"""Abstract StateStore interface.

The StateStore is the single source of truth for all state in the framework.
All components must read/write state through this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from core.state.models import StateChangeEvent, StateEntry


class StateStore(ABC):
    """Abstract base class for state storage.

    Implements the single source of truth pattern. All state changes must
    go through this interface. Supports optimistic locking via version numbers.

    Implementations must be async-compatible for non-blocking operations.
    """

    @abstractmethod
    async def get(self, key: str) -> StateEntry | None:
        """Retrieve a state entry by key.

        Args:
            key: The unique key of the state entry.

        Returns:
            The StateEntry if found, None otherwise.
        """
        ...

    @abstractmethod
    async def put(
        self,
        key: str,
        value: dict,
        expected_version: int | None = None,
        updated_by: str = "system",
    ) -> StateEntry:
        """Store or update a state entry.

        Args:
            key: The unique key of the state entry.
            value: The state data to store.
            expected_version: If provided, the operation will fail with
                VersionConflictError if the current version doesn't match.
                This enables optimistic locking.
            updated_by: ID of the agent or "system" making the update.

        Returns:
            The updated StateEntry with incremented version.

        Raises:
            VersionConflictError: If expected_version doesn't match current version.
        """
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a state entry.

        Args:
            key: The unique key of the state entry to delete.

        Returns:
            True if the entry was deleted, False if it didn't exist.
        """
        ...

    @abstractmethod
    async def list(self, prefix: str = "") -> list[StateEntry]:
        """List all state entries matching a prefix.

        Args:
            prefix: Key prefix to filter by. Empty string returns all entries.

        Returns:
            List of matching StateEntry objects, sorted by key.
        """
        ...

    @abstractmethod
    async def watch(self, prefix: str = "") -> AsyncIterator[StateChangeEvent]:
        """Watch for state changes matching a prefix.

        This is an async generator that yields StateChangeEvent objects
        whenever a matching state entry is created, updated, or deleted.

        Args:
            prefix: Key prefix to watch. Empty string watches all entries.

        Yields:
            StateChangeEvent for each change.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the state store and release resources."""
        ...

    async def __aenter__(self) -> StateStore:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
