"""PostgreSQL StateStore implementation.

Mirrors the SQLiteStateStore interface using asyncpg for high-throughput
production deployments. Supports optimistic locking, LISTEN/NOTIFY-based
watch, and connection pooling.

Optional dependency group ``[postgres]`` must be installed::

    pip install agent-framework[postgres]
    # or: pip install asyncpg>=0.29

The SQLiteStateStore is the default and works without any extra dependencies.
Use PostgreSQLStateStore only when you need a shared, horizontally-scalable
state backend.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, AsyncIterator

from core.state.models import StateChangeEvent, StateEntry
from core.state.state_store import StateStore
from errors.exceptions import VersionConflictError

try:
    import asyncpg  # type: ignore[import]
    from asyncpg import Pool  # type: ignore[import]
except ImportError as _exc:
    raise ImportError(
        "asyncpg is required for PostgreSQLStateStore. "
        "Install it with: pip install agent-framework[postgres]"
    ) from _exc


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS state_entries (
    key         TEXT PRIMARY KEY,
    value       JSONB NOT NULL DEFAULT '{}',
    version     INTEGER NOT NULL DEFAULT 1,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by  TEXT NOT NULL DEFAULT 'system'
)
"""


class PostgreSQLStateStore(StateStore):
    """StateStore backed by PostgreSQL.

    Uses an asyncpg connection pool and LISTEN/NOTIFY for watch support.

    Do not instantiate directly — use the async factory :meth:`create`.

    Args:
        pool: An open asyncpg connection pool.
    """

    def __init__(self, pool: Pool) -> None:
        self._pool = pool
        self._watchers: dict[str, list[asyncio.Queue[StateChangeEvent]]] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def create(
        cls,
        dsn: str,
        pool_size: int = 10,
    ) -> PostgreSQLStateStore:
        """Async factory — creates the pool and schema.

        Args:
            dsn: PostgreSQL DSN, e.g. ``postgresql://user:pass@host/db``.
            pool_size: Maximum pool connections (default 10).

        Returns:
            Initialized PostgreSQLStateStore.
        """
        pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=2,
            max_size=pool_size,
            command_timeout=30,
        )
        async with pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE)

        instance = cls(pool)
        return instance

    # ------------------------------------------------------------------
    # StateStore ABC
    # ------------------------------------------------------------------

    async def get(self, key: str) -> StateEntry | None:
        """Get a state entry by key.

        Args:
            key: The state key to retrieve.

        Returns:
            StateEntry if found, None otherwise.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM state_entries WHERE key = $1", key
            )
        if row is None:
            return None
        return self._row_to_entry(row)

    async def put(
        self,
        key: str,
        value: StateEntry,
        expected_version: int | None = None,
    ) -> StateEntry:
        """Upsert a state entry with optional optimistic locking.

        Args:
            key: The state key.
            value: The StateEntry to persist.
            expected_version: If set, the current version must match or
                VersionConflictError is raised.

        Returns:
            The updated StateEntry with new version number.

        Raises:
            VersionConflictError: If expected_version does not match.
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                existing = await conn.fetchrow(
                    "SELECT version FROM state_entries WHERE key = $1 FOR UPDATE",
                    key,
                )

                if expected_version is not None and existing is not None:
                    if existing["version"] != expected_version:
                        raise VersionConflictError(
                            f"Version conflict on key '{key}'",
                            key=key,
                            expected_version=expected_version,
                            actual_version=existing["version"],
                        )

                new_version = (existing["version"] + 1) if existing else 1
                now = datetime.utcnow()

                await conn.execute(
                    """
                    INSERT INTO state_entries (key, value, version, updated_at, updated_by)
                    VALUES ($1, $2::jsonb, $3, $4, $5)
                    ON CONFLICT (key) DO UPDATE SET
                        value      = EXCLUDED.value,
                        version    = EXCLUDED.version,
                        updated_at = EXCLUDED.updated_at,
                        updated_by = EXCLUDED.updated_by
                    """,
                    key,
                    json.dumps(value.value),
                    new_version,
                    now,
                    value.updated_by,
                )

                # NOTIFY watchers
                channel = f"state_{key.replace('/', '_')}"
                payload = json.dumps(
                    {
                        "key": key,
                        "version": new_version,
                        "change_type": "created" if not existing else "updated",
                        "changed_by": value.updated_by,
                    }
                )
                await conn.execute(f"NOTIFY {channel}, '{payload}'")

        updated = StateEntry(
            key=key,
            value=value.value,
            version=new_version,
            updated_at=now,
            updated_by=value.updated_by,
        )

        # Dispatch to in-process watchers
        await self._dispatch_event(
            key=key,
            old_value=None,
            new_value=value.value,
            change_type="created" if not existing else "updated",
            version=new_version,
            changed_by=value.updated_by,
        )

        return updated

    async def delete(self, key: str) -> bool:
        """Delete a state entry.

        Args:
            key: The key to delete.

        Returns:
            True if the entry existed and was deleted.
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM state_entries WHERE key = $1", key
            )

        deleted = result.endswith("1")
        if deleted:
            await self._dispatch_event(
                key=key,
                old_value=None,
                new_value=None,
                change_type="deleted",
                version=0,
                changed_by="system",
            )
        return deleted

    async def list(self, prefix: str = "") -> list[StateEntry]:
        """List all entries matching a key prefix.

        Args:
            prefix: Key prefix to filter by. Empty string returns all entries.

        Returns:
            List of matching StateEntry objects.
        """
        async with self._pool.acquire() as conn:
            if prefix:
                rows = await conn.fetch(
                    "SELECT * FROM state_entries WHERE key LIKE $1 ORDER BY key",
                    f"{prefix}%",
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM state_entries ORDER BY key"
                )
        return [self._row_to_entry(row) for row in rows]

    async def watch(self, prefix: str = "") -> AsyncIterator[StateChangeEvent]:
        """Watch for changes to keys matching a prefix.

        Yields StateChangeEvent whenever a matching key is created, updated,
        or deleted. Uses an in-process asyncio.Queue; LISTEN/NOTIFY events
        from other processes are received via _listen_loop (Phase 4).

        Args:
            prefix: Key prefix to filter. Empty string watches all keys.

        Yields:
            StateChangeEvent for each matching change.
        """
        queue: asyncio.Queue[StateChangeEvent] = asyncio.Queue()
        self._watchers.setdefault(prefix, []).append(queue)
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            watchers = self._watchers.get(prefix, [])
            if queue in watchers:
                watchers.remove(queue)

    async def close(self) -> None:
        """Close the connection pool."""
        await self._pool.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: Any) -> StateEntry:
        value = row["value"]
        if isinstance(value, str):
            value = json.loads(value)
        return StateEntry(
            key=row["key"],
            value=value,
            version=row["version"],
            updated_at=row["updated_at"],
            updated_by=row["updated_by"],
        )

    async def _dispatch_event(
        self,
        key: str,
        old_value: dict[str, Any] | None,
        new_value: dict[str, Any] | None,
        change_type: str,
        version: int,
        changed_by: str,
    ) -> None:
        """Dispatch a StateChangeEvent to all matching in-process watchers."""
        event = StateChangeEvent(
            key=key,
            old_value=old_value,
            new_value=new_value,
            change_type=change_type,  # type: ignore[arg-type]
            version=version,
            changed_by=changed_by,
        )
        for watched_prefix, queues in self._watchers.items():
            if key.startswith(watched_prefix):
                for q in queues:
                    await q.put(event)
