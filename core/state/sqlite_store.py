"""SQLite implementation of StateStore.

Uses aiosqlite for async operations. Implements optimistic locking
via version numbers.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

import aiosqlite

from core.state.models import StateChangeEvent, StateEntry
from core.state.state_store import StateStore
from errors.exceptions import VersionConflictError


class SQLiteStateStore(StateStore):
    """SQLite-based state store implementation.

    Provides persistent state storage with optimistic locking support.
    Uses aiosqlite for non-blocking database operations.

    Args:
        db_path: Path to the SQLite database file. Use ":memory:" for in-memory.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._watchers: list[asyncio.Queue[StateChangeEvent]] = []
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure database connection and schema are initialized."""
        if self._initialized:
            return

        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS states (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL,
                updated_by TEXT NOT NULL DEFAULT 'system'
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_states_key_prefix
            ON states (key)
        """)
        await self._db.commit()
        self._initialized = True

    async def _notify_watchers(self, event: StateChangeEvent) -> None:
        """Notify all watchers of a state change."""
        for queue in self._watchers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def get(self, key: str) -> StateEntry | None:
        """Retrieve a state entry by key."""
        await self._ensure_initialized()
        assert self._db is not None

        async with self._db.execute(
            "SELECT key, value, version, updated_at, updated_by FROM states WHERE key = ?",
            (key,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None

            return StateEntry(
                key=row["key"],
                value=json.loads(row["value"]),
                version=row["version"],
                updated_at=datetime.fromisoformat(row["updated_at"]),
                updated_by=row["updated_by"],
            )

    async def put(
        self,
        key: str,
        value: dict[str, Any],
        expected_version: int | None = None,
        updated_by: str = "system",
    ) -> StateEntry:
        """Store or update a state entry with optimistic locking."""
        await self._ensure_initialized()
        assert self._db is not None

        now = datetime.utcnow()
        value_json = json.dumps(value)

        existing = await self.get(key)

        if expected_version is not None:
            if existing is None:
                raise VersionConflictError(
                    f"State key '{key}' does not exist",
                    key=key,
                    expected_version=expected_version,
                    actual_version=0,
                )
            if existing.version != expected_version:
                raise VersionConflictError(
                    f"Version conflict for key '{key}'",
                    key=key,
                    expected_version=expected_version,
                    actual_version=existing.version,
                )

        if existing is None:
            new_version = 1
            await self._db.execute(
                """
                INSERT INTO states (key, value, version, updated_at, updated_by)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, value_json, new_version, now.isoformat(), updated_by),
            )
            change_type = "created"
            old_value = None
        else:
            new_version = existing.version + 1
            await self._db.execute(
                """
                UPDATE states
                SET value = ?, version = ?, updated_at = ?, updated_by = ?
                WHERE key = ?
                """,
                (value_json, new_version, now.isoformat(), updated_by, key),
            )
            change_type = "updated"
            old_value = existing.value

        await self._db.commit()

        entry = StateEntry(
            key=key,
            value=value,
            version=new_version,
            updated_at=now,
            updated_by=updated_by,
        )

        event = StateChangeEvent(
            key=key,
            old_value=old_value,
            new_value=value,
            change_type=change_type,
            version=new_version,
            timestamp=now,
            changed_by=updated_by,
        )
        await self._notify_watchers(event)

        return entry

    async def delete(self, key: str) -> bool:
        """Delete a state entry."""
        await self._ensure_initialized()
        assert self._db is not None

        existing = await self.get(key)
        if existing is None:
            return False

        await self._db.execute("DELETE FROM states WHERE key = ?", (key,))
        await self._db.commit()

        event = StateChangeEvent(
            key=key,
            old_value=existing.value,
            new_value=None,
            change_type="deleted",
            version=existing.version,
            timestamp=datetime.utcnow(),
            changed_by="system",
        )
        await self._notify_watchers(event)

        return True

    async def list(self, prefix: str = "") -> list[StateEntry]:
        """List all state entries matching a prefix."""
        await self._ensure_initialized()
        assert self._db is not None

        if prefix:
            query = """
                SELECT key, value, version, updated_at, updated_by
                FROM states
                WHERE key LIKE ?
                ORDER BY key
            """
            params = (f"{prefix}%",)
        else:
            query = """
                SELECT key, value, version, updated_at, updated_by
                FROM states
                ORDER BY key
            """
            params = ()

        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [
                StateEntry(
                    key=row["key"],
                    value=json.loads(row["value"]),
                    version=row["version"],
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    updated_by=row["updated_by"],
                )
                for row in rows
            ]

    async def watch(self, prefix: str = "") -> AsyncIterator[StateChangeEvent]:
        """Watch for state changes matching a prefix."""
        await self._ensure_initialized()

        queue: asyncio.Queue[StateChangeEvent] = asyncio.Queue(maxsize=100)
        self._watchers.append(queue)

        try:
            while True:
                event = await queue.get()
                if prefix and not event.key.startswith(prefix):
                    continue
                yield event
        finally:
            self._watchers.remove(queue)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            self._initialized = False

    async def clear(self) -> None:
        """Clear all state entries. Useful for testing."""
        await self._ensure_initialized()
        assert self._db is not None

        await self._db.execute("DELETE FROM states")
        await self._db.commit()
