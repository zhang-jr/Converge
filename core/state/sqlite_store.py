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

import uuid

import aiosqlite

from core.state.models import StateChangeEvent, StateEntry
from core.state.state_store import StateStore
from errors.exceptions import RollbackError, VersionConflictError


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
        # Snapshot metadata: one row per snapshot (always present, even for empty store)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS state_snapshot_meta (
                snapshot_id TEXT PRIMARY KEY,
                prefix TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        # Snapshot data: one row per captured key (may be zero rows for empty store)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS state_snapshots (
                snapshot_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                version INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                updated_by TEXT NOT NULL,
                PRIMARY KEY (snapshot_id, key)
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshots_id
            ON state_snapshots (snapshot_id)
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

    async def snapshot(self, prefix: str = "") -> str:
        """Create a point-in-time snapshot of entries matching prefix.

        Always records a metadata row so restore() can find the snapshot
        even when the store is empty at the time of snapshotting.

        Args:
            prefix: Key prefix to capture.  Empty string captures everything.

        Returns:
            A UUID-based snapshot ID for later use with restore/delete_snapshot.
        """
        await self._ensure_initialized()
        assert self._db is not None

        snapshot_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        # Always write a metadata row (works even for empty stores)
        await self._db.execute(
            "INSERT INTO state_snapshot_meta (snapshot_id, prefix, created_at) VALUES (?, ?, ?)",
            (snapshot_id, prefix, now),
        )

        # Copy matching state entries into snapshot data table
        if prefix:
            await self._db.execute(
                """
                INSERT INTO state_snapshots (snapshot_id, key, value, version, updated_at, updated_by)
                SELECT ?, key, value, version, updated_at, updated_by
                FROM states WHERE key LIKE ?
                """,
                (snapshot_id, f"{prefix}%"),
            )
        else:
            await self._db.execute(
                """
                INSERT INTO state_snapshots (snapshot_id, key, value, version, updated_at, updated_by)
                SELECT ?, key, value, version, updated_at, updated_by
                FROM states
                """,
                (snapshot_id,),
            )

        await self._db.commit()
        return snapshot_id

    async def restore(self, snapshot_id: str) -> None:
        """Restore state from a previously created snapshot.

        Keys in the snapshot's prefix scope are replaced with their
        snapshotted values.  Keys added after the snapshot (within the
        same prefix) are deleted.

        Args:
            snapshot_id: Snapshot ID returned by snapshot().

        Raises:
            RollbackError: If snapshot_id does not exist.
        """
        await self._ensure_initialized()
        assert self._db is not None

        # Verify snapshot exists via metadata table (present even for empty snapshots)
        async with self._db.execute(
            "SELECT prefix FROM state_snapshot_meta WHERE snapshot_id = ?",
            (snapshot_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            raise RollbackError(
                f"Snapshot '{snapshot_id}' not found",
                snapshot_id=snapshot_id,
            )

        prefix: str = row["prefix"]

        # Delete current entries in the prefix scope
        if prefix:
            await self._db.execute(
                "DELETE FROM states WHERE key LIKE ?", (f"{prefix}%",)
            )
        else:
            await self._db.execute("DELETE FROM states")

        # Re-insert from snapshot data (may be zero rows if store was empty)
        await self._db.execute(
            """
            INSERT INTO states (key, value, version, updated_at, updated_by)
            SELECT key, value, version, updated_at, updated_by
            FROM state_snapshots WHERE snapshot_id = ?
            """,
            (snapshot_id,),
        )

        await self._db.commit()

        # Notify watchers about the restore
        now = datetime.utcnow()
        async with self._db.execute(
            "SELECT key, value FROM states" + (" WHERE key LIKE ?" if prefix else ""),
            (f"{prefix}%",) if prefix else (),
        ) as cursor:
            rows = await cursor.fetchall()

        for r in rows:
            event = StateChangeEvent(
                key=r["key"],
                old_value=None,
                new_value=json.loads(r["value"]),
                change_type="updated",
                version=0,
                timestamp=now,
                changed_by="system:rollback",
            )
            await self._notify_watchers(event)

    async def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete a snapshot and its metadata to free storage.

        Args:
            snapshot_id: Snapshot ID returned by snapshot().
        """
        await self._ensure_initialized()
        assert self._db is not None

        await self._db.execute(
            "DELETE FROM state_snapshots WHERE snapshot_id = ?", (snapshot_id,)
        )
        await self._db.execute(
            "DELETE FROM state_snapshot_meta WHERE snapshot_id = ?", (snapshot_id,)
        )
        await self._db.commit()

    async def list_snapshots(self) -> list[str]:
        """Return all snapshot IDs stored in this database.

        Returns:
            List of snapshot ID strings ordered by creation time.
        """
        await self._ensure_initialized()
        assert self._db is not None

        async with self._db.execute(
            "SELECT snapshot_id FROM state_snapshot_meta ORDER BY created_at"
        ) as cursor:
            rows = await cursor.fetchall()
        return [r["snapshot_id"] for r in rows]

    async def clear(self) -> None:
        """Clear all state entries. Useful for testing."""
        await self._ensure_initialized()
        assert self._db is not None

        await self._db.execute("DELETE FROM states")
        await self._db.commit()
