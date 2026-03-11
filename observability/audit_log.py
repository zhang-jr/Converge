"""Append-only audit log backed by SQLite.

Every significant agent action is recorded as an immutable AuditEntry.
The log is INSERT-ONLY — no UPDATE or DELETE is ever executed on the table.

Typical integration points:
- Tracer.log_tool_call()        → action="tool_execute"
- Tracer.log_human_intervention() → action="approval_request" / "approval_decision"
- ReconcileLoop on_loop_start   → action="agent_start"
- ReconcileLoop on_convergence  → action="agent_end"
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Literal

import aiosqlite
from pydantic import BaseModel, Field


class AuditEntry(BaseModel):
    """A single immutable audit log entry.

    Attributes:
        entry_id: Unique ID (UUID) for this entry.
        actor: Agent ID or human identifier that performed the action.
        action: Categorized action type.
        resource: The resource affected (e.g., tool name, state key).
        outcome: Result of the action.
        trace_id: Associated reconcile trace ID.
        timestamp: When the event occurred.
        details: Arbitrary additional context.
    """

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    actor: str
    action: Literal[
        "tool_execute",
        "state_write",
        "approval_request",
        "approval_decision",
        "agent_start",
        "agent_end",
    ]
    resource: str = ""
    outcome: Literal["success", "failure", "pending"] = "success"
    trace_id: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: dict[str, Any] = Field(default_factory=dict)


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS audit_entries (
    entry_id  TEXT PRIMARY KEY,
    actor     TEXT NOT NULL,
    action    TEXT NOT NULL,
    resource  TEXT NOT NULL DEFAULT '',
    outcome   TEXT NOT NULL DEFAULT 'success',
    trace_id  TEXT NOT NULL DEFAULT '',
    timestamp TEXT NOT NULL,
    details   TEXT NOT NULL DEFAULT '{}'
)
"""


class AuditLog:
    """Append-only audit log with SQLite backend.

    The table is created on first use. No UPDATE or DELETE operations are
    ever performed — entries are permanent records.

    Args:
        db_path: Path to the SQLite database file.
            Use ``":memory:"`` for testing.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def _get_conn(self) -> aiosqlite.Connection:
        """Open and initialise the connection if needed."""
        if self._conn is None:
            self._conn = await aiosqlite.connect(self._db_path)
            self._conn.row_factory = aiosqlite.Row
            await self._conn.execute(_CREATE_TABLE)
            await self._conn.commit()
        return self._conn

    async def log(self, entry: AuditEntry) -> str:
        """Append an audit entry to the log.

        Args:
            entry: The entry to persist.

        Returns:
            The entry_id of the stored entry.
        """
        conn = await self._get_conn()
        await conn.execute(
            """
            INSERT INTO audit_entries
                (entry_id, actor, action, resource, outcome, trace_id, timestamp, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.entry_id,
                entry.actor,
                entry.action,
                entry.resource,
                entry.outcome,
                entry.trace_id,
                entry.timestamp.isoformat(),
                json.dumps(entry.details, default=str),
            ),
        )
        await conn.commit()
        return entry.entry_id

    async def query(
        self,
        actor: str | None = None,
        action: str | None = None,
        trace_id: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit entries with optional filters.

        Args:
            actor: Filter by actor (exact match).
            action: Filter by action type (exact match).
            trace_id: Filter by trace ID (exact match).
            since: Return only entries at or after this timestamp.
            limit: Maximum number of entries to return.

        Returns:
            List of matching AuditEntry objects ordered by timestamp ascending.
        """
        conn = await self._get_conn()

        clauses: list[str] = []
        params: list[Any] = []

        if actor is not None:
            clauses.append("actor = ?")
            params.append(actor)
        if action is not None:
            clauses.append("action = ?")
            params.append(action)
        if trace_id is not None:
            clauses.append("trace_id = ?")
            params.append(trace_id)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())

        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(limit)

        sql = f"SELECT * FROM audit_entries {where} ORDER BY timestamp ASC LIMIT ?"
        async with conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()

        return [
            AuditEntry(
                entry_id=row["entry_id"],
                actor=row["actor"],
                action=row["action"],  # type: ignore[arg-type]
                resource=row["resource"],
                outcome=row["outcome"],  # type: ignore[arg-type]
                trace_id=row["trace_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                details=json.loads(row["details"]),
            )
            for row in rows
        ]

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
