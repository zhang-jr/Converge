"""Unit tests for AuditLog."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from observability.audit_log import AuditEntry, AuditLog


class TestAuditLog:
    @pytest.fixture()
    async def log(self) -> AuditLog:
        al = AuditLog(db_path=":memory:")
        yield al
        await al.close()

    # ------------------------------------------------------------------
    # log / query
    # ------------------------------------------------------------------

    async def test_log_returns_entry_id(self, log: AuditLog) -> None:
        entry = AuditEntry(actor="agent-1", action="agent_start", resource="test")
        eid = await log.log(entry)
        assert eid == entry.entry_id

    async def test_query_all_entries(self, log: AuditLog) -> None:
        for i in range(3):
            await log.log(AuditEntry(actor=f"a{i}", action="agent_start"))
        entries = await log.query()
        assert len(entries) == 3

    async def test_query_filter_by_actor(self, log: AuditLog) -> None:
        await log.log(AuditEntry(actor="alice", action="tool_execute"))
        await log.log(AuditEntry(actor="bob", action="agent_start"))
        entries = await log.query(actor="alice")
        assert len(entries) == 1
        assert entries[0].actor == "alice"

    async def test_query_filter_by_action(self, log: AuditLog) -> None:
        await log.log(AuditEntry(actor="a", action="tool_execute"))
        await log.log(AuditEntry(actor="a", action="agent_end"))
        entries = await log.query(action="tool_execute")
        assert len(entries) == 1
        assert entries[0].action == "tool_execute"

    async def test_query_filter_by_trace_id(self, log: AuditLog) -> None:
        await log.log(AuditEntry(actor="a", action="agent_start", trace_id="trace-1"))
        await log.log(AuditEntry(actor="a", action="agent_end", trace_id="trace-2"))
        entries = await log.query(trace_id="trace-1")
        assert len(entries) == 1
        assert entries[0].trace_id == "trace-1"

    async def test_query_filter_by_since(self, log: AuditLog) -> None:
        old = AuditEntry(
            actor="a",
            action="agent_start",
            timestamp=datetime(2020, 1, 1),
        )
        recent = AuditEntry(
            actor="a",
            action="agent_end",
            timestamp=datetime.utcnow(),
        )
        await log.log(old)
        await log.log(recent)

        cutoff = datetime(2021, 1, 1)
        entries = await log.query(since=cutoff)
        assert len(entries) == 1
        assert entries[0].action == "agent_end"

    async def test_query_limit(self, log: AuditLog) -> None:
        for _ in range(10):
            await log.log(AuditEntry(actor="a", action="agent_start"))
        entries = await log.query(limit=3)
        assert len(entries) == 3

    # ------------------------------------------------------------------
    # Immutability — no UPDATE or DELETE
    # ------------------------------------------------------------------

    async def test_entries_cannot_be_updated(self, log: AuditLog) -> None:
        """Directly attempting UPDATE on the table should not change stored data."""
        entry = AuditEntry(actor="alice", action="agent_start", resource="original")
        await log.log(entry)

        # Simulate an errant UPDATE attempt
        conn = await log._get_conn()
        await conn.execute(
            "UPDATE audit_entries SET resource = 'modified' WHERE entry_id = ?",
            (entry.entry_id,),
        )
        # We deliberately do NOT commit so the UPDATE has no effect (autocommit off)
        await conn.rollback()

        entries = await log.query(actor="alice")
        assert entries[0].resource == "original"

    async def test_duplicate_entry_id_raises(self, log: AuditLog) -> None:
        """Inserting the same entry_id twice should raise."""
        entry = AuditEntry(actor="a", action="agent_start")
        await log.log(entry)
        with pytest.raises(Exception):  # PRIMARY KEY violation
            await log.log(entry)

    # ------------------------------------------------------------------
    # Details serialisation
    # ------------------------------------------------------------------

    async def test_details_round_trip(self, log: AuditLog) -> None:
        details = {"tool": "search", "params": {"q": "hello"}, "count": 42}
        entry = AuditEntry(actor="a", action="tool_execute", details=details)
        await log.log(entry)
        entries = await log.query()
        assert entries[0].details == details

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------

    async def test_close_is_idempotent(self) -> None:
        al = AuditLog(":memory:")
        await al.close()
        await al.close()  # second close should not raise
