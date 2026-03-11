"""Shared pytest fixtures for all tests.

All async tests run with asyncio_mode="auto" (set in pyproject.toml).
No @pytest.mark.asyncio decorator needed on individual tests.
"""

from __future__ import annotations

import pytest

from core.state.sqlite_store import SQLiteStateStore


@pytest.fixture
async def state_store() -> SQLiteStateStore:
    """Fresh in-memory SQLite StateStore for each test.

    Uses ":memory:" so no files are created and no cleanup is needed
    beyond calling close() on teardown.
    """
    store = SQLiteStateStore(":memory:")
    yield store
    await store.close()
