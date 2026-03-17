"""Unit tests for memory/episodic.py — Episodic memory."""

import asyncio

import pytest

from memory.episodic import Episode, EpisodicMemory


@pytest.fixture
async def memory() -> EpisodicMemory:
    """Fresh in-memory EpisodicMemory, initialized and closed per test."""
    mem = EpisodicMemory(":memory:")
    await mem.initialize()
    yield mem
    await mem.close()


# =============================================================================
# store() and get()
# =============================================================================


class TestStoreAndGet:
    """Tests for storing and retrieving episodes."""

    async def test_store_returns_episode_id(self, memory: EpisodicMemory):
        """store() returns a non-empty episode_id string."""
        ep_id = await memory.store(
            agent_id="agent-1",
            task_summary="Analyzed auth module",
            outcome="Found 2 issues",
        )
        assert isinstance(ep_id, str)
        assert len(ep_id) > 0

    async def test_get_retrieves_stored_episode(self, memory: EpisodicMemory):
        """get() retrieves the episode stored with the returned ID."""
        ep_id = await memory.store(
            agent_id="agent-1",
            task_summary="Analyzed auth module",
            outcome="Found 2 issues",
            context_tags=["auth", "security"],
        )

        episode = await memory.get(ep_id)
        assert episode is not None
        assert episode.episode_id == ep_id
        assert episode.agent_id == "agent-1"
        assert episode.task_summary == "Analyzed auth module"
        assert episode.outcome == "Found 2 issues"
        assert "auth" in episode.context_tags
        assert "security" in episode.context_tags

    async def test_get_missing_returns_none(self, memory: EpisodicMemory):
        """get() returns None for an unknown episode_id."""
        result = await memory.get("no-such-id")
        assert result is None

    async def test_store_with_metadata(self, memory: EpisodicMemory):
        """Metadata is stored and retrieved correctly."""
        ep_id = await memory.store(
            agent_id="agent-1",
            task_summary="Performance test",
            outcome="95ms p99 latency",
            metadata={"tokens": 1234, "model": "claude-3"},
        )
        episode = await memory.get(ep_id)
        assert episode is not None
        assert episode.metadata["tokens"] == 1234
        assert episode.metadata["model"] == "claude-3"

    async def test_default_tags_are_empty(self, memory: EpisodicMemory):
        """Episodes without tags have an empty context_tags list."""
        ep_id = await memory.store(
            agent_id="a",
            task_summary="Simple task",
            outcome="Done",
        )
        episode = await memory.get(ep_id)
        assert episode is not None
        assert episode.context_tags == []


# =============================================================================
# search()
# =============================================================================


class TestSearch:
    """Tests for keyword search."""

    async def test_search_by_keyword_in_summary(self, memory: EpisodicMemory):
        """search() matches against task_summary."""
        await memory.store("a", "Security audit of login module", "Found XSS")
        await memory.store("a", "Performance profiling of payment", "Slow query")

        results = await memory.search("Security audit")
        assert len(results) == 1
        assert results[0].task_summary == "Security audit of login module"

    async def test_search_by_keyword_in_outcome(self, memory: EpisodicMemory):
        """search() matches against outcome."""
        await memory.store("a", "Code review", "Found SQL injection risk")
        await memory.store("a", "Code review", "All tests passing")

        results = await memory.search("SQL injection")
        assert len(results) == 1
        assert "SQL injection" in results[0].outcome

    async def test_search_by_keyword_in_tags(self, memory: EpisodicMemory):
        """search() matches against context_tags."""
        await memory.store("a", "Task A", "Result A", context_tags=["security"])
        await memory.store("a", "Task B", "Result B", context_tags=["performance"])

        results = await memory.search("security")
        assert len(results) == 1

    async def test_search_empty_query_returns_recent(self, memory: EpisodicMemory):
        """Empty query returns most recent episodes."""
        await memory.store("a", "Task 1", "Result 1")
        await memory.store("a", "Task 2", "Result 2")

        results = await memory.search("", limit=10)
        assert len(results) == 2

    async def test_search_respects_limit(self, memory: EpisodicMemory):
        """search() returns at most `limit` results."""
        for i in range(10):
            await memory.store("a", f"Task {i}", f"Result {i}")

        results = await memory.search("", limit=3)
        assert len(results) <= 3

    async def test_search_ordered_newest_first(self, memory: EpisodicMemory):
        """Results are ordered most-recent first."""
        await memory.store("a", "Old task", "Old result")
        await asyncio.sleep(0.01)
        await memory.store("a", "New task", "New result")

        results = await memory.search("")
        assert results[0].task_summary == "New task"
        assert results[1].task_summary == "Old task"

    async def test_search_filter_by_agent(self, memory: EpisodicMemory):
        """agent_id filter restricts results to that agent."""
        await memory.store("agent-1", "Task by 1", "Done")
        await memory.store("agent-2", "Task by 2", "Done")

        results = await memory.search("", agent_id="agent-1")
        assert all(r.agent_id == "agent-1" for r in results)
        assert len(results) == 1

    async def test_search_filter_by_tags(self, memory: EpisodicMemory):
        """tags filter restricts results to episodes with any matching tag."""
        await memory.store("a", "Auth task", "Done", context_tags=["auth"])
        await memory.store("a", "Payment task", "Done", context_tags=["payment"])
        await memory.store("a", "Mixed task", "Done", context_tags=["auth", "payment"])

        results = await memory.search("", tags=["auth"])
        task_summaries = [r.task_summary for r in results]
        assert "Auth task" in task_summaries
        assert "Mixed task" in task_summaries
        assert "Payment task" not in task_summaries

    async def test_search_no_match_returns_empty(self, memory: EpisodicMemory):
        """search() returns empty list when no episodes match."""
        await memory.store("a", "Task", "Result")
        results = await memory.search("definitely-not-in-any-episode")
        assert results == []


# =============================================================================
# list_recent()
# =============================================================================


class TestListRecent:
    """Tests for list_recent()."""

    async def test_list_recent_all(self, memory: EpisodicMemory):
        """list_recent() returns episodes ordered newest first."""
        await memory.store("a", "First", "R1")
        await asyncio.sleep(0.01)
        await memory.store("a", "Second", "R2")

        recent = await memory.list_recent(limit=10)
        assert recent[0].task_summary == "Second"
        assert recent[1].task_summary == "First"

    async def test_list_recent_with_agent_filter(self, memory: EpisodicMemory):
        """list_recent() respects agent_id filter."""
        await memory.store("agent-1", "A1 task", "Done")
        await memory.store("agent-2", "A2 task", "Done")

        recent = await memory.list_recent(agent_id="agent-1")
        assert all(r.agent_id == "agent-1" for r in recent)


# =============================================================================
# delete()
# =============================================================================


class TestDelete:
    """Tests for episode deletion."""

    async def test_delete_existing_episode(self, memory: EpisodicMemory):
        """delete() removes the episode and returns True."""
        ep_id = await memory.store("a", "Task", "Result")
        result = await memory.delete(ep_id)
        assert result is True
        assert await memory.get(ep_id) is None

    async def test_delete_missing_returns_false(self, memory: EpisodicMemory):
        """delete() returns False for unknown episode IDs."""
        result = await memory.delete("no-such-id")
        assert result is False

    async def test_deleted_episode_not_in_search(self, memory: EpisodicMemory):
        """Deleted episodes do not appear in search results."""
        ep_id = await memory.store("a", "Unique task XYZ", "Done")
        await memory.delete(ep_id)

        results = await memory.search("Unique task XYZ")
        assert results == []


# =============================================================================
# store_eviction()
# =============================================================================


class TestStoreEviction:
    """Tests for context-window eviction archiving."""

    async def test_eviction_stored_as_episode(self, memory: EpisodicMemory):
        """store_eviction() creates an episode tagged with 'eviction'."""
        await memory.store_eviction(
            evicted_content="Some context that was evicted",
            role="user",
            step=5,
        )

        results = await memory.search("evicted", tags=["eviction"])
        assert len(results) == 1
        assert results[0].agent_id == "context_manager"
        assert "user" in results[0].context_tags

    async def test_eviction_metadata_contains_step(self, memory: EpisodicMemory):
        """store_eviction() records the step number in metadata."""
        await memory.store_eviction(
            evicted_content="Content from step 7",
            role="assistant",
            step=7,
        )

        results = await memory.list_recent(agent_id="context_manager")
        assert len(results) == 1
        assert results[0].metadata["step_number"] == 7


# =============================================================================
# VectorEpisodicMemory (requires chromadb — skipped if not installed)
# =============================================================================


@pytest.fixture
async def vector_memory():
    """Fresh VectorEpisodicMemory with in-process ChromaDB (if available)."""
    pytest.importorskip("chromadb", reason="chromadb not installed; skipping vector tests")
    from memory.episodic import VectorEpisodicMemory

    mem = VectorEpisodicMemory(":memory:")
    await mem.initialize()
    yield mem
    await mem.close()


class TestVectorEpisodicMemoryStore:
    """Tests that VectorEpisodicMemory satisfies the same contract as EpisodicMemory."""

    async def test_store_returns_episode_id(self, vector_memory):
        """store() returns a non-empty episode_id."""
        ep_id = await vector_memory.store(
            agent_id="agent-v",
            task_summary="Reviewed auth module",
            outcome="Found SQL injection",
        )
        assert isinstance(ep_id, str) and len(ep_id) > 0

    async def test_get_retrieves_stored_episode(self, vector_memory):
        """get() retrieves the episode stored via store()."""
        ep_id = await vector_memory.store(
            agent_id="agent-v",
            task_summary="Reviewed auth module",
            outcome="Found SQL injection",
            context_tags=["security", "sql"],
        )
        episode = await vector_memory.get(ep_id)
        assert episode is not None
        assert episode.episode_id == ep_id
        assert episode.agent_id == "agent-v"
        assert "security" in episode.context_tags

    async def test_get_missing_returns_none(self, vector_memory):
        """get() returns None for an unknown ID."""
        assert await vector_memory.get("no-such-id") is None

    async def test_store_with_metadata(self, vector_memory):
        """Metadata round-trips correctly through ChromaDB."""
        ep_id = await vector_memory.store(
            agent_id="a",
            task_summary="Performance test",
            outcome="95ms p99",
            metadata={"tokens": 42, "model": "test"},
        )
        episode = await vector_memory.get(ep_id)
        assert episode is not None
        assert episode.metadata["tokens"] == 42

    async def test_delete_existing_episode(self, vector_memory):
        """delete() returns True and removes the episode."""
        ep_id = await vector_memory.store("a", "Task", "Result")
        assert await vector_memory.delete(ep_id) is True
        assert await vector_memory.get(ep_id) is None

    async def test_delete_missing_returns_false(self, vector_memory):
        """delete() returns False for unknown IDs."""
        assert await vector_memory.delete("no-such-id") is False

    async def test_semantic_search_returns_relevant_results(self, vector_memory):
        """search() returns semantically related episodes."""
        await vector_memory.store("a", "Security audit of login module", "Found XSS")
        await vector_memory.store("a", "Performance test of payment service", "Slow query found")

        results = await vector_memory.search("security vulnerability", limit=5)
        assert len(results) >= 1
        # The security episode should rank higher than performance
        assert any("Security" in r.task_summary or "XSS" in r.outcome for r in results)

    async def test_empty_query_falls_back_to_list_recent(self, vector_memory):
        """Empty query returns recent episodes without error."""
        await vector_memory.store("a", "Task 1", "Result 1")
        await vector_memory.store("a", "Task 2", "Result 2")

        results = await vector_memory.search("", limit=10)
        assert len(results) == 2

    async def test_list_recent_newest_first(self, vector_memory):
        """list_recent() returns episodes sorted newest-first."""
        await vector_memory.store("a", "Old task", "Old result")
        await asyncio.sleep(0.05)
        await vector_memory.store("a", "New task", "New result")

        recent = await vector_memory.list_recent(limit=10)
        assert len(recent) == 2
        assert recent[0].task_summary == "New task"

    async def test_list_recent_agent_filter(self, vector_memory):
        """list_recent() respects agent_id filter."""
        await vector_memory.store("agent-1", "Task by 1", "Done")
        await vector_memory.store("agent-2", "Task by 2", "Done")

        recent = await vector_memory.list_recent(agent_id="agent-1")
        assert all(r.agent_id == "agent-1" for r in recent)

    async def test_store_eviction_works(self, vector_memory):
        """store_eviction() stores an episode tagged 'eviction'."""
        await vector_memory.store_eviction(
            evicted_content="Some context was evicted",
            role="user",
            step=3,
        )
        results = await vector_memory.list_recent(agent_id="context_manager")
        assert len(results) == 1
        assert "eviction" in results[0].context_tags


# =============================================================================
# Episode Model
# =============================================================================


class TestEpisodeModel:
    """Tests for the Episode Pydantic model."""

    def test_to_search_text_combines_fields(self):
        """to_search_text() concatenates searchable fields."""
        episode = Episode(
            agent_id="a",
            task_summary="Found bug",
            outcome="Fixed it",
            context_tags=["bug", "fix"],
        )
        text = episode.to_search_text()
        assert "Found bug" in text
        assert "Fixed it" in text
        assert "bug" in text
        assert "fix" in text

    def test_episode_id_auto_generated(self):
        """Each Episode gets a unique episode_id."""
        e1 = Episode(agent_id="a", task_summary="T", outcome="O")
        e2 = Episode(agent_id="a", task_summary="T", outcome="O")
        assert e1.episode_id != e2.episode_id
