"""Episodic memory for long-term experience storage.

Stores past task summaries and outcomes, retrievable by keyword search.
Backed by SQLite for persistence. Interface is designed to be upgraded
to vector search in Phase 3 without changing the caller API (ADR-005).

Analogous to human episodic memory: "I remember doing something like this before,
and the result was X."
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Episode(BaseModel):
    """A single episodic memory record.

    Attributes:
        episode_id: Unique identifier (UUID).
        agent_id: Which agent produced this episode.
        task_summary: Summary of what the task was about.
        outcome: What the result was.
        context_tags: Tags for filtering and retrieval.
        metadata: Additional structured data.
        created_at: When this episode was recorded.
    """

    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    task_summary: str
    outcome: str
    context_tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_search_text(self) -> str:
        """Combine all searchable fields into a single string."""
        return " ".join([
            self.task_summary,
            self.outcome,
            " ".join(self.context_tags),
        ])


class EpisodicMemory:
    """Persistent episodic memory using SQLite.

    Stores and retrieves past task experiences. Uses substring keyword
    search for retrieval. The interface is intentionally stable so that
    the backend can be swapped to a vector store in Phase 3 without
    changing callers.

    Usage:
        memory = EpisodicMemory(":memory:")
        await memory.initialize()

        ep_id = await memory.store(
            agent_id="analyst",
            task_summary="Analyzed 1000 Python files for security issues",
            outcome="Found 3 SQL injection vulnerabilities",
            context_tags=["code-analysis", "security"],
        )

        results = await memory.search("security analysis", limit=5)
        for ep in results:
            print(ep.task_summary, "->", ep.outcome)
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        """Initialize episodic memory.

        Args:
            db_path: Path to SQLite database. Use ":memory:" for in-process storage.
        """
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        """Initialize the database schema. Must be called before use."""
        await asyncio.to_thread(self._init_db)

    def _init_db(self) -> None:
        """Create tables and indexes (sync, run in thread pool)."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id   TEXT PRIMARY KEY,
                agent_id     TEXT NOT NULL,
                task_summary TEXT NOT NULL,
                outcome      TEXT NOT NULL,
                context_tags TEXT NOT NULL DEFAULT '[]',
                metadata     TEXT NOT NULL DEFAULT '{}',
                created_at   TEXT NOT NULL,
                search_text  TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_agent ON episodes (agent_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes (created_at DESC)"
        )
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or lazily create the database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    async def store(
        self,
        agent_id: str,
        task_summary: str,
        outcome: str,
        context_tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a new episodic memory.

        Args:
            agent_id: ID of the agent whose experience this is.
            task_summary: What the task was about.
            outcome: What happened / what the result was.
            context_tags: Tags for categorization and retrieval.
            metadata: Additional structured data (e.g., token counts, timing).

        Returns:
            episode_id of the stored episode.
        """
        episode = Episode(
            agent_id=agent_id,
            task_summary=task_summary,
            outcome=outcome,
            context_tags=context_tags or [],
            metadata=metadata or {},
        )
        await asyncio.to_thread(self._insert_episode, episode)
        return episode.episode_id

    def _insert_episode(self, episode: Episode) -> None:
        """Insert an episode into the database (sync)."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO episodes
                (episode_id, agent_id, task_summary, outcome,
                 context_tags, metadata, created_at, search_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode.episode_id,
                episode.agent_id,
                episode.task_summary,
                episode.outcome,
                json.dumps(episode.context_tags),
                json.dumps(episode.metadata),
                episode.created_at.isoformat(),
                episode.to_search_text(),
            ),
        )
        conn.commit()

    async def store_eviction(self, evicted_content: str, role: str, step: int) -> None:
        """Store a context-window eviction fragment as an episodic memory.

        Called by ContextWindowManager when messages are evicted from the
        context window to avoid total information loss.

        Args:
            evicted_content: The content of the evicted message.
            role: The role of the evicted message (user/assistant/tool).
            step: The reconcile loop step number of the evicted message.
        """
        await self.store(
            agent_id="context_manager",
            task_summary=f"Evicted {role} message from step {step}",
            outcome=evicted_content[:500],
            context_tags=["eviction", role],
            metadata={"step_number": step, "full_content": evicted_content},
        )

    async def search(
        self,
        query: str,
        agent_id: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[Episode]:
        """Search episodic memory by keyword.

        Performs substring matching against the combined search text
        (task_summary + outcome + context_tags). An empty query returns
        most recent episodes.

        Args:
            query: Keyword query string.
            agent_id: Optional filter by agent.
            tags: Optional filter by tags (episode must have at least one matching tag).
            limit: Maximum results to return.

        Returns:
            List of matching episodes, most recent first.
        """
        return await asyncio.to_thread(self._search_sync, query, agent_id, tags, limit)

    def _search_sync(
        self,
        query: str,
        agent_id: str | None,
        tags: list[str] | None,
        limit: int,
    ) -> list[Episode]:
        """Sync search implementation."""
        conn = self._get_conn()
        sql = "SELECT * FROM episodes WHERE 1=1"
        params: list[Any] = []

        if query:
            sql += " AND search_text LIKE ?"
            params.append(f"%{query}%")

        if agent_id:
            sql += " AND agent_id = ?"
            params.append(agent_id)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        episodes = [self._row_to_episode(row) for row in rows]

        # Post-filter by tags if requested (any tag must match)
        if tags:
            episodes = [e for e in episodes if any(t in e.context_tags for t in tags)]

        return episodes

    def _row_to_episode(self, row: sqlite3.Row) -> Episode:
        """Convert a database row to an Episode model."""
        return Episode(
            episode_id=row["episode_id"],
            agent_id=row["agent_id"],
            task_summary=row["task_summary"],
            outcome=row["outcome"],
            context_tags=json.loads(row["context_tags"]),
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    async def get(self, episode_id: str) -> Episode | None:
        """Retrieve a specific episode by ID.

        Args:
            episode_id: The episode identifier.

        Returns:
            Episode if found, None otherwise.
        """
        return await asyncio.to_thread(self._get_sync, episode_id)

    def _get_sync(self, episode_id: str) -> Episode | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_episode(row)

    async def list_recent(
        self,
        agent_id: str | None = None,
        limit: int = 20,
    ) -> list[Episode]:
        """List most recent episodes.

        Args:
            agent_id: Optional filter by agent.
            limit: Maximum number of results.

        Returns:
            List of recent episodes, most recent first.
        """
        return await self.search("", agent_id=agent_id, limit=limit)

    async def delete(self, episode_id: str) -> bool:
        """Delete an episode by ID.

        Args:
            episode_id: The episode to delete.

        Returns:
            True if deleted, False if not found.
        """
        return await asyncio.to_thread(self._delete_sync, episode_id)

    def _delete_sync(self, episode_id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM episodes WHERE episode_id = ?", (episode_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# =============================================================================
# Vector backend
# =============================================================================


class VectorEpisodicMemory:
    """Episodic memory using ChromaDB for semantic (embedding-based) search.

    Provides the same interface as :class:`EpisodicMemory` so callers can
    switch backends without code changes.  Requires the ``[vector]``
    optional dependency group::

        pip install agent-framework[vector]

    Usage::

        memory = VectorEpisodicMemory()   # in-process ephemeral
        await memory.initialize()

        ep_id = await memory.store(
            agent_id="analyst",
            task_summary="Reviewed auth module for SQL injection",
            outcome="Found 1 critical issue in login.py",
            context_tags=["security", "sql"],
        )

        results = await memory.search("SQL injection auth")
        # Returns semantically similar episodes, not just substring matches.
    """

    def __init__(
        self,
        path: str = ":memory:",
        collection_name: str = "episodes",
    ) -> None:
        """Initialize vector episodic memory.

        Args:
            path: Storage path.  Use ``":memory:"`` for an ephemeral in-process
                store; provide a file-system path for persistence.
            collection_name: ChromaDB collection name.
        """
        self._path = path
        self._collection_name = collection_name
        self._client: Any = None
        self._collection: Any = None

    async def initialize(self) -> None:
        """Create the ChromaDB client and collection. Must be called before use.

        Raises:
            ImportError: If ``chromadb`` is not installed.
        """
        await asyncio.to_thread(self._init_chromadb)

    def _init_chromadb(self) -> None:
        """Create chromadb client and get/create collection (sync)."""
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "chromadb is required for VectorEpisodicMemory. "
                "Install with: pip install agent-framework[vector]"
            ) from exc

        if self._path == ":memory:":
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=self._path)

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def store(
        self,
        agent_id: str,
        task_summary: str,
        outcome: str,
        context_tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a new episodic memory and return its episode_id.

        Args:
            agent_id: ID of the agent whose experience this is.
            task_summary: What the task was about.
            outcome: What happened / what the result was.
            context_tags: Tags for categorization and filtering.
            metadata: Additional structured data.

        Returns:
            episode_id of the stored episode.
        """
        episode = Episode(
            agent_id=agent_id,
            task_summary=task_summary,
            outcome=outcome,
            context_tags=context_tags or [],
            metadata=metadata or {},
        )
        await asyncio.to_thread(self._add_episode, episode)
        return episode.episode_id

    def _add_episode(self, episode: Episode) -> None:
        """Insert episode into ChromaDB collection (sync)."""
        import json

        self._collection.add(
            ids=[episode.episode_id],
            documents=[episode.to_search_text()],
            metadatas=[{
                "agent_id": episode.agent_id,
                "task_summary": episode.task_summary,
                "outcome": episode.outcome,
                "context_tags_json": json.dumps(episode.context_tags),
                "metadata_json": json.dumps(episode.metadata),
                "created_at": episode.created_at.isoformat(),
                "created_at_ts": episode.created_at.timestamp(),
            }],
        )

    async def store_eviction(
        self,
        evicted_content: str,
        role: str,
        step: int,
    ) -> None:
        """Store a context-window eviction fragment as an episodic memory.

        Args:
            evicted_content: The content of the evicted message.
            role: The role of the evicted message (user/assistant/tool).
            step: The reconcile loop step number of the evicted message.
        """
        await self.store(
            agent_id="context_manager",
            task_summary=f"Evicted {role} message from step {step}",
            outcome=evicted_content[:500],
            context_tags=["eviction", role],
            metadata={"step_number": step, "full_content": evicted_content},
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        agent_id: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[Episode]:
        """Search episodic memory using vector similarity.

        An empty query falls back to :meth:`list_recent` (no meaningful
        embedding can be produced from an empty string).

        Args:
            query: Natural-language query for semantic similarity search.
            agent_id: Optional filter by agent.
            tags: Optional post-filter by tags (any match).
            limit: Maximum results to return.

        Returns:
            List of matching episodes, most similar first.
        """
        if not query:
            return await self.list_recent(agent_id=agent_id, limit=limit)
        return await asyncio.to_thread(self._query_sync, query, agent_id, tags, limit)

    def _query_sync(
        self,
        query: str,
        agent_id: str | None,
        tags: list[str] | None,
        limit: int,
    ) -> list[Episode]:
        """ChromaDB vector query (sync)."""
        where: dict[str, Any] = {}
        if agent_id:
            where["agent_id"] = {"$eq": agent_id}

        kwargs: dict[str, Any] = {
            "query_texts": [query],
            # Over-fetch so tag post-filter still returns up to `limit`.
            "n_results": min(limit * 3, 100),
        }
        if where:
            kwargs["where"] = where

        try:
            results = self._collection.query(**kwargs)
        except Exception:
            return []

        ids: list[str] = results.get("ids", [[]])[0]
        metadatas: list[dict[str, Any]] = results.get("metadatas", [[]])[0]

        episodes = [
            self._meta_to_episode(ep_id, meta)
            for ep_id, meta in zip(ids, metadatas)
        ]

        if tags:
            episodes = [e for e in episodes if any(t in e.context_tags for t in tags)]

        return episodes[:limit]

    def _meta_to_episode(self, episode_id: str, meta: dict[str, Any]) -> Episode:
        """Reconstruct an Episode from ChromaDB metadata."""
        import json

        return Episode(
            episode_id=episode_id,
            agent_id=meta["agent_id"],
            task_summary=meta["task_summary"],
            outcome=meta["outcome"],
            context_tags=json.loads(meta.get("context_tags_json", "[]")),
            metadata=json.loads(meta.get("metadata_json", "{}")),
            created_at=datetime.fromisoformat(meta["created_at"]),
        )

    async def get(self, episode_id: str) -> Episode | None:
        """Retrieve a specific episode by ID.

        Args:
            episode_id: The episode identifier.

        Returns:
            Episode if found, None otherwise.
        """
        return await asyncio.to_thread(self._get_sync, episode_id)

    def _get_sync(self, episode_id: str) -> Episode | None:
        """Sync get by ID."""
        try:
            result = self._collection.get(ids=[episode_id])
        except Exception:
            return None

        ids: list[str] = result.get("ids", [])
        metadatas: list[dict[str, Any]] = result.get("metadatas", [])

        if not ids:
            return None

        return self._meta_to_episode(ids[0], metadatas[0])

    async def list_recent(
        self,
        agent_id: str | None = None,
        limit: int = 20,
    ) -> list[Episode]:
        """List most recent episodes, sorted by creation time.

        Args:
            agent_id: Optional filter by agent.
            limit: Maximum number of results.

        Returns:
            List of recent episodes, most recent first.
        """
        return await asyncio.to_thread(self._list_recent_sync, agent_id, limit)

    def _list_recent_sync(
        self,
        agent_id: str | None,
        limit: int,
    ) -> list[Episode]:
        """Sync list_recent: get all (optionally filtered) and sort by time."""
        kwargs: dict[str, Any] = {}
        if agent_id:
            kwargs["where"] = {"agent_id": {"$eq": agent_id}}

        try:
            result = self._collection.get(**kwargs)
        except Exception:
            return []

        ids: list[str] = result.get("ids", [])
        metadatas: list[dict[str, Any]] = result.get("metadatas", []) or []

        episodes = [
            self._meta_to_episode(ep_id, meta)
            for ep_id, meta in zip(ids, metadatas)
        ]

        # Sort by created_at timestamp descending (newest first).
        episodes.sort(key=lambda e: e.created_at.timestamp(), reverse=True)

        return episodes[:limit]

    async def delete(self, episode_id: str) -> bool:
        """Delete an episode by ID.

        Args:
            episode_id: The episode to delete.

        Returns:
            True if deleted, False if not found.
        """
        return await asyncio.to_thread(self._delete_sync, episode_id)

    def _delete_sync(self, episode_id: str) -> bool:
        """Sync delete by ID."""
        try:
            result = self._collection.get(ids=[episode_id])
        except Exception:
            return False

        if not result.get("ids"):
            return False

        self._collection.delete(ids=[episode_id])
        return True

    async def close(self) -> None:
        """Release ChromaDB client resources."""
        # EphemeralClient has no explicit close; set to None for GC.
        self._client = None
        self._collection = None
