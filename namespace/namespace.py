"""Namespace isolation for multi-tenant agent deployments.

A Namespace provides isolated execution environments for agents:
- State store key prefixing prevents cross-namespace state leakage.
- Agent membership tracking for quota enforcement.
- Scoped RBAC policies per namespace.

Analogous to Kubernetes Namespaces: resources in one namespace are
isolated from resources in another.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from errors.exceptions import AgentFrameworkError

if TYPE_CHECKING:
    from tools.rbac import RBACManager


class NamespaceConfig(BaseModel):
    """Configuration for a Namespace.

    Attributes:
        namespace_id: Unique identifier (auto-generated if not specified).
        name: Human-readable name (used as lookup key).
        description: Purpose of this namespace.
        state_prefix: Key prefix for state isolation (auto-derived if empty).
        default_tools: Tools available to all agents in this namespace.
        max_agents: Maximum concurrent agents (0 = unlimited).
        metadata: Arbitrary metadata.
    """

    namespace_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    description: str = ""
    state_prefix: str = ""
    default_tools: list[str] = Field(default_factory=list)
    max_agents: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        # Derive state_prefix from namespace_id if not explicitly set
        if not self.state_prefix:
            object.__setattr__(self, "state_prefix", f"ns/{self.namespace_id}")


class Namespace:
    """Runtime namespace instance.

    Provides isolated execution context for agents. Wraps state store
    key access with namespace prefixing, and tracks agent membership.

    Usage:
        ns = Namespace(NamespaceConfig(name="production"))
        ns.add_agent("agent-1")
        namespaced_key = ns.make_key("task-state")  # "ns/abc123/task-state"
    """

    def __init__(
        self,
        config: NamespaceConfig,
        rbac: RBACManager | None = None,
    ) -> None:
        self._config = config
        self._rbac = rbac
        self._agents: set[str] = set()
        self._created_at: datetime = datetime.utcnow()

    @property
    def namespace_id(self) -> str:
        """Unique namespace identifier."""
        return self._config.namespace_id

    @property
    def name(self) -> str:
        """Human-readable namespace name."""
        return self._config.name

    @property
    def config(self) -> NamespaceConfig:
        """Namespace configuration."""
        return self._config

    @property
    def rbac(self) -> RBACManager | None:
        """RBAC manager for this namespace."""
        return self._rbac

    def make_key(self, key: str) -> str:
        """Create a namespaced state key.

        Args:
            key: The bare key.

        Returns:
            Namespaced key with the namespace prefix.
        """
        return f"{self._config.state_prefix}/{key}"

    def strip_prefix(self, namespaced_key: str) -> str:
        """Strip the namespace prefix from a key.

        Args:
            namespaced_key: Key with namespace prefix.

        Returns:
            Bare key without prefix, or the original key if prefix not present.
        """
        prefix = self._config.state_prefix + "/"
        if namespaced_key.startswith(prefix):
            return namespaced_key[len(prefix):]
        return namespaced_key

    def owns_key(self, key: str) -> bool:
        """Check if a key belongs to this namespace.

        Args:
            key: The key to check.

        Returns:
            True if the key starts with this namespace's prefix.
        """
        return key.startswith(self._config.state_prefix + "/")

    def add_agent(self, agent_id: str) -> None:
        """Register an agent as belonging to this namespace.

        Args:
            agent_id: The agent to add.

        Raises:
            AgentFrameworkError: If the max_agents limit is exceeded.
        """
        limit = self._config.max_agents
        if limit > 0 and len(self._agents) >= limit:
            raise AgentFrameworkError(
                f"Namespace '{self.name}' agent limit ({limit}) exceeded",
                context={"namespace_id": self.namespace_id, "current_count": len(self._agents)},
            )
        self._agents.add(agent_id)

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from this namespace.

        Args:
            agent_id: The agent to remove.
        """
        self._agents.discard(agent_id)

    def has_agent(self, agent_id: str) -> bool:
        """Check if an agent belongs to this namespace.

        Args:
            agent_id: The agent to check.

        Returns:
            True if the agent is registered in this namespace.
        """
        return agent_id in self._agents

    def list_agents(self) -> list[str]:
        """List all agents registered in this namespace."""
        return list(self._agents)

    def to_dict(self) -> dict[str, Any]:
        """Serialize namespace info to dict for observability."""
        return {
            "namespace_id": self.namespace_id,
            "name": self.name,
            "description": self._config.description,
            "state_prefix": self._config.state_prefix,
            "agent_count": len(self._agents),
            "agents": list(self._agents),
            "created_at": self._created_at.isoformat(),
        }


class NamespaceManager:
    """Manages the lifecycle of multiple namespaces.

    Provides namespace creation, lookup, and deletion. Acts as the
    central registry for all namespaces in the framework.

    Usage:
        manager = NamespaceManager()
        prod = manager.create("production", "Production agents")
        staging = manager.create("staging", "Staging agents", max_agents=5)

        ns = manager.get("production")
        manager.delete("staging")
    """

    def __init__(self) -> None:
        self._namespaces: dict[str, Namespace] = {}   # name -> Namespace
        self._by_id: dict[str, Namespace] = {}        # namespace_id -> Namespace

    def create(
        self,
        name: str,
        description: str = "",
        default_tools: list[str] | None = None,
        max_agents: int = 0,
        rbac: RBACManager | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Namespace:
        """Create a new namespace.

        Args:
            name: Unique namespace name (used for lookup).
            description: Purpose description.
            default_tools: Tools available to all agents in this namespace.
            max_agents: Maximum concurrent agents (0 = unlimited).
            rbac: Optional RBAC manager scoped to this namespace.
            metadata: Arbitrary metadata.

        Returns:
            The created Namespace.

        Raises:
            ValueError: If a namespace with this name already exists.
        """
        if name in self._namespaces:
            raise ValueError(f"Namespace '{name}' already exists")

        config = NamespaceConfig(
            name=name,
            description=description,
            default_tools=default_tools or [],
            max_agents=max_agents,
            metadata=metadata or {},
        )
        ns = Namespace(config, rbac=rbac)
        self._namespaces[name] = ns
        self._by_id[ns.namespace_id] = ns
        return ns

    def get(self, name: str) -> Namespace:
        """Get a namespace by name.

        Args:
            name: Namespace name.

        Returns:
            The Namespace.

        Raises:
            KeyError: If namespace does not exist.
        """
        if name not in self._namespaces:
            raise KeyError(f"Namespace '{name}' does not exist")
        return self._namespaces[name]

    def get_by_id(self, namespace_id: str) -> Namespace:
        """Get a namespace by its unique ID.

        Args:
            namespace_id: The namespace UUID.

        Returns:
            The Namespace.

        Raises:
            KeyError: If namespace does not exist.
        """
        if namespace_id not in self._by_id:
            raise KeyError(f"Namespace ID '{namespace_id}' does not exist")
        return self._by_id[namespace_id]

    def find_by_agent(self, agent_id: str) -> Namespace | None:
        """Find the namespace that contains a specific agent.

        Args:
            agent_id: The agent to look up.

        Returns:
            The Namespace containing the agent, or None.
        """
        for ns in self._namespaces.values():
            if ns.has_agent(agent_id):
                return ns
        return None

    def delete(self, name: str) -> bool:
        """Delete a namespace.

        Args:
            name: Namespace name to delete.

        Returns:
            True if deleted, False if not found.
        """
        if name not in self._namespaces:
            return False
        ns = self._namespaces.pop(name)
        self._by_id.pop(ns.namespace_id, None)
        return True

    def list_namespaces(self) -> list[str]:
        """List all namespace names."""
        return list(self._namespaces.keys())

    def exists(self, name: str) -> bool:
        """Check if a namespace exists by name."""
        return name in self._namespaces

    def summary(self) -> list[dict[str, Any]]:
        """Get summary info for all namespaces."""
        return [ns.to_dict() for ns in self._namespaces.values()]
