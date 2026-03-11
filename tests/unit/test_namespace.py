"""Unit tests for namespace/namespace.py — Namespace isolation."""

import pytest

from errors.exceptions import AgentFrameworkError
from namespace.namespace import Namespace, NamespaceConfig, NamespaceManager


# =============================================================================
# NamespaceConfig
# =============================================================================


class TestNamespaceConfig:
    """Tests for NamespaceConfig model."""

    def test_state_prefix_auto_derived(self):
        """state_prefix is auto-derived from namespace_id when not set."""
        config = NamespaceConfig(name="production")
        assert config.state_prefix == f"ns/{config.namespace_id}"

    def test_custom_state_prefix(self):
        """Custom state_prefix is preserved when explicitly set."""
        config = NamespaceConfig(name="production", state_prefix="custom/prefix")
        assert config.state_prefix == "custom/prefix"

    def test_unique_namespace_ids(self):
        """Each NamespaceConfig gets a unique namespace_id."""
        a = NamespaceConfig(name="a")
        b = NamespaceConfig(name="b")
        assert a.namespace_id != b.namespace_id


# =============================================================================
# Namespace Key Operations
# =============================================================================


class TestNamespaceKeys:
    """Tests for key prefixing and stripping."""

    @pytest.fixture
    def ns(self) -> Namespace:
        config = NamespaceConfig(name="prod", namespace_id="abc123")
        return Namespace(config)

    def test_make_key_prefixes_correctly(self, ns: Namespace):
        """make_key prepends the state prefix."""
        result = ns.make_key("my-state")
        assert result == "ns/abc123/my-state"

    def test_strip_prefix_removes_prefix(self, ns: Namespace):
        """strip_prefix removes the namespace prefix."""
        namespaced = "ns/abc123/my-state"
        assert ns.strip_prefix(namespaced) == "my-state"

    def test_strip_prefix_no_op_on_foreign_key(self, ns: Namespace):
        """strip_prefix returns the key unchanged if prefix is absent."""
        assert ns.strip_prefix("other/key") == "other/key"

    def test_owns_key_true_for_prefixed(self, ns: Namespace):
        """owns_key returns True for keys with this namespace's prefix."""
        assert ns.owns_key("ns/abc123/something")

    def test_owns_key_false_for_other_prefix(self, ns: Namespace):
        """owns_key returns False for keys with a different prefix."""
        assert not ns.owns_key("ns/other123/something")
        assert not ns.owns_key("unrelated/key")

    def test_roundtrip_make_strip(self, ns: Namespace):
        """make_key and strip_prefix are inverse operations."""
        original = "task/state"
        assert ns.strip_prefix(ns.make_key(original)) == original


# =============================================================================
# Agent Membership
# =============================================================================


class TestAgentMembership:
    """Tests for agent registration within a namespace."""

    @pytest.fixture
    def ns(self) -> Namespace:
        config = NamespaceConfig(name="prod")
        return Namespace(config)

    def test_add_agent(self, ns: Namespace):
        """Adding an agent registers it in the namespace."""
        ns.add_agent("agent-1")
        assert ns.has_agent("agent-1")

    def test_agent_not_in_namespace_by_default(self, ns: Namespace):
        """has_agent returns False for agents not registered."""
        assert not ns.has_agent("stranger")

    def test_remove_agent(self, ns: Namespace):
        """Removing an agent unregisters it."""
        ns.add_agent("agent-1")
        ns.remove_agent("agent-1")
        assert not ns.has_agent("agent-1")

    def test_remove_nonexistent_agent_is_noop(self, ns: Namespace):
        """Removing an agent that was never added is a no-op."""
        ns.remove_agent("nobody")  # Should not raise

    def test_list_agents(self, ns: Namespace):
        """list_agents returns all registered agents."""
        ns.add_agent("agent-1")
        ns.add_agent("agent-2")
        agents = ns.list_agents()
        assert "agent-1" in agents
        assert "agent-2" in agents

    def test_list_agents_empty(self, ns: Namespace):
        """list_agents returns empty list when no agents."""
        assert ns.list_agents() == []

    def test_max_agents_limit_enforced(self):
        """Adding agents beyond max_agents raises AgentFrameworkError."""
        config = NamespaceConfig(name="small", max_agents=2)
        ns = Namespace(config)

        ns.add_agent("agent-1")
        ns.add_agent("agent-2")

        with pytest.raises(AgentFrameworkError, match="limit"):
            ns.add_agent("agent-3")

    def test_max_agents_zero_means_unlimited(self):
        """max_agents=0 means no limit."""
        config = NamespaceConfig(name="unlimited", max_agents=0)
        ns = Namespace(config)

        for i in range(100):
            ns.add_agent(f"agent-{i}")

        assert len(ns.list_agents()) == 100


# =============================================================================
# Namespace Properties
# =============================================================================


class TestNamespaceProperties:
    """Tests for Namespace property accessors."""

    def test_namespace_id(self):
        config = NamespaceConfig(name="prod", namespace_id="fixed-id")
        ns = Namespace(config)
        assert ns.namespace_id == "fixed-id"

    def test_name(self):
        config = NamespaceConfig(name="staging")
        ns = Namespace(config)
        assert ns.name == "staging"

    def test_to_dict_contains_expected_keys(self):
        config = NamespaceConfig(name="prod", namespace_id="abc")
        ns = Namespace(config)
        ns.add_agent("agent-1")

        d = ns.to_dict()
        assert d["namespace_id"] == "abc"
        assert d["name"] == "prod"
        assert "agent-1" in d["agents"]
        assert "created_at" in d


# =============================================================================
# NamespaceManager
# =============================================================================


class TestNamespaceManager:
    """Tests for NamespaceManager lifecycle operations."""

    @pytest.fixture
    def manager(self) -> NamespaceManager:
        return NamespaceManager()

    def test_create_namespace(self, manager: NamespaceManager):
        """create() returns a Namespace with the given name."""
        ns = manager.create("production", "Production workloads")
        assert ns.name == "production"

    def test_create_duplicate_raises(self, manager: NamespaceManager):
        """Creating a namespace with a duplicate name raises ValueError."""
        manager.create("production")
        with pytest.raises(ValueError, match="production"):
            manager.create("production")

    def test_get_by_name(self, manager: NamespaceManager):
        """get() retrieves a namespace by name."""
        manager.create("staging")
        ns = manager.get("staging")
        assert ns.name == "staging"

    def test_get_missing_raises(self, manager: NamespaceManager):
        """get() raises KeyError for unknown names."""
        with pytest.raises(KeyError, match="ghost"):
            manager.get("ghost")

    def test_get_by_id(self, manager: NamespaceManager):
        """get_by_id() retrieves a namespace by its UUID."""
        ns = manager.create("prod")
        retrieved = manager.get_by_id(ns.namespace_id)
        assert retrieved.name == "prod"

    def test_get_by_id_missing_raises(self, manager: NamespaceManager):
        """get_by_id() raises KeyError for unknown IDs."""
        with pytest.raises(KeyError):
            manager.get_by_id("no-such-id")

    def test_delete_namespace(self, manager: NamespaceManager):
        """delete() removes a namespace and returns True."""
        manager.create("staging")
        result = manager.delete("staging")
        assert result is True
        assert not manager.exists("staging")

    def test_delete_missing_returns_false(self, manager: NamespaceManager):
        """delete() returns False for unknown namespaces."""
        assert manager.delete("ghost") is False

    def test_exists(self, manager: NamespaceManager):
        """exists() is True for created namespaces, False otherwise."""
        manager.create("prod")
        assert manager.exists("prod")
        assert not manager.exists("staging")

    def test_list_namespaces(self, manager: NamespaceManager):
        """list_namespaces() returns all created names."""
        manager.create("prod")
        manager.create("staging")
        names = manager.list_namespaces()
        assert "prod" in names
        assert "staging" in names

    def test_find_by_agent(self, manager: NamespaceManager):
        """find_by_agent() returns the namespace containing the agent."""
        prod = manager.create("prod")
        staging = manager.create("staging")

        prod.add_agent("agent-prod")
        staging.add_agent("agent-staging")

        assert manager.find_by_agent("agent-prod") is prod
        assert manager.find_by_agent("agent-staging") is staging

    def test_find_by_agent_returns_none_when_not_found(
        self, manager: NamespaceManager
    ):
        """find_by_agent() returns None when agent is in no namespace."""
        manager.create("prod")
        assert manager.find_by_agent("ghost") is None

    def test_create_with_max_agents(self, manager: NamespaceManager):
        """Namespaces can be created with a max_agents limit."""
        ns = manager.create("limited", max_agents=3)
        ns.add_agent("a")
        ns.add_agent("b")
        ns.add_agent("c")
        with pytest.raises(AgentFrameworkError):
            ns.add_agent("d")

    def test_isolation_between_namespaces(self, manager: NamespaceManager):
        """Two namespaces produce non-overlapping key prefixes."""
        ns1 = manager.create("ns1")
        ns2 = manager.create("ns2")

        key1 = ns1.make_key("shared-key")
        key2 = ns2.make_key("shared-key")
        assert key1 != key2
