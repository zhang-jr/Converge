"""Unit tests for tools/rbac.py — RBAC permission system."""

import pytest

from errors.exceptions import ToolPermissionError
from tools.rbac import (
    ROLE_ADMIN,
    ROLE_OPERATOR,
    ROLE_READ_ONLY,
    RBACManager,
    Role,
)


@pytest.fixture
def rbac() -> RBACManager:
    """Fresh RBACManager for each test."""
    return RBACManager()


# =============================================================================
# Role Management
# =============================================================================


class TestRoleManagement:
    """Tests for role creation and mutation."""

    def test_builtin_roles_exist(self, rbac: RBACManager):
        """Three built-in roles are always present."""
        assert rbac.get_role(ROLE_READ_ONLY) is not None
        assert rbac.get_role(ROLE_OPERATOR) is not None
        assert rbac.get_role(ROLE_ADMIN) is not None

    def test_create_custom_role(self, rbac: RBACManager):
        """Creating a new role returns the Role object."""
        role = rbac.create_role("analyst", "Analyst", "Read-only analysis role")

        assert role.role_id == "analyst"
        assert role.name == "Analyst"
        assert role.description == "Read-only analysis role"

    def test_create_duplicate_role_raises(self, rbac: RBACManager):
        """Creating a role with a duplicate ID raises ValueError."""
        rbac.create_role("analyst", "Analyst")
        with pytest.raises(ValueError, match="analyst"):
            rbac.create_role("analyst", "Another Analyst")

    def test_get_missing_role_raises(self, rbac: RBACManager):
        """Getting a non-existent role raises KeyError."""
        with pytest.raises(KeyError, match="nonexistent"):
            rbac.get_role("nonexistent")

    def test_list_roles_includes_builtins(self, rbac: RBACManager):
        """list_roles returns at least the three built-in roles."""
        roles = rbac.list_roles()
        assert ROLE_READ_ONLY in roles
        assert ROLE_OPERATOR in roles
        assert ROLE_ADMIN in roles

    def test_grant_tool_to_role(self, rbac: RBACManager):
        """Granting a tool to a role makes the role grant that permission."""
        rbac.create_role("analyst", "Analyst")
        rbac.grant_role_tool("analyst", "read_file")

        role = rbac.get_role("analyst")
        assert role.grants("read_file", "execute")

    def test_grant_tool_default_permission_is_execute(self, rbac: RBACManager):
        """Default permission when granting is 'execute'."""
        rbac.create_role("analyst", "Analyst")
        rbac.grant_role_tool("analyst", "read_file")

        role = rbac.get_role("analyst")
        assert role.grants("read_file", "execute")
        assert not role.grants("read_file", "dry_run")

    def test_grant_multiple_permissions(self, rbac: RBACManager):
        """Multiple permissions can be granted for one tool."""
        rbac.create_role("power", "Power User")
        rbac.grant_role_tool("power", "deploy", ["execute", "dry_run"])

        role = rbac.get_role("power")
        assert role.grants("deploy", "execute")
        assert role.grants("deploy", "dry_run")

    def test_revoke_tool_from_role(self, rbac: RBACManager):
        """Revoking a tool removes all permissions for that tool."""
        rbac.create_role("analyst", "Analyst")
        rbac.grant_role_tool("analyst", "read_file")
        rbac.revoke_role_tool("analyst", "read_file")

        role = rbac.get_role("analyst")
        assert not role.grants("read_file", "execute")

    def test_revoke_nonexistent_tool_is_noop(self, rbac: RBACManager):
        """Revoking a tool that was never granted is a no-op."""
        rbac.create_role("analyst", "Analyst")
        rbac.revoke_role_tool("analyst", "never_granted")  # Should not raise

    def test_grant_tool_to_missing_role_raises(self, rbac: RBACManager):
        """Granting a tool to a non-existent role raises KeyError."""
        with pytest.raises(KeyError, match="ghost"):
            rbac.grant_role_tool("ghost", "read_file")


# =============================================================================
# Subject Role Assignment
# =============================================================================


class TestSubjectAssignment:
    """Tests for assigning/revoking roles on subjects (agents)."""

    def test_assign_role_to_subject(self, rbac: RBACManager):
        """Assigning a role makes the subject have that role."""
        rbac.create_role("analyst", "Analyst")
        rbac.assign_role("agent-1", "analyst")

        roles = rbac.get_subject_roles("agent-1")
        assert any(r.role_id == "analyst" for r in roles)

    def test_assign_multiple_roles(self, rbac: RBACManager):
        """A subject can have multiple roles simultaneously."""
        rbac.create_role("reader", "Reader")
        rbac.create_role("writer", "Writer")
        rbac.assign_role("agent-1", "reader")
        rbac.assign_role("agent-1", "writer")

        roles = rbac.get_subject_roles("agent-1")
        role_ids = {r.role_id for r in roles}
        assert "reader" in role_ids
        assert "writer" in role_ids

    def test_assign_missing_role_raises(self, rbac: RBACManager):
        """Assigning a non-existent role raises KeyError."""
        with pytest.raises(KeyError, match="nonexistent"):
            rbac.assign_role("agent-1", "nonexistent")

    def test_revoke_role_from_subject(self, rbac: RBACManager):
        """Revoking a role removes it from the subject."""
        rbac.create_role("analyst", "Analyst")
        rbac.assign_role("agent-1", "analyst")
        rbac.revoke_role("agent-1", "analyst")

        roles = rbac.get_subject_roles("agent-1")
        assert not any(r.role_id == "analyst" for r in roles)

    def test_revoke_unassigned_role_is_noop(self, rbac: RBACManager):
        """Revoking a role that was never assigned is a no-op."""
        rbac.create_role("analyst", "Analyst")
        rbac.revoke_role("agent-1", "analyst")  # Should not raise

    def test_clear_subject_roles(self, rbac: RBACManager):
        """Clearing all roles leaves the subject with no roles."""
        rbac.create_role("reader", "Reader")
        rbac.create_role("writer", "Writer")
        rbac.assign_role("agent-1", "reader")
        rbac.assign_role("agent-1", "writer")
        rbac.clear_subject_roles("agent-1")

        assert rbac.get_subject_roles("agent-1") == []

    def test_get_roles_for_unknown_subject(self, rbac: RBACManager):
        """Getting roles for an unknown subject returns an empty list."""
        assert rbac.get_subject_roles("nobody") == []


# =============================================================================
# Permission Checks
# =============================================================================


class TestPermissionChecks:
    """Tests for has_permission() and check_permission()."""

    def test_no_roles_means_no_permission(self, rbac: RBACManager):
        """A subject with no roles has no permissions."""
        assert not rbac.has_permission("agent-1", "read_file")

    def test_granted_tool_permission(self, rbac: RBACManager):
        """Subject with a role granting a tool can use it."""
        rbac.create_role("analyst", "Analyst")
        rbac.grant_role_tool("analyst", "read_file")
        rbac.assign_role("agent-1", "analyst")

        assert rbac.has_permission("agent-1", "read_file")

    def test_ungranted_tool_denied(self, rbac: RBACManager):
        """Subject cannot use a tool their role does not grant."""
        rbac.create_role("analyst", "Analyst")
        rbac.grant_role_tool("analyst", "read_file")
        rbac.assign_role("agent-1", "analyst")

        assert not rbac.has_permission("agent-1", "delete_file")

    def test_admin_role_grants_all(self, rbac: RBACManager):
        """Admin role grants access to any tool without explicit grants."""
        rbac.assign_role("agent-1", ROLE_ADMIN)

        assert rbac.has_permission("agent-1", "read_file")
        assert rbac.has_permission("agent-1", "delete_file")
        assert rbac.has_permission("agent-1", "nuke_everything")

    def test_check_permission_passes_silently(self, rbac: RBACManager):
        """check_permission does not raise when permission is granted."""
        rbac.create_role("analyst", "Analyst")
        rbac.grant_role_tool("analyst", "read_file")
        rbac.assign_role("agent-1", "analyst")

        rbac.check_permission("agent-1", "read_file")  # Should not raise

    def test_check_permission_raises_on_deny(self, rbac: RBACManager):
        """check_permission raises ToolPermissionError when denied."""
        with pytest.raises(ToolPermissionError) as exc_info:
            rbac.check_permission("agent-1", "delete_file")

        assert exc_info.value.tool_name == "delete_file"
        assert exc_info.value.agent_id == "agent-1"

    def test_permission_type_distinction(self, rbac: RBACManager):
        """execute and dry_run permissions are checked independently."""
        rbac.create_role("previewer", "Previewer")
        rbac.grant_role_tool("previewer", "deploy", ["dry_run"])
        rbac.assign_role("agent-1", "previewer")

        assert rbac.has_permission("agent-1", "deploy", "dry_run")
        assert not rbac.has_permission("agent-1", "deploy", "execute")

    def test_permission_union_across_roles(self, rbac: RBACManager):
        """Permissions are the union of all assigned roles."""
        rbac.create_role("reader", "Reader")
        rbac.create_role("writer", "Writer")
        rbac.grant_role_tool("reader", "read_file")
        rbac.grant_role_tool("writer", "write_file")
        rbac.assign_role("agent-1", "reader")
        rbac.assign_role("agent-1", "writer")

        assert rbac.has_permission("agent-1", "read_file")
        assert rbac.has_permission("agent-1", "write_file")
        assert not rbac.has_permission("agent-1", "delete_file")


# =============================================================================
# Accessible Tools Listing
# =============================================================================


class TestAccessibleTools:
    """Tests for list_accessible_tools()."""

    def test_no_roles_returns_empty(self, rbac: RBACManager):
        """Subject with no roles has no accessible tools."""
        result = rbac.list_accessible_tools("agent-1")
        assert result == []

    def test_lists_granted_tools(self, rbac: RBACManager):
        """Returns tools granted by the subject's roles."""
        rbac.create_role("analyst", "Analyst")
        rbac.grant_role_tool("analyst", "read_file")
        rbac.grant_role_tool("analyst", "search_code")
        rbac.assign_role("agent-1", "analyst")

        tools = rbac.list_accessible_tools("agent-1")
        assert "read_file" in tools
        assert "search_code" in tools

    def test_admin_returns_none(self, rbac: RBACManager):
        """Admin role signals unrestricted access with None return."""
        rbac.assign_role("agent-1", ROLE_ADMIN)

        result = rbac.list_accessible_tools("agent-1")
        assert result is None  # None = all tools


# =============================================================================
# Role Dataclass
# =============================================================================


class TestRoleDataclass:
    """Tests for the Role dataclass directly."""

    def test_grants_returns_false_for_unknown_tool(self):
        role = Role(role_id="test", name="Test")
        assert not role.grants("any_tool", "execute")

    def test_grant_tool_adds_permission(self):
        role = Role(role_id="test", name="Test")
        role.grant_tool("read_file")
        assert role.grants("read_file", "execute")

    def test_grant_tool_is_additive(self):
        role = Role(role_id="test", name="Test")
        role.grant_tool("read_file", ["execute"])
        role.grant_tool("read_file", ["dry_run"])
        assert role.grants("read_file", "execute")
        assert role.grants("read_file", "dry_run")

    def test_revoke_tool_clears_all_permissions(self):
        role = Role(role_id="test", name="Test")
        role.grant_tool("read_file", ["execute", "dry_run"])
        role.revoke_tool("read_file")
        assert not role.grants("read_file", "execute")
        assert not role.grants("read_file", "dry_run")
