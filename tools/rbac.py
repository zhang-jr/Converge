"""Role-Based Access Control for the Tool system.

Extends the ToolRegistry's flat permission model with roles, where each role
grants named permissions on specific tools. Subjects (agents) are assigned roles;
the framework checks roles before granting tool access.

Default stance: deny. Roles must be explicitly created and assigned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from errors.exceptions import ToolPermissionError

PermissionType = Literal["execute", "dry_run"]

# Built-in role identifiers
ROLE_READ_ONLY = "read_only"
ROLE_OPERATOR = "operator"
ROLE_ADMIN = "admin"


@dataclass
class Role:
    """A named set of tool permissions.

    Attributes:
        role_id: Unique identifier for the role.
        name: Human-readable role name.
        description: What this role is for.
        tool_permissions: Mapping of tool_name -> set of allowed permission types.
    """

    role_id: str
    name: str
    description: str = ""
    tool_permissions: dict[str, set[PermissionType]] = field(default_factory=dict)

    def grants(self, tool_name: str, permission: PermissionType) -> bool:
        """Check if this role grants a specific permission on a tool."""
        return permission in self.tool_permissions.get(tool_name, set())

    def grant_tool(
        self,
        tool_name: str,
        permissions: list[PermissionType] | None = None,
    ) -> None:
        """Grant permissions for a tool to this role.

        Args:
            tool_name: The tool to grant access to.
            permissions: Permission types to grant (default: ["execute"]).
        """
        if permissions is None:
            permissions = ["execute"]
        if tool_name not in self.tool_permissions:
            self.tool_permissions[tool_name] = set()
        self.tool_permissions[tool_name].update(permissions)

    def revoke_tool(self, tool_name: str) -> None:
        """Revoke all permissions for a tool from this role."""
        self.tool_permissions.pop(tool_name, None)


class RBACManager:
    """Role-Based Access Control manager.

    Manages roles and subject-to-role assignments. Works alongside
    ToolRegistry to enforce fine-grained tool permissions.

    Built-in roles:
        - read_only: Only tools explicitly granted to this role.
        - operator: Only tools explicitly granted to this role.
        - admin: All tools, no per-tool grants needed.

    Usage:
        rbac = RBACManager()
        rbac.create_role("analyst", "Code analysis tools")
        rbac.grant_role_tool("analyst", "read_file")
        rbac.grant_role_tool("analyst", "search_code")
        rbac.assign_role("agent-1", "analyst")

        rbac.check_permission("agent-1", "read_file")   # OK
        rbac.check_permission("agent-1", "delete_file") # ToolPermissionError
    """

    def __init__(self) -> None:
        self._roles: dict[str, Role] = {}
        self._subject_roles: dict[str, set[str]] = {}  # subject_id -> role_ids
        self._init_builtin_roles()

    def _init_builtin_roles(self) -> None:
        """Initialize built-in system roles."""
        self._roles[ROLE_READ_ONLY] = Role(
            role_id=ROLE_READ_ONLY,
            name="Read Only",
            description="Can only use tools explicitly granted to this role.",
        )
        self._roles[ROLE_OPERATOR] = Role(
            role_id=ROLE_OPERATOR,
            name="Operator",
            description="Can use tools explicitly granted to this role.",
        )
        self._roles[ROLE_ADMIN] = Role(
            role_id=ROLE_ADMIN,
            name="Admin",
            description="Unrestricted access to all tools.",
        )

    def create_role(
        self,
        role_id: str,
        name: str,
        description: str = "",
    ) -> Role:
        """Create a new role.

        Args:
            role_id: Unique identifier for the role.
            name: Human-readable name.
            description: Role description.

        Returns:
            The created Role.

        Raises:
            ValueError: If a role with the same ID already exists.
        """
        if role_id in self._roles:
            raise ValueError(f"Role '{role_id}' already exists")
        role = Role(role_id=role_id, name=name, description=description)
        self._roles[role_id] = role
        return role

    def get_role(self, role_id: str) -> Role:
        """Get a role by ID.

        Args:
            role_id: The role identifier.

        Returns:
            The Role.

        Raises:
            KeyError: If the role does not exist.
        """
        if role_id not in self._roles:
            raise KeyError(f"Role '{role_id}' does not exist")
        return self._roles[role_id]

    def list_roles(self) -> list[str]:
        """List all role IDs."""
        return list(self._roles.keys())

    def grant_role_tool(
        self,
        role_id: str,
        tool_name: str,
        permissions: list[PermissionType] | None = None,
    ) -> None:
        """Grant tool permissions to a role.

        Args:
            role_id: The role to update.
            tool_name: The tool to grant access to.
            permissions: Permission types to grant (default: ["execute"]).

        Raises:
            KeyError: If the role does not exist.
        """
        self.get_role(role_id).grant_tool(tool_name, permissions)

    def revoke_role_tool(self, role_id: str, tool_name: str) -> None:
        """Revoke all permissions for a tool from a role.

        Args:
            role_id: The role to update.
            tool_name: The tool to revoke.

        Raises:
            KeyError: If the role does not exist.
        """
        self.get_role(role_id).revoke_tool(tool_name)

    def assign_role(self, subject_id: str, role_id: str) -> None:
        """Assign a role to a subject (agent).

        Args:
            subject_id: The subject (agent) to assign the role to.
            role_id: The role to assign.

        Raises:
            KeyError: If the role does not exist.
        """
        if role_id not in self._roles:
            raise KeyError(f"Role '{role_id}' does not exist")
        if subject_id not in self._subject_roles:
            self._subject_roles[subject_id] = set()
        self._subject_roles[subject_id].add(role_id)

    def revoke_role(self, subject_id: str, role_id: str) -> None:
        """Revoke a role from a subject.

        Args:
            subject_id: The subject to revoke from.
            role_id: The role to revoke.
        """
        if subject_id in self._subject_roles:
            self._subject_roles[subject_id].discard(role_id)

    def clear_subject_roles(self, subject_id: str) -> None:
        """Remove all role assignments for a subject.

        Args:
            subject_id: The subject to clear.
        """
        self._subject_roles.pop(subject_id, None)

    def get_subject_roles(self, subject_id: str) -> list[Role]:
        """Get all roles assigned to a subject.

        Args:
            subject_id: The subject identifier.

        Returns:
            List of assigned roles.
        """
        role_ids = self._subject_roles.get(subject_id, set())
        return [self._roles[rid] for rid in role_ids if rid in self._roles]

    def has_permission(
        self,
        subject_id: str,
        tool_name: str,
        permission: PermissionType = "execute",
    ) -> bool:
        """Check if a subject has permission to use a tool.

        Admin role grants access to everything. Other roles must have an
        explicit tool grant.

        Args:
            subject_id: The subject to check.
            tool_name: The tool to check access for.
            permission: The permission type to check.

        Returns:
            True if the subject has permission.
        """
        for role in self.get_subject_roles(subject_id):
            if role.role_id == ROLE_ADMIN:
                return True
            if role.grants(tool_name, permission):
                return True
        return False

    def check_permission(
        self,
        subject_id: str,
        tool_name: str,
        permission: PermissionType = "execute",
    ) -> None:
        """Assert that a subject has permission; raise if not.

        Args:
            subject_id: The subject to check.
            tool_name: The tool to check access for.
            permission: The permission type to check.

        Raises:
            ToolPermissionError: If the subject lacks permission.
        """
        if not self.has_permission(subject_id, tool_name, permission):
            raise ToolPermissionError(
                f"Subject '{subject_id}' lacks '{permission}' permission for tool '{tool_name}'",
                tool_name=tool_name,
                required_permission=permission,
                agent_id=subject_id,
            )

    def list_accessible_tools(self, subject_id: str) -> list[str] | None:
        """List all tools accessible to a subject.

        Args:
            subject_id: The subject identifier.

        Returns:
            List of tool names the subject can access, or None if admin (all tools).
        """
        roles = self.get_subject_roles(subject_id)
        accessible: set[str] = set()
        for role in roles:
            if role.role_id == ROLE_ADMIN:
                return None  # None signals "all tools"
            accessible.update(role.tool_permissions.keys())
        return list(accessible)
