"""Tool Registry for managing available tools.

Agents must obtain tools through the registry, not via direct imports.
This enables permission checking, auditing, and dynamic tool management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from errors.exceptions import ToolPermissionError

if TYPE_CHECKING:
    from tools.base import ToolBase


class ToolRegistry:
    """Central registry for all available tools.

    Provides tool registration, lookup, and permission-aware access.
    Agents should use get_tool() or get_tools() to access tools.

    Usage:
        registry = ToolRegistry()
        registry.register(MyTool())
        tool = registry.get_tool("my_tool", agent_id="agent-1")
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolBase] = {}
        self._permissions: dict[str, set[str]] = {}

    def register(self, tool: ToolBase) -> None:
        """Register a tool in the registry.

        Args:
            tool: The tool instance to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def unregister(self, tool_name: str) -> bool:
        """Unregister a tool from the registry.

        Args:
            tool_name: Name of the tool to unregister.

        Returns:
            True if the tool was unregistered, False if it wasn't registered.
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False

    def get_tool(
        self,
        tool_name: str,
        agent_id: str | None = None,
        check_permission: bool = True,
    ) -> ToolBase:
        """Get a tool by name with optional permission check.

        Args:
            tool_name: Name of the tool to retrieve.
            agent_id: ID of the agent requesting the tool.
            check_permission: Whether to check permissions.

        Returns:
            The requested tool.

        Raises:
            KeyError: If the tool is not registered.
            ToolPermissionError: If the agent lacks permission.
        """
        if tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_name}' is not registered")

        if check_permission and agent_id:
            if not self.has_permission(agent_id, tool_name):
                raise ToolPermissionError(
                    f"Agent '{agent_id}' lacks permission for tool '{tool_name}'",
                    tool_name=tool_name,
                    agent_id=agent_id,
                )

        return self._tools[tool_name]

    def get_tools(
        self,
        tool_names: list[str],
        agent_id: str | None = None,
        check_permission: bool = True,
    ) -> list[ToolBase]:
        """Get multiple tools by name.

        Args:
            tool_names: Names of the tools to retrieve.
            agent_id: ID of the agent requesting the tools.
            check_permission: Whether to check permissions.

        Returns:
            List of requested tools.
        """
        return [
            self.get_tool(name, agent_id, check_permission)
            for name in tool_names
        ]

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of registered tool names.
        """
        return list(self._tools.keys())

    def list_tools_for_agent(self, agent_id: str) -> list[str]:
        """List tools accessible by a specific agent.

        Args:
            agent_id: ID of the agent.

        Returns:
            List of tool names the agent can access.
        """
        if agent_id not in self._permissions:
            return []
        return [
            name
            for name in self._tools
            if name in self._permissions[agent_id]
        ]

    def grant_permission(self, agent_id: str, tool_name: str) -> None:
        """Grant an agent permission to use a tool.

        Args:
            agent_id: ID of the agent.
            tool_name: Name of the tool to grant access to.
        """
        if agent_id not in self._permissions:
            self._permissions[agent_id] = set()
        self._permissions[agent_id].add(tool_name)

    def grant_permissions(self, agent_id: str, tool_names: list[str]) -> None:
        """Grant an agent permission to use multiple tools.

        Args:
            agent_id: ID of the agent.
            tool_names: Names of the tools to grant access to.
        """
        for tool_name in tool_names:
            self.grant_permission(agent_id, tool_name)

    def revoke_permission(self, agent_id: str, tool_name: str) -> None:
        """Revoke an agent's permission to use a tool.

        Args:
            agent_id: ID of the agent.
            tool_name: Name of the tool to revoke access to.
        """
        if agent_id in self._permissions:
            self._permissions[agent_id].discard(tool_name)

    def has_permission(self, agent_id: str, tool_name: str) -> bool:
        """Check if an agent has permission to use a tool.

        Args:
            agent_id: ID of the agent.
            tool_name: Name of the tool.

        Returns:
            True if the agent has permission.
        """
        if agent_id not in self._permissions:
            return False
        return tool_name in self._permissions[agent_id]

    def clear_permissions(self, agent_id: str | None = None) -> None:
        """Clear permissions for an agent or all agents.

        Args:
            agent_id: If provided, clear only this agent's permissions.
                     If None, clear all permissions.
        """
        if agent_id is None:
            self._permissions.clear()
        elif agent_id in self._permissions:
            del self._permissions[agent_id]

    def get_tool_info(self, tool_name: str) -> dict:
        """Get metadata about a tool.

        Args:
            tool_name: Name of the tool.

        Returns:
            Dictionary with tool metadata.

        Raises:
            KeyError: If the tool is not registered.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            raise KeyError(f"Tool '{tool_name}' is not registered")
        return tool.to_dict()

    def get_all_tool_info(self) -> list[dict]:
        """Get metadata about all registered tools.

        Returns:
            List of dictionaries with tool metadata.
        """
        return [tool.to_dict() for tool in self._tools.values()]


# Global default registry instance
_default_registry: ToolRegistry | None = None


def get_default_registry() -> ToolRegistry:
    """Get the default global tool registry.

    Returns:
        The default ToolRegistry instance.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry


def reset_default_registry() -> None:
    """Reset the default global tool registry.

    Useful for testing.
    """
    global _default_registry
    _default_registry = None
