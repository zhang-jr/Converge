"""SkillRegistry — central catalog of available skills.

Skills are registered by name and retrieved by name when loaded into an
AgentFramework instance.  The registry is intentionally simple: a thin
dict wrapper that raises a descriptive error on missing lookups.

Usage::

    from skills.registry import SkillRegistry
    from skills.builtin.code_review import create_code_review_skill

    registry = SkillRegistry()
    registry.register(create_code_review_skill())

    skill = registry.get("code_review")
"""

from __future__ import annotations

from skills.base import SkillBase


class SkillRegistry:
    """In-memory registry of SkillBase instances keyed by name.

    Thread-safety: not thread-safe by default. Skills are expected to be
    registered during application startup, not concurrently.
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillBase] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, skill: SkillBase) -> None:
        """Register a skill, overwriting any existing skill with the same name.

        Args:
            skill: The SkillBase instance to register.
        """
        self._skills[skill.name] = skill

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, name: str) -> SkillBase:
        """Retrieve a skill by name.

        Args:
            name: The skill's unique identifier.

        Returns:
            The registered SkillBase instance.

        Raises:
            KeyError: If no skill with that name is registered.
        """
        if name not in self._skills:
            available = ", ".join(sorted(self._skills)) or "(none)"
            raise KeyError(
                f"Skill '{name}' not found in registry. "
                f"Available skills: {available}"
            )
        return self._skills[name]

    def list_skills(self) -> list[str]:
        """Return sorted list of all registered skill names."""
        return sorted(self._skills.keys())

    def __contains__(self, name: object) -> bool:
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        return f"SkillRegistry({self.list_skills()})"
