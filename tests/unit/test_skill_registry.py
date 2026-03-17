"""Unit tests for skills/base.py, skills/registry.py, and skills/builtin/code_review.py."""

from __future__ import annotations

import pytest

from core.state.models import ConvergenceCriterion
from skills.base import SkillBase
from skills.builtin.code_review import create_code_review_skill
from skills.registry import SkillRegistry
from tools.code.file_tools import ReadFileTool
from tools.code.search_tools import GlobTool, GrepTool


# =============================================================================
# SkillBase
# =============================================================================


class TestSkillBase:
    """Tests for the SkillBase Pydantic model."""

    def test_minimal_skill_creation(self):
        """SkillBase can be created with just name and description."""
        skill = SkillBase(name="my-skill", description="Does something")
        assert skill.name == "my-skill"
        assert skill.description == "Does something"
        assert skill.tools == []
        assert skill.system_prompt_addon == ""
        assert skill.convergence_criteria == []

    def test_skill_with_tools(self):
        """SkillBase stores ToolBase instances in tools list."""
        tool = ReadFileTool()
        skill = SkillBase(name="reader", description="Reads files", tools=[tool])
        assert len(skill.tools) == 1
        assert skill.tools[0].name == "read_file"

    def test_skill_with_system_prompt_addon(self):
        """system_prompt_addon is stored correctly."""
        skill = SkillBase(
            name="expert",
            description="Domain expert",
            system_prompt_addon="You are an expert. Always cite sources.",
        )
        assert "cite sources" in skill.system_prompt_addon

    def test_skill_with_convergence_criteria(self):
        """convergence_criteria list is stored correctly."""
        criteria = [
            ConvergenceCriterion(
                criterion_type="lint_clean",
                description="No lint errors",
            )
        ]
        skill = SkillBase(
            name="linter",
            description="Linting skill",
            convergence_criteria=criteria,
        )
        assert len(skill.convergence_criteria) == 1
        assert skill.convergence_criteria[0].criterion_type == "lint_clean"

    def test_repr_shows_tool_names(self):
        """__repr__ includes tool names and criteria count."""
        skill = SkillBase(
            name="s",
            description="d",
            tools=[ReadFileTool()],
            convergence_criteria=[
                ConvergenceCriterion(criterion_type="lint_clean", description="x")
            ],
        )
        r = repr(skill)
        assert "read_file" in r
        assert "criteria=1" in r


# =============================================================================
# SkillRegistry
# =============================================================================


class TestSkillRegistry:
    """Tests for SkillRegistry."""

    def _make_skill(self, name: str = "test-skill") -> SkillBase:
        return SkillBase(name=name, description=f"{name} description")

    def test_register_and_get(self):
        """register() followed by get() retrieves the same skill."""
        registry = SkillRegistry()
        skill = self._make_skill("my-skill")
        registry.register(skill)
        retrieved = registry.get("my-skill")
        assert retrieved is skill

    def test_get_unknown_raises_key_error(self):
        """get() raises KeyError with descriptive message for unknown names."""
        registry = SkillRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_error_message_lists_available_skills(self):
        """KeyError message includes available skill names."""
        registry = SkillRegistry()
        registry.register(self._make_skill("skill-a"))
        with pytest.raises(KeyError, match="skill-a"):
            registry.get("nonexistent")

    def test_list_skills_returns_sorted_names(self):
        """list_skills() returns skill names in alphabetical order."""
        registry = SkillRegistry()
        registry.register(self._make_skill("zebra"))
        registry.register(self._make_skill("alpha"))
        registry.register(self._make_skill("middle"))
        assert registry.list_skills() == ["alpha", "middle", "zebra"]

    def test_list_skills_empty(self):
        """list_skills() returns empty list when no skills registered."""
        assert SkillRegistry().list_skills() == []

    def test_contains_registered_skill(self):
        """__contains__ returns True for registered skills."""
        registry = SkillRegistry()
        registry.register(self._make_skill("skill-x"))
        assert "skill-x" in registry

    def test_contains_unknown_skill(self):
        """__contains__ returns False for unregistered skills."""
        registry = SkillRegistry()
        assert "missing" not in registry

    def test_len_reflects_registered_count(self):
        """__len__ returns the number of registered skills."""
        registry = SkillRegistry()
        assert len(registry) == 0
        registry.register(self._make_skill("a"))
        registry.register(self._make_skill("b"))
        assert len(registry) == 2

    def test_register_overwrites_existing_skill(self):
        """Registering a skill with an existing name replaces the old one."""
        registry = SkillRegistry()
        skill_v1 = SkillBase(name="s", description="v1")
        skill_v2 = SkillBase(name="s", description="v2")
        registry.register(skill_v1)
        registry.register(skill_v2)
        assert registry.get("s").description == "v2"

    def test_repr_shows_skill_names(self):
        """__repr__ includes registered skill names."""
        registry = SkillRegistry()
        registry.register(self._make_skill("skill-a"))
        assert "skill-a" in repr(registry)


# =============================================================================
# Built-in: CodeReviewSkill
# =============================================================================


class TestCodeReviewSkill:
    """Tests for the built-in code_review skill factory."""

    def test_skill_has_correct_name(self):
        """create_code_review_skill() returns a skill named 'code_review'."""
        skill = create_code_review_skill()
        assert skill.name == "code_review"

    def test_skill_has_description(self):
        """Skill description is non-empty."""
        skill = create_code_review_skill()
        assert len(skill.description) > 0

    def test_skill_tools_are_correct_types(self):
        """Skill tools include ReadFileTool, GlobTool, and GrepTool."""
        skill = create_code_review_skill()
        tool_types = {type(t) for t in skill.tools}
        assert ReadFileTool in tool_types
        assert GlobTool in tool_types
        assert GrepTool in tool_types

    def test_skill_has_three_tools(self):
        """Skill has exactly 3 tools."""
        assert len(create_code_review_skill().tools) == 3

    def test_skill_has_nonempty_system_prompt_addon(self):
        """system_prompt_addon contains review instructions."""
        skill = create_code_review_skill()
        assert skill.system_prompt_addon.strip() != ""
        assert "SEVERITY" in skill.system_prompt_addon

    def test_skill_has_convergence_criterion(self):
        """Skill declares at least one convergence criterion."""
        skill = create_code_review_skill()
        assert len(skill.convergence_criteria) >= 1

    def test_skill_criterion_is_custom_probe(self):
        """The convergence criterion type is 'custom_probe'."""
        skill = create_code_review_skill()
        assert skill.convergence_criteria[0].criterion_type == "custom_probe"

    def test_factory_returns_independent_instances(self):
        """Each call to create_code_review_skill() returns a new instance."""
        s1 = create_code_review_skill()
        s2 = create_code_review_skill()
        assert s1 is not s2
        # But tools are independent objects too
        assert s1.tools[0] is not s2.tools[0]


# =============================================================================
# AgentFramework.load_skill integration
# =============================================================================


class TestAgentFrameworkLoadSkill:
    """Integration tests for AgentFramework.load_skill()."""

    def test_load_skill_registers_tools(self):
        """load_skill() registers all skill tools in the ToolRegistry."""
        from api.declarative import AgentFramework

        framework = AgentFramework()
        skill = create_code_review_skill()
        framework.load_skill(skill)

        registered = framework._tool_registry.list_tools()
        assert "read_file" in registered
        assert "glob" in registered
        assert "grep" in registered

    def test_load_skill_returns_self_for_chaining(self):
        """load_skill() returns the framework instance for method chaining."""
        from api.declarative import AgentFramework

        framework = AgentFramework()
        result = framework.load_skill(create_code_review_skill())
        assert result is framework

    def test_load_skill_by_name(self):
        """load_skill() can accept a skill name if pre-registered in registry."""
        from api.declarative import AgentFramework

        framework = AgentFramework()
        skill = create_code_review_skill()
        framework._skill_registry.register(skill)
        framework.load_skill("code_review")  # load by name, not instance

        assert "read_file" in framework._tool_registry.list_tools()

    def test_load_skill_unknown_name_raises(self):
        """load_skill() raises KeyError for unknown skill names."""
        from api.declarative import AgentFramework

        framework = AgentFramework()
        with pytest.raises(KeyError, match="not found"):
            framework.load_skill("nonexistent-skill")

    def test_skill_system_prompt_merged_into_config(self):
        """Loaded skill system_prompt_addon is appended to agent config prompt."""
        from api.declarative import AgentFramework
        from core.state.models import AgentConfig

        framework = AgentFramework()
        framework.load_skill(create_code_review_skill())

        base_config = AgentConfig(
            agent_id="a",
            system_prompt="Base instructions.",
        )
        merged = framework._apply_skills_to_config(base_config)
        assert merged is not None
        assert "Base instructions." in merged.system_prompt
        assert "SEVERITY" in merged.system_prompt  # from skill addon

    def test_skill_criteria_collected(self):
        """Loaded skill convergence criteria are available via _collect_skill_criteria."""
        from api.declarative import AgentFramework

        framework = AgentFramework()
        framework.load_skill(create_code_review_skill())
        criteria = framework._collect_skill_criteria()
        assert len(criteria) >= 1
        assert any(c.criterion_type == "custom_probe" for c in criteria)

    def test_multiple_skills_prompts_combined(self):
        """Multiple loaded skills contribute addons to the system prompt."""
        from api.declarative import AgentFramework

        framework = AgentFramework()
        skill_a = SkillBase(
            name="a",
            description="A",
            system_prompt_addon="Addon from A.",
        )
        skill_b = SkillBase(
            name="b",
            description="B",
            system_prompt_addon="Addon from B.",
        )
        framework.load_skill(skill_a)
        framework.load_skill(skill_b)

        merged = framework._apply_skills_to_config(None)
        assert merged is not None
        assert "Addon from A." in merged.system_prompt
        assert "Addon from B." in merged.system_prompt

    def test_no_skills_config_unchanged(self):
        """_apply_skills_to_config() returns original config when no skills loaded."""
        from api.declarative import AgentFramework
        from core.state.models import AgentConfig

        framework = AgentFramework()
        config = AgentConfig(agent_id="a", system_prompt="Original.")
        result = framework._apply_skills_to_config(config)
        assert result is config  # same object, unchanged
