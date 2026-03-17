"""SkillBase — the atomic unit of a reusable agent capability.

A Skill bundles together:
- A set of Tools the agent needs to accomplish a domain task.
- A ``system_prompt_addon`` that extends the agent's persona / instructions.
- Optional ``convergence_criteria`` that define "done" for that skill's task.

Skills are composable: multiple skills can be loaded into a single
AgentFramework instance, and their tool sets, prompts, and criteria are
merged transparently.

Analogous to a kubectl plugin or a Helm chart: a named, versioned package of
capabilities that can be dropped into a running framework with one line.

Usage::

    from skills.base import SkillBase
    from tools.code.file_tools import ReadFileTool

    skill = SkillBase(
        name="file_reader",
        description="Reads source files for analysis",
        tools=[ReadFileTool()],
        system_prompt_addon="Always cite file paths when referencing code.",
    )
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SkillBase(BaseModel):
    """A reusable capability module for agents.

    Attributes:
        name: Unique identifier for this skill (used as registry key).
        description: Human-readable description of what the skill provides.
        tools: ToolBase instances that will be registered when the skill is loaded.
        system_prompt_addon: Additional instructions appended to the agent's
            system prompt when this skill is active.
        convergence_criteria: ConvergenceCriterion list merged into the
            DesiredState when the skill is loaded.  Agents stop when all
            criteria are satisfied (see ConvergenceCriteriaProbe).
    """

    # Allow arbitrary types so ToolBase (ABC) and ConvergenceCriterion can live here.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    tools: list[Any] = Field(default_factory=list)
    system_prompt_addon: str = ""
    convergence_criteria: list[Any] = Field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"SkillBase(name={self.name!r}, "
            f"tools={[t.name for t in self.tools]}, "
            f"criteria={len(self.convergence_criteria)})"
        )
