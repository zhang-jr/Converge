"""Built-in code review skill.

Provides a ready-to-use Skill that equips an agent with the tools and
prompt extensions needed to do systematic code review:

- ``ReadFileTool``  — read individual source files
- ``GlobTool``      — discover files matching a pattern
- ``GrepTool``      — search for patterns (anti-patterns, TODO, security issues)

The system_prompt_addon instructs the agent to produce structured review
output with severity labels and file:line citations.

Usage::

    from skills.builtin.code_review import create_code_review_skill
    from api.declarative import AgentFramework

    framework = AgentFramework()
    framework.load_skill(create_code_review_skill())
    result = await framework.run("Review the auth module for security issues")
"""

from __future__ import annotations

from core.state.models import ConvergenceCriterion
from skills.base import SkillBase
from tools.code.file_tools import ReadFileTool
from tools.code.search_tools import GlobTool, GrepTool

_CODE_REVIEW_SYSTEM_PROMPT = """
You are a senior code reviewer. Follow these principles:

PROCESS:
1. Start by discovering relevant files with GlobTool.
2. Read each file with ReadFileTool before commenting on it.
3. Use GrepTool to search for known anti-patterns (e.g. hardcoded secrets,
   unsafe eval, missing input validation, SQL string concatenation).
4. Record findings incrementally — do not wait until all files are read.

OUTPUT FORMAT:
For each issue, emit a structured finding:
  [SEVERITY] file/path.py:LINE — Short description
  Details: One sentence explaining the risk and recommended fix.

SEVERITY LEVELS:
  [CRITICAL] — security vulnerabilities, data loss risks
  [MAJOR]    — correctness bugs, missing error handling
  [MINOR]    — code style, performance hints, dead code

COMPLETION:
When all files in scope have been reviewed, emit:
  REVIEW COMPLETE: X critical, Y major, Z minor issues found.
"""


def create_code_review_skill() -> SkillBase:
    """Create the built-in code review skill.

    Returns:
        A configured SkillBase instance ready to be loaded into AgentFramework.

    Example::

        skill = create_code_review_skill()
        framework.load_skill(skill)
    """
    return SkillBase(
        name="code_review",
        description=(
            "Systematic code review: discovers files with Glob, reads them "
            "with ReadFile, and searches for anti-patterns with Grep. "
            "Produces structured findings with severity labels."
        ),
        tools=[ReadFileTool(), GlobTool(), GrepTool()],
        system_prompt_addon=_CODE_REVIEW_SYSTEM_PROMPT,
        convergence_criteria=[
            ConvergenceCriterion(
                criterion_type="custom_probe",
                description="Review complete: all files reviewed and issues documented",
                params={"completion_phrase": "REVIEW COMPLETE"},
            ),
        ],
    )
