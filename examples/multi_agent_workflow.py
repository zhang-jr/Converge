"""Multi-Agent Workflow Example.

Demonstrates Phase 2 capabilities:
1. Declarative Workflow with multiple steps and dependency tracking (DAG mode)
2. Multi-Agent pipeline: research → draft → review
3. RBAC: agents get only the tools they need
4. Namespace isolation: workflow runs in its own namespace
5. Working memory: agents share a scratchpad
6. Episodic memory: task outcomes are stored for future retrieval

Run:
    python -m examples.multi_agent_workflow
"""

from __future__ import annotations

import asyncio

from core.agent.multi_agent import MultiAgentOrchestrator, pipeline, supervisor
from core.runtime.agent_runtime import AgentRuntime
from core.state.models import AgentConfig, DesiredState
from core.workflow.controller import WorkflowController
from core.workflow.workflow import WorkflowSpec, WorkflowStep
from memory.context_manager import ContextWindowManager
from memory.episodic import EpisodicMemory
from memory.working import WorkingMemory
from namespace.namespace import NamespaceManager
from tools.rbac import ROLE_ADMIN, ROLE_READ_ONLY, RBACManager


# =============================================================================
# Example 1: Declarative Workflow (DAG mode)
# =============================================================================

async def run_workflow_example() -> None:
    """Run a 3-step DAG workflow: analyze → plan → execute."""
    print("\n=== Example 1: Declarative Workflow (DAG mode) ===")

    analyst_config = AgentConfig(
        agent_id="analyst",
        name="Code Analyst",
        description="Analyzes code for issues",
        tools=[],
    )
    planner_config = AgentConfig(
        agent_id="planner",
        name="Solution Planner",
        description="Creates an action plan based on analysis",
        tools=[],
    )
    executor_config = AgentConfig(
        agent_id="executor",
        name="Task Executor",
        description="Executes the plan and produces the final result",
        tools=[],
    )

    spec = WorkflowSpec(
        name="Code Review Pipeline",
        description="Analyze code, plan improvements, then execute",
        execution_mode="dag",
        steps=[
            WorkflowStep(
                step_id="analyze",
                name="Analyze",
                agent_config=analyst_config,
                goal="Analyze the codebase and identify top 3 improvement areas",
            ),
            WorkflowStep(
                step_id="plan",
                name="Plan",
                agent_config=planner_config,
                goal="Create a prioritized improvement plan",
                depends_on=["analyze"],
                context_from=["analyze"],
            ),
            WorkflowStep(
                step_id="execute",
                name="Execute",
                agent_config=executor_config,
                goal="Implement the top priority improvement from the plan",
                depends_on=["plan"],
                context_from=["analyze", "plan"],
                on_failure="skip",  # Non-fatal: skip if execution fails
            ),
        ],
    )

    async with AgentRuntime(db_path=":memory:") as runtime:
        controller = WorkflowController(runtime)
        result = await controller.run(spec)

    print(f"Workflow status: {result.status}")
    print(f"Steps: {result.completed_steps} completed, "
          f"{result.failed_steps} failed, "
          f"{result.skipped_steps} skipped")
    for step_id, step_status in result.step_statuses.items():
        print(f"  [{step_status.status:10}] {step_status.step_name}")


# =============================================================================
# Example 2: Multi-Agent Pipeline
# =============================================================================

async def run_pipeline_example() -> None:
    """Run a research → draft → review pipeline."""
    print("\n=== Example 2: Multi-Agent Pipeline ===")

    agents = [
        AgentConfig(
            agent_id="researcher",
            name="Researcher",
            description="Gathers information on the topic",
        ),
        AgentConfig(
            agent_id="writer",
            name="Writer",
            description="Drafts a report based on research",
        ),
        AgentConfig(
            agent_id="reviewer",
            name="Reviewer",
            description="Reviews and improves the draft",
        ),
    ]

    config = pipeline(agents, name="Research Pipeline")

    async with AgentRuntime(db_path=":memory:") as runtime:
        orchestrator = MultiAgentOrchestrator(config, runtime)
        result = await orchestrator.run(
            DesiredState(goal="Write a concise report on async Python patterns")
        )

    print(f"Pipeline status: {result.status}")
    print(f"Agents: {result.completed_agents}/{result.total_agents} completed")
    for agent_id, agent_result in result.agent_results.items():
        status = "converged" if agent_result.converged else "failed"
        print(f"  [{status:10}] {agent_id}")


# =============================================================================
# Example 3: Supervisor Pattern
# =============================================================================

async def run_supervisor_example() -> None:
    """Run a supervisor with two specialists."""
    print("\n=== Example 3: Supervisor Pattern ===")

    supervisor_cfg = AgentConfig(
        agent_id="lead",
        name="Lead Agent",
        description="Plans and synthesizes multi-agent work",
    )
    specialist_1 = AgentConfig(
        agent_id="backend-specialist",
        name="Backend Specialist",
        description="Handles backend/API concerns",
    )
    specialist_2 = AgentConfig(
        agent_id="frontend-specialist",
        name="Frontend Specialist",
        description="Handles UI/UX concerns",
    )

    config = supervisor(
        supervisor_config=supervisor_cfg,
        specialists=[specialist_1, specialist_2],
        name="Full-Stack Team",
    )

    async with AgentRuntime(db_path=":memory:") as runtime:
        orchestrator = MultiAgentOrchestrator(config, runtime)
        result = await orchestrator.run(
            DesiredState(goal="Design a task management application")
        )

    print(f"Supervisor status: {result.status}")
    print(f"Agents: {result.completed_agents}/{result.total_agents} completed")


# =============================================================================
# Example 4: RBAC + Namespace
# =============================================================================

async def run_rbac_namespace_example() -> None:
    """Demonstrate RBAC and namespace isolation."""
    print("\n=== Example 4: RBAC + Namespace Isolation ===")

    # Set up RBAC
    rbac = RBACManager()

    # Create a custom "analyst" role with limited tool access
    rbac.create_role("analyst", "Code Analyst", "Read-only code analysis tools")
    rbac.grant_role_tool("analyst", "read_file")
    rbac.grant_role_tool("analyst", "search_code")

    # Assign roles to agents
    rbac.assign_role("analyst-agent", "analyst")
    rbac.assign_role("admin-agent", ROLE_ADMIN)

    # Check permissions
    can_read = rbac.has_permission("analyst-agent", "read_file")
    can_delete = rbac.has_permission("analyst-agent", "delete_file")
    admin_can_delete = rbac.has_permission("admin-agent", "delete_file")

    print(f"analyst-agent can read_file:   {can_read}")
    print(f"analyst-agent can delete_file: {can_delete}")
    print(f"admin-agent can delete_file:   {admin_can_delete}")

    # Set up Namespaces
    ns_manager = NamespaceManager()
    prod_ns = ns_manager.create("production", "Production workloads", max_agents=10)
    staging_ns = ns_manager.create("staging", "Staging workloads", max_agents=5)

    prod_ns.add_agent("analyst-agent")
    staging_ns.add_agent("test-agent")

    print(f"\nProduction namespace prefix: {prod_ns.config.state_prefix}")
    print(f"  Key 'task-1' -> '{prod_ns.make_key('task-1')}'")
    print(f"  Agents: {prod_ns.list_agents()}")

    # Verify isolation: prod keys are separate from staging keys
    prod_key = prod_ns.make_key("shared-state")
    staging_key = staging_ns.make_key("shared-state")
    print(f"\nProd key:    {prod_key}")
    print(f"Staging key: {staging_key}")
    print(f"Keys are isolated: {prod_key != staging_key}")


# =============================================================================
# Example 5: Memory Components
# =============================================================================

async def run_memory_example() -> None:
    """Demonstrate working memory, context window, and episodic memory."""
    print("\n=== Example 5: Memory Components ===")

    # Working memory
    wm = WorkingMemory(max_entries=100)
    wm.set("current_plan", {"steps": ["analyze", "implement", "test"]}, tags=["plan"])
    wm.set("temp_result", {"lines": 1234}, ttl_seconds=60.0, tags=["metrics"])

    print("Working memory snapshot:", wm.snapshot())
    print("Plan entries:", wm.get_by_tag("plan"))

    # Context window management
    ctx = ContextWindowManager(max_tokens=1024, reserve_tokens=256)
    ctx.add_system("You are a helpful code analysis agent.")
    ctx.add_user("Analyze the authentication module.", step=1)
    ctx.add_assistant("I'll look at auth.py for security issues.", step=1)
    ctx.add_tool_result("Found: token expiry not validated.", step=1)

    stats = ctx.get_stats()
    print(f"\nContext window: {stats['used_tokens']}/{stats['budget']} tokens used")
    print(f"Messages in window: {stats['message_count']}")

    # Episodic memory
    ep_memory = EpisodicMemory(":memory:")
    await ep_memory.initialize()

    ep_id = await ep_memory.store(
        agent_id="analyst",
        task_summary="Security audit of authentication module",
        outcome="Found 2 vulnerabilities: expired token reuse, missing CSRF protection",
        context_tags=["security", "auth", "audit"],
    )
    print(f"\nStored episode: {ep_id}")

    await ep_memory.store(
        agent_id="analyst",
        task_summary="Code review for payment processing module",
        outcome="Found SQL injection risk in query builder",
        context_tags=["security", "payment", "review"],
    )

    results = await ep_memory.search("security", limit=5)
    print(f"Search 'security' returned {len(results)} episode(s):")
    for ep in results:
        print(f"  - {ep.task_summary}: {ep.outcome[:60]}...")

    await ep_memory.close()


# =============================================================================
# Main
# =============================================================================

async def main() -> None:
    """Run all examples."""
    await run_workflow_example()
    await run_pipeline_example()
    await run_supervisor_example()
    await run_rbac_namespace_example()
    await run_memory_example()
    print("\nAll Phase 2 examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
