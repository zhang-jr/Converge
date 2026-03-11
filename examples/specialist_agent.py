"""SpecialistAgent example — Operator pattern for Phase 3.

Demonstrates:
1. SpecialistAgent subclass with domain system_prompt + specialist tools
2. CompositeQualityProbe ([LoopDetectorProbe, ConfidenceProbe, DefaultQualityProbe])
3. OTelTracer in ConsoleExporter mode
4. AuditLog integration
5. MetricsCollector integration
6. CallbackHumanInterventionHandler wired into ReconcileLoop + WorkflowController
7. AgentScheduler for concurrent specialist dispatch
8. Namespace isolation + RBAC

Run with:
    python examples/specialist_agent.py
"""

from __future__ import annotations

import asyncio
import sys
import os

# Make imports work from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.agent.agent import Agent, AgentReconcileLoop
from core.runtime.agent_runtime import AgentRuntime
from core.runtime.human_intervention import CallbackHumanInterventionHandler
from core.runtime.scheduler import AgentScheduler, ScheduledTask
from core.state.models import AgentConfig, DesiredState, HumanDecision, LoopContext, StepOutput
from core.state.sqlite_store import SQLiteStateStore
from namespace.namespace import Namespace, NamespaceConfig, NamespaceManager
from observability.audit_log import AuditEntry, AuditLog
from observability.metrics import MetricsCollector
from observability.tracer import OTelTracer
from probes.confidence_probe import ConfidenceProbe
from probes.loop_detector import LoopDetectorProbe
from probes.quality_probe import CompositeQualityProbe, DefaultQualityProbe
from tools.base import ReadOnlyTool, ToolResult
from tools.rbac import RBACManager
from tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Specialist Tools
# ---------------------------------------------------------------------------


class DataAnalysisTool(ReadOnlyTool):
    """Simulated data analysis tool for the SpecialistAgent."""

    @property
    def name(self) -> str:
        return "data_analysis"

    @property
    def description(self) -> str:
        return "Analyse a dataset and return summary statistics"

    async def execute(self, params: dict) -> ToolResult:
        dataset = params.get("dataset", "sales")
        return ToolResult(
            success=True,
            output={
                "dataset": dataset,
                "rows": 1000,
                "mean": 42.5,
                "std": 12.3,
                "status": "analysis completed successfully",
            },
        )


class ReportGeneratorTool(ReadOnlyTool):
    """Simulated report generation tool."""

    @property
    def name(self) -> str:
        return "generate_report"

    @property
    def description(self) -> str:
        return "Generate a formatted report from analysis results"

    async def execute(self, params: dict) -> ToolResult:
        title = params.get("title", "Analysis Report")
        return ToolResult(
            success=True,
            output={
                "title": title,
                "pages": 3,
                "status": "report generation done and finished",
            },
        )


# ---------------------------------------------------------------------------
# SpecialistAgent — Operator pattern
# ---------------------------------------------------------------------------


class DataAnalystAgent(Agent):
    """Domain specialist agent for data analytics.

    Inherits from Agent and overrides the system_prompt with domain-specific
    instructions. Tools are scoped via ToolRegistry + RBAC.
    """

    SPECIALIST_SYSTEM_PROMPT = """You are an expert data analyst agent.
Your role is to:
1. Analyse datasets using the data_analysis tool
2. Generate reports using the generate_report tool
3. Provide clear, quantitative insights

Always reason step-by-step and use available tools before concluding.
When analysis is complete, state 'Task completed successfully'.
"""

    async def think_and_act(self, diff: dict, context: LoopContext) -> StepOutput:
        """Override to inject specialist system prompt."""
        # Inject specialist prompt into config temporarily
        original_prompt = self._config.system_prompt
        self._config.system_prompt = self.SPECIALIST_SYSTEM_PROMPT
        try:
            result = await super().think_and_act(diff, context)
        finally:
            self._config.system_prompt = original_prompt
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_composite_probe() -> CompositeQualityProbe:
    """Build the composite quality probe for Phase 3."""
    return CompositeQualityProbe(
        probes=[
            LoopDetectorProbe(window_size=8, repeat_threshold=3),
            ConfidenceProbe(hard_threshold=0.2, soft_threshold=0.5),
            DefaultQualityProbe(),
        ],
        strategy="all_pass",
    )


async def build_audit_log() -> AuditLog:
    al = AuditLog(db_path=":memory:")
    return al


async def auto_approve_handler(
    reason: str,
    context: LoopContext,
    pending_action: dict | None,
) -> HumanDecision:
    """Auto-approve for the example; in production show a UI dialog."""
    print(f"  [HumanGate] Approval requested: {reason[:80]}")
    return HumanDecision(approved=True, feedback="auto-approved in example")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 60)
    print("Phase 3 SpecialistAgent Example")
    print("=" * 60)

    # --- Observability setup ---
    tracer = OTelTracer(agent_id="data-analyst", service_name="specialist-example")
    audit_log = await build_audit_log()
    metrics = MetricsCollector()

    # --- Namespace + RBAC ---
    ns_config = NamespaceConfig(
        name="analytics",
        description="Analytics namespace",
        max_agents=10,
    )
    ns_manager = NamespaceManager()
    await ns_manager.create_namespace(ns_config)

    rbac = RBACManager()
    rbac.assign_role("analyst-agent", "operator")

    # --- Tool registry ---
    registry = ToolRegistry(rbac_manager=rbac)
    registry.register(DataAnalysisTool())
    registry.register(ReportGeneratorTool())
    registry.grant(
        agent_id="analyst-agent",
        tool_name="data_analysis",
        can_execute=True,
    )
    registry.grant(
        agent_id="analyst-agent",
        tool_name="generate_report",
        can_execute=True,
    )

    # --- Human intervention handler ---
    intervention_handler = CallbackHumanInterventionHandler(auto_approve_handler)

    # --- StateStore ---
    state_store = SQLiteStateStore(":memory:")

    # --- AgentRuntime with Phase 3 integrations ---
    runtime = AgentRuntime(
        state_store=state_store,
        tool_registry=registry,
        quality_probe=build_composite_probe(),
        metrics_collector=metrics,
        human_intervention_handler=intervention_handler,
    )
    await runtime.initialize()

    # --- Agent config ---
    config = AgentConfig(
        agent_id="analyst-agent",
        name="Data Analyst",
        description="Specialist data analytics agent",
        safety_max_steps=5,
        confidence_threshold=0.4,
    )

    # --- Scenario 1: Single specialist run ---
    print("\n[Scenario 1] Single specialist run")
    await audit_log.log(AuditEntry(
        actor="analyst-agent",
        action="agent_start",
        trace_id="trace-001",
        details={"goal": "Analyse Q4 sales data"},
    ))

    result = await runtime.run(
        goal="Analyse Q4 sales data and generate executive summary report",
        agent_config=config,
    )

    print(f"  Status: {result.status}")
    print(f"  Steps:  {result.total_steps}")
    print(f"  Trace:  {result.trace_id}")

    await audit_log.log(AuditEntry(
        actor="analyst-agent",
        action="agent_end",
        trace_id=result.trace_id,
        outcome="success" if result.converged else "failure",
        details={"status": result.status},
    ))

    # --- Scenario 2: Concurrent tasks via AgentScheduler ---
    print("\n[Scenario 2] AgentScheduler — 3 concurrent tasks")

    tasks = [
        ScheduledTask(
            goal=f"Analyse dataset region_{i} and report",
            agent_config=AgentConfig(agent_id=f"analyst-{i}", safety_max_steps=3),
            priority=i,
        )
        for i in range(3)
    ]

    async with AgentScheduler(runtime, max_concurrent=2) as scheduler:
        for t in tasks:
            await scheduler.submit(t)
        results = await scheduler.run_until_empty()

    print(f"  Completed {len(results)} tasks")
    for i, r in enumerate(results):
        print(f"  Task {i}: status={r.status}")

    # --- Scenario 3: Metrics summary ---
    print("\n[Scenario 3] Metrics Summary")
    summary = metrics.get_summary()
    print(f"  Total runs:    {summary['total_runs']}")
    print(f"  Successful:    {summary['successful_runs']}")
    print(f"  Avg latency:   {summary['avg_latency_ms']:.1f}ms")
    print(f"  Total steps:   {summary['total_steps']}")

    # --- Scenario 4: Audit log query ---
    print("\n[Scenario 4] Audit Log")
    entries = await audit_log.query(actor="analyst-agent")
    print(f"  Audit entries for analyst-agent: {len(entries)}")
    for entry in entries:
        print(f"  [{entry.action}] outcome={entry.outcome}")

    # --- Prometheus export ---
    print("\n[Scenario 5] Prometheus Export (snippet)")
    prom = metrics.to_prometheus()
    lines = [l for l in prom.splitlines() if not l.startswith("#")][:4]
    for line in lines:
        print(f"  {line}")

    await audit_log.close()
    await runtime.shutdown()

    print("\n" + "=" * 60)
    print("SpecialistAgent example completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
