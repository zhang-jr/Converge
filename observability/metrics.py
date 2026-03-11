"""Metrics collection for the Agent Framework.

Provides a lightweight, in-memory metrics system compatible with Prometheus
exposition format. Metrics are gathered per reconcile run and can be queried
or exported at any time.

Integration point: AgentRuntime.run() calls GlobalMetrics.record_run() after
each reconcile finishes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from core.state.models import ReconcileResult


class MetricEntry(BaseModel):
    """A single metric data point.

    Attributes:
        name: Metric name (snake_case, e.g. ``agent_steps_total``).
        value: Numeric value.
        labels: Key-value label pairs for dimensionality.
        timestamp: When the metric was recorded.
        metric_type: Prometheus metric type.
    """

    name: str
    value: float
    labels: dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metric_type: Literal["counter", "gauge", "histogram"] = "counter"


class AgentMetrics(BaseModel):
    """Aggregated metrics for a single reconcile run.

    Attributes:
        agent_id: ID of the agent.
        trace_id: Trace ID from the reconcile run.
        token_usage: Estimated or measured token usage.
        latency_ms: Total wall-clock time for the run.
        step_count: Number of reconcile steps executed.
        tool_calls: Total number of tool invocations.
        error_count: Number of failed tool calls or hard-fail probes.
        success: Whether the run converged successfully.
        recorded_at: When these metrics were captured.
    """

    agent_id: str = ""
    trace_id: str = ""
    token_usage: int = 0
    latency_ms: float = 0.0
    step_count: int = 0
    tool_calls: int = 0
    error_count: int = 0
    success: bool = False
    recorded_at: datetime = Field(default_factory=datetime.utcnow)


class MetricsCollector:
    """In-memory metrics collector with Prometheus export support.

    All metrics are stored in RAM. For production use, flush periodically
    to a time-series backend (Victoria Metrics, Prometheus push gateway, etc.).

    Usage::

        collector = MetricsCollector()
        collector.record_run(result, agent_id="my-agent")
        print(collector.to_prometheus())
    """

    def __init__(self) -> None:
        self._entries: list[MetricEntry] = []
        self._run_metrics: list[AgentMetrics] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, entry: MetricEntry) -> None:
        """Append a raw metric entry.

        Args:
            entry: The metric data point to store.
        """
        self._entries.append(entry)

    def record_run(
        self,
        result: ReconcileResult,
        agent_id: str = "",
        token_usage: int = 0,
    ) -> AgentMetrics:
        """Extract and store metrics from a completed ReconcileResult.

        Creates both an AgentMetrics summary and individual MetricEntry records
        for counter-style tracking.

        Args:
            result: The completed reconcile result.
            agent_id: ID of the agent that ran.
            token_usage: Token count if tracked externally.

        Returns:
            The AgentMetrics summary that was stored.
        """
        tool_calls = sum(len(s.tool_calls) for s in result.steps)
        error_count = sum(
            1 for s in result.steps for tc in s.tool_calls if not tc.success
        )

        metrics = AgentMetrics(
            agent_id=agent_id,
            trace_id=result.trace_id,
            token_usage=token_usage,
            latency_ms=result.duration_ms,
            step_count=result.total_steps,
            tool_calls=tool_calls,
            error_count=error_count,
            success=result.converged,
        )
        self._run_metrics.append(metrics)

        labels: dict[str, str] = {"agent_id": agent_id, "status": result.status}

        self.record(MetricEntry(name="agent_runs_total", value=1.0, labels=labels))
        self.record(
            MetricEntry(
                name="agent_steps_total",
                value=float(result.total_steps),
                labels=labels,
            )
        )
        self.record(
            MetricEntry(
                name="agent_latency_ms",
                value=result.duration_ms,
                labels=labels,
                metric_type="histogram",
            )
        )
        self.record(
            MetricEntry(
                name="agent_tool_calls_total",
                value=float(tool_calls),
                labels=labels,
            )
        )
        self.record(
            MetricEntry(
                name="agent_errors_total",
                value=float(error_count),
                labels=labels,
            )
        )

        return metrics

    def get_summary(self, agent_id: str | None = None) -> dict[str, Any]:
        """Return aggregated summary statistics.

        Args:
            agent_id: Filter to a specific agent. If None, all agents.

        Returns:
            Dict with counts, totals, and averages.
        """
        runs = self._run_metrics
        if agent_id is not None:
            runs = [r for r in runs if r.agent_id == agent_id]

        if not runs:
            return {"total_runs": 0}

        total_runs = len(runs)
        successful = sum(1 for r in runs if r.success)
        total_steps = sum(r.step_count for r in runs)
        total_tool_calls = sum(r.tool_calls for r in runs)
        total_errors = sum(r.error_count for r in runs)
        avg_latency = sum(r.latency_ms for r in runs) / total_runs
        avg_steps = total_steps / total_runs

        return {
            "total_runs": total_runs,
            "successful_runs": successful,
            "failed_runs": total_runs - successful,
            "success_rate": successful / total_runs,
            "total_steps": total_steps,
            "avg_steps_per_run": avg_steps,
            "total_tool_calls": total_tool_calls,
            "total_errors": total_errors,
            "avg_latency_ms": avg_latency,
        }

    def to_prometheus(self) -> str:
        """Export all metrics in Prometheus exposition format.

        Returns:
            Multi-line string compatible with Prometheus text format.
        """
        # Group entries by metric name to emit HELP/TYPE headers once
        by_name: dict[str, list[MetricEntry]] = {}
        for entry in self._entries:
            by_name.setdefault(entry.name, []).append(entry)

        lines: list[str] = []
        for name, entries in by_name.items():
            metric_type = entries[0].metric_type
            lines.append(f"# HELP {name} Agent Framework metric: {name}")
            lines.append(f"# TYPE {name} {metric_type}")
            for entry in entries:
                label_str = ""
                if entry.labels:
                    kv = ",".join(
                        f'{k}="{v}"' for k, v in sorted(entry.labels.items())
                    )
                    label_str = "{" + kv + "}"
                lines.append(f"{name}{label_str} {entry.value}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all stored metrics."""
        self._entries.clear()
        self._run_metrics.clear()


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_global_metrics: MetricsCollector | None = None


def get_global_metrics() -> MetricsCollector:
    """Get the global MetricsCollector singleton.

    Returns:
        The global MetricsCollector instance.
    """
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


def set_global_metrics(collector: MetricsCollector) -> None:
    """Replace the global MetricsCollector (useful in tests).

    Args:
        collector: The MetricsCollector to use globally.
    """
    global _global_metrics
    _global_metrics = collector
