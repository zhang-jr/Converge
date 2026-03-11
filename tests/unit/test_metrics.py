"""Unit tests for MetricsCollector."""

from __future__ import annotations

import pytest

from core.state.models import ReconcileResult, StepOutput, ToolCall
from observability.metrics import MetricEntry, MetricsCollector, get_global_metrics, set_global_metrics


def _result(
    status: str = "converged",
    steps: list[StepOutput] | None = None,
    duration_ms: float = 100.0,
) -> ReconcileResult:
    return ReconcileResult(
        status=status,  # type: ignore[arg-type]
        steps=steps or [],
        total_steps=len(steps) if steps else 0,
        converged=(status == "converged"),
        duration_ms=duration_ms,
    )


class TestMetricsCollector:
    @pytest.fixture()
    def collector(self) -> MetricsCollector:
        return MetricsCollector()

    # ------------------------------------------------------------------
    # record / get_summary
    # ------------------------------------------------------------------

    def test_record_raw_entry(self, collector: MetricsCollector) -> None:
        entry = MetricEntry(name="my_counter", value=1.0, labels={"env": "test"})
        collector.record(entry)
        assert len(collector._entries) == 1

    def test_record_run_basic(self, collector: MetricsCollector) -> None:
        result = _result(duration_ms=250.0)
        metrics = collector.record_run(result, agent_id="agent-1")
        assert metrics.agent_id == "agent-1"
        assert metrics.latency_ms == 250.0
        assert metrics.success is True
        assert metrics.step_count == 0

    def test_record_run_counts_tool_calls(self, collector: MetricsCollector) -> None:
        steps = [
            StepOutput(
                step_number=1,
                action="a",
                tool_calls=[
                    ToolCall(tool_name="t1", success=True),
                    ToolCall(tool_name="t2", success=False, error="e"),
                ],
            )
        ]
        result = _result(steps=steps)
        metrics = collector.record_run(result, agent_id="a")
        assert metrics.tool_calls == 2
        assert metrics.error_count == 1

    def test_get_summary_empty(self, collector: MetricsCollector) -> None:
        summary = collector.get_summary()
        assert summary["total_runs"] == 0

    def test_get_summary_multiple_runs(self, collector: MetricsCollector) -> None:
        collector.record_run(_result("converged", duration_ms=100), "a")
        collector.record_run(_result("failed", duration_ms=200), "a")
        summary = collector.get_summary(agent_id="a")
        assert summary["total_runs"] == 2
        assert summary["successful_runs"] == 1
        assert summary["failed_runs"] == 1
        assert summary["avg_latency_ms"] == 150.0

    def test_get_summary_filters_by_agent(self, collector: MetricsCollector) -> None:
        collector.record_run(_result(), "agent-1")
        collector.record_run(_result(), "agent-2")
        s1 = collector.get_summary("agent-1")
        s2 = collector.get_summary("agent-2")
        assert s1["total_runs"] == 1
        assert s2["total_runs"] == 1

    # ------------------------------------------------------------------
    # to_prometheus
    # ------------------------------------------------------------------

    def test_to_prometheus_format(self, collector: MetricsCollector) -> None:
        collector.record_run(_result(), "my-agent")
        prom = collector.to_prometheus()
        assert "# HELP" in prom
        assert "# TYPE" in prom
        assert "agent_runs_total" in prom
        assert "agent_latency_ms" in prom

    def test_to_prometheus_labels(self, collector: MetricsCollector) -> None:
        collector.record_run(_result(status="converged"), "agent-x")
        prom = collector.to_prometheus()
        assert 'agent_id="agent-x"' in prom
        assert 'status="converged"' in prom

    def test_to_prometheus_empty(self, collector: MetricsCollector) -> None:
        assert collector.to_prometheus() == ""

    # ------------------------------------------------------------------
    # clear
    # ------------------------------------------------------------------

    def test_clear(self, collector: MetricsCollector) -> None:
        collector.record_run(_result(), "a")
        collector.clear()
        assert collector.get_summary()["total_runs"] == 0
        assert collector.to_prometheus() == ""

    # ------------------------------------------------------------------
    # Global singleton
    # ------------------------------------------------------------------

    def test_global_metrics_singleton(self) -> None:
        m1 = get_global_metrics()
        m2 = get_global_metrics()
        assert m1 is m2

    def test_set_global_metrics(self) -> None:
        custom = MetricsCollector()
        set_global_metrics(custom)
        assert get_global_metrics() is custom
        # Reset to fresh instance after test
        set_global_metrics(MetricsCollector())
