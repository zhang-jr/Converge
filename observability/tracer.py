"""Structured tracing for observability.

Every ReconcileLoop step produces structured trace data including:
- trace_id: Unique identifier for the reconcile run
- step info: Step number, action, reasoning
- tool I/O: Tool calls with parameters and results
- probe results: Quality probe evaluations
- state changes: Keys modified in StateStore

Phase 3 will integrate with OpenTelemetry for distributed tracing.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from core.state.models import StepOutput
    from probes.quality_probe import ProbeResult


class TraceEvent(BaseModel):
    """A single trace event.

    Attributes:
        trace_id: Unique identifier for the reconcile run.
        span_id: Unique identifier for this event.
        parent_span_id: Parent span if nested.
        event_type: Type of event (step, tool_call, probe, etc.).
        timestamp: When the event occurred.
        agent_id: ID of the agent.
        data: Event-specific data.
    """

    trace_id: str
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_span_id: str | None = None
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_id: str = ""
    data: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(
            {
                "trace_id": self.trace_id,
                "span_id": self.span_id,
                "parent_span_id": self.parent_span_id,
                "event_type": self.event_type,
                "timestamp": self.timestamp.isoformat(),
                "agent_id": self.agent_id,
                "data": self.data,
            },
            default=str,
        )


class Tracer:
    """Structured tracer for observability.

    Provides structured logging of all reconcile loop activity.
    In Phase 3, this will integrate with OpenTelemetry.

    Usage:
        tracer = Tracer(agent_id="agent-1")
        trace_id = tracer.start_trace()
        tracer.log_step(step_output)
        tracer.log_tool_call(tool_name, params, result)
        tracer.log_probe_result(probe_result)
        tracer.end_trace()
    """

    def __init__(
        self,
        agent_id: str = "",
        logger: logging.Logger | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        """Initialize the tracer.

        Args:
            agent_id: ID of the agent being traced.
            logger: Custom logger. If None, creates a new one.
            log_level: Logging level for trace events.
        """
        self._agent_id = agent_id
        self._trace_id: str | None = None
        self._events: list[TraceEvent] = []
        self._current_span_id: str | None = None
        self._log_level = log_level

        if logger is None:
            self._logger = logging.getLogger(f"agent_framework.tracer.{agent_id}")
            self._logger.setLevel(log_level)
            if not self._logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                handler.setFormatter(
                    logging.Formatter("%(message)s")
                )
                self._logger.addHandler(handler)
        else:
            self._logger = logger

    @property
    def trace_id(self) -> str | None:
        """Current trace ID."""
        return self._trace_id

    @property
    def events(self) -> list[TraceEvent]:
        """All trace events for the current trace."""
        return self._events.copy()

    def start_trace(self, trace_id: str | None = None) -> str:
        """Start a new trace.

        Args:
            trace_id: Optional trace ID. If None, generates a new one.

        Returns:
            The trace ID.
        """
        self._trace_id = trace_id or str(uuid.uuid4())
        self._events = []
        self._current_span_id = None

        event = TraceEvent(
            trace_id=self._trace_id,
            event_type="trace_start",
            agent_id=self._agent_id,
            data={"start_time": datetime.utcnow().isoformat()},
        )
        self._log_event(event)
        return self._trace_id

    def end_trace(self, status: str = "completed") -> None:
        """End the current trace.

        Args:
            status: Final status of the trace.
        """
        if self._trace_id is None:
            return

        event = TraceEvent(
            trace_id=self._trace_id,
            event_type="trace_end",
            agent_id=self._agent_id,
            data={
                "status": status,
                "end_time": datetime.utcnow().isoformat(),
                "total_events": len(self._events),
            },
        )
        self._log_event(event)
        self._trace_id = None

    def log_step(
        self,
        step_output: StepOutput,
        state_diff: dict[str, Any] | None = None,
    ) -> str:
        """Log a reconcile loop step.

        Args:
            step_output: The step output to log.
            state_diff: Changes made to the state store.

        Returns:
            The span ID for this step.
        """
        if self._trace_id is None:
            self.start_trace()

        event = TraceEvent(
            trace_id=self._trace_id,  # type: ignore
            event_type="step",
            agent_id=self._agent_id,
            data={
                "step_number": step_output.step_number,
                "action": step_output.action,
                "reasoning": step_output.reasoning,
                "result": str(step_output.result) if step_output.result else None,
                "tool_calls_count": len(step_output.tool_calls),
                "state_changes": step_output.state_changes,
                "state_diff": state_diff,
            },
        )
        self._current_span_id = event.span_id
        self._log_event(event)
        return event.span_id

    def log_tool_call(
        self,
        tool_name: str,
        params: dict[str, Any],
        result: Any,
        success: bool = True,
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> str:
        """Log a tool invocation.

        Args:
            tool_name: Name of the tool called.
            params: Parameters passed to the tool.
            result: Result from the tool.
            success: Whether the call succeeded.
            error: Error message if failed.
            duration_ms: Execution time in milliseconds.

        Returns:
            The span ID for this tool call.
        """
        if self._trace_id is None:
            self.start_trace()

        event = TraceEvent(
            trace_id=self._trace_id,  # type: ignore
            parent_span_id=self._current_span_id,
            event_type="tool_call",
            agent_id=self._agent_id,
            data={
                "tool_name": tool_name,
                "params": params,
                "result": str(result) if result else None,
                "success": success,
                "error": error,
                "duration_ms": duration_ms,
            },
        )
        self._log_event(event)
        return event.span_id

    def log_probe_result(self, probe_result: ProbeResult, probe_name: str = "") -> str:
        """Log a quality probe evaluation.

        Args:
            probe_result: The probe result to log.
            probe_name: Name of the probe.

        Returns:
            The span ID for this probe result.
        """
        if self._trace_id is None:
            self.start_trace()

        event = TraceEvent(
            trace_id=self._trace_id,  # type: ignore
            parent_span_id=self._current_span_id,
            event_type="probe_result",
            agent_id=self._agent_id,
            data={
                "probe_name": probe_name,
                "verdict": probe_result.verdict,
                "confidence": probe_result.confidence,
                "reason": probe_result.reason,
                "should_converge": probe_result.should_converge,
                "suggestions": probe_result.suggestions,
            },
        )
        self._log_event(event)
        return event.span_id

    def log_state_change(
        self,
        key: str,
        old_value: Any,
        new_value: Any,
        change_type: str,
    ) -> str:
        """Log a state store change.

        Args:
            key: The state key that changed.
            old_value: Previous value.
            new_value: New value.
            change_type: Type of change (created/updated/deleted).

        Returns:
            The span ID for this state change.
        """
        if self._trace_id is None:
            self.start_trace()

        event = TraceEvent(
            trace_id=self._trace_id,  # type: ignore
            parent_span_id=self._current_span_id,
            event_type="state_change",
            agent_id=self._agent_id,
            data={
                "key": key,
                "old_value": str(old_value) if old_value else None,
                "new_value": str(new_value) if new_value else None,
                "change_type": change_type,
            },
        )
        self._log_event(event)
        return event.span_id

    def log_error(self, error: Exception, context: dict[str, Any] | None = None) -> str:
        """Log an error.

        Args:
            error: The exception that occurred.
            context: Additional context.

        Returns:
            The span ID for this error.
        """
        if self._trace_id is None:
            self.start_trace()

        event = TraceEvent(
            trace_id=self._trace_id,  # type: ignore
            parent_span_id=self._current_span_id,
            event_type="error",
            agent_id=self._agent_id,
            data={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
            },
        )
        self._log_event(event)
        return event.span_id

    def log_human_intervention(
        self,
        reason: str,
        pending_action: str | None = None,
        decision: dict[str, Any] | None = None,
    ) -> str:
        """Log a human intervention request or decision.

        Args:
            reason: Why intervention was needed.
            pending_action: The action awaiting approval.
            decision: The human's decision if made.

        Returns:
            The span ID for this event.
        """
        if self._trace_id is None:
            self.start_trace()

        event = TraceEvent(
            trace_id=self._trace_id,  # type: ignore
            parent_span_id=self._current_span_id,
            event_type="human_intervention",
            agent_id=self._agent_id,
            data={
                "reason": reason,
                "pending_action": pending_action,
                "decision": decision,
            },
        )
        self._log_event(event)
        return event.span_id

    def _log_event(self, event: TraceEvent) -> None:
        """Internal method to log an event."""
        self._events.append(event)
        self._logger.log(self._log_level, event.to_json())

    def get_trace_summary(self) -> dict[str, Any]:
        """Get a summary of the current trace.

        Returns:
            Dictionary with trace summary statistics.
        """
        if not self._events:
            return {}

        steps = [e for e in self._events if e.event_type == "step"]
        tool_calls = [e for e in self._events if e.event_type == "tool_call"]
        errors = [e for e in self._events if e.event_type == "error"]
        probes = [e for e in self._events if e.event_type == "probe_result"]

        return {
            "trace_id": self._trace_id,
            "agent_id": self._agent_id,
            "total_events": len(self._events),
            "steps": len(steps),
            "tool_calls": len(tool_calls),
            "errors": len(errors),
            "probe_evaluations": len(probes),
        }


# Global tracer instance for convenience
_global_tracer: Tracer | None = None


def get_global_tracer() -> Tracer:
    """Get the global tracer instance.

    Returns:
        The global Tracer instance.
    """
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer()
    return _global_tracer


def set_global_tracer(tracer: Tracer) -> None:
    """Set the global tracer instance.

    Args:
        tracer: The Tracer instance to use globally.
    """
    global _global_tracer
    _global_tracer = tracer


# ---------------------------------------------------------------------------
# OpenTelemetry integration (optional — requires [otel] extras)
# ---------------------------------------------------------------------------


class OTelTracer(Tracer):
    """Tracer subclass that emits OpenTelemetry spans.

    Falls back gracefully to JSON-only logging when the ``opentelemetry-sdk``
    package is not installed, making the ``[otel]`` extras truly optional.

    The base class :class:`Tracer` behaviour is always preserved: all events
    are written to the structured JSON logger regardless of OTel availability.

    Args:
        agent_id: ID of the agent being traced.
        service_name: OTel service name (default ``"agent-framework"``).
        otlp_endpoint: gRPC endpoint for OTLP exporter, e.g.
            ``"http://localhost:4317"``. If ``None``, uses ConsoleSpanExporter.
        logger: Optional custom logger (forwarded to :class:`Tracer`).
        log_level: Logging level (forwarded to :class:`Tracer`).
    """

    def __init__(
        self,
        agent_id: str = "",
        service_name: str = "agent-framework",
        otlp_endpoint: str | None = None,
        logger: logging.Logger | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        super().__init__(agent_id=agent_id, logger=logger, log_level=log_level)
        self._otel_enabled = False
        self._otel_span: Any = None
        self._otel_tracer: Any = None

        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider

            resource = Resource.create({"service.name": service_name, "agent.id": agent_id})
            provider = TracerProvider(resource=resource)

            if otlp_endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                        OTLPSpanExporter,
                    )
                    from opentelemetry.sdk.trace.export import BatchSpanProcessor

                    exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                    provider.add_span_processor(BatchSpanProcessor(exporter))
                except ImportError:
                    self._logger.warning(
                        "opentelemetry-exporter-otlp not installed; "
                        "falling back to ConsoleSpanExporter"
                    )
                    self._add_console_exporter(provider)
            else:
                self._add_console_exporter(provider)

            otel_trace.set_tracer_provider(provider)
            self._otel_tracer = otel_trace.get_tracer(service_name)
            self._otel_enabled = True

        except ImportError:
            self._logger.debug(
                "opentelemetry-sdk not installed; OTel spans disabled. "
                "Install with: pip install agent-framework[otel]"
            )

    @staticmethod
    def _add_console_exporter(provider: Any) -> None:
        try:
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )

            provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
        except ImportError:
            # ConsoleSpanExporter as ultimate fallback
            try:
                from opentelemetry.sdk.trace.export import (
                    ConsoleSpanExporter,
                    SimpleSpanProcessor,
                )

                provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
            except ImportError:
                pass

    def start_trace(self, trace_id: str | None = None) -> str:
        """Start a trace and optionally open an OTel root span.

        Args:
            trace_id: Optional custom trace ID.

        Returns:
            The trace ID string.
        """
        tid = super().start_trace(trace_id)
        if self._otel_enabled and self._otel_tracer is not None:
            self._otel_span = self._otel_tracer.start_span(
                name=f"reconcile.{self._agent_id}",
                attributes={"trace_id": tid, "agent_id": self._agent_id},
            )
        return tid

    def log_step(
        self,
        step_output: StepOutput,
        state_diff: dict[str, Any] | None = None,
    ) -> str:
        """Log a step as a JSON event and an OTel span event.

        Args:
            step_output: The step output to log.
            state_diff: State changes made in this step.

        Returns:
            Span ID string.
        """
        span_id = super().log_step(step_output, state_diff)
        if self._otel_enabled and self._otel_span is not None:
            self._otel_span.add_event(
                name="step",
                attributes={
                    "step_number": step_output.step_number,
                    "action": step_output.action[:256],
                    "reasoning": (step_output.reasoning or "")[:512],
                    "span_id": span_id,
                },
            )
        return span_id

    def log_tool_call(
        self,
        tool_name: str,
        params: dict[str, Any],
        result: Any,
        success: bool = True,
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> str:
        """Log a tool call as a JSON event and an OTel span event.

        Args:
            tool_name: Name of the tool.
            params: Tool parameters.
            result: Tool result.
            success: Whether the call succeeded.
            error: Error message if failed.
            duration_ms: Execution time in milliseconds.

        Returns:
            Span ID string.
        """
        span_id = super().log_tool_call(tool_name, params, result, success, error, duration_ms)
        if self._otel_enabled and self._otel_span is not None:
            self._otel_span.add_event(
                name="tool_call",
                attributes={
                    "tool_name": tool_name,
                    "success": success,
                    "duration_ms": duration_ms,
                    "error": error or "",
                    "span_id": span_id,
                },
            )
        return span_id

    def end_trace(self, status: str = "completed") -> None:
        """End the trace and close the OTel root span.

        Args:
            status: Final status of the trace.
        """
        super().end_trace(status)
        if self._otel_enabled and self._otel_span is not None:
            self._otel_span.set_attribute("final_status", status)
            self._otel_span.end()
            self._otel_span = None
