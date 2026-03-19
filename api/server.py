"""HTTP API Server for the Agent Framework.

Exposes the framework capabilities over REST + WebSocket so that
any language/client can submit agent runs and stream results.

Endpoints:
    POST /runs                       Submit a new agent run (async)
    GET  /runs                       List all runs
    GET  /runs/{run_id}              Get run status and result
    GET  /runs/{run_id}/steps        Get completed steps for a run
    WS   /runs/{run_id}/stream       Real-time step event stream
    POST /workflows                  Submit a workflow (stub, Phase 8+)
    GET  /health                     Health check

Usage:
    # Start server
    uvicorn api.server:app --reload --port 8000

    # Submit a run
    curl -X POST http://localhost:8000/runs \\
         -H 'Content-Type: application/json' \\
         -d '{"goal": "List all Python files in /tmp"}'

    # Stream run events
    wscat -c ws://localhost:8000/runs/{run_id}/stream

Optional [api] dependencies:
    pip install "agent-framework[api]"
    (fastapi, uvicorn, httpx)
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, Literal

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel as _BaseModel
    _HAS_FASTAPI = True
except ImportError as _fastapi_err:
    raise ImportError(
        "FastAPI is required for the API server. "
        "Install with: pip install 'agent-framework[api]'"
    ) from _fastapi_err


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RunRequest(_BaseModel):
    """Payload for POST /runs.

    Attributes:
        goal: Natural language goal for the agent.
        constraints: Optional list of constraint strings.
        context: Optional key/value context passed to the agent.
        agent_id: Agent identifier used in tracing (default: 'api-agent').
        model: LLM model override.  If omitted uses framework default.
        safety_max_steps: Maximum reconcile loop iterations (default 50).
        enable_rollback: Whether to enable state rollback on failure.
    """

    goal: str
    constraints: list[str] = []
    context: dict[str, Any] = {}
    agent_id: str = "api-agent"
    model: str | None = None
    safety_max_steps: int = 50
    enable_rollback: bool = False


class WorkflowRequest(_BaseModel):
    """Payload for POST /workflows (stub)."""

    name: str
    steps: list[dict[str, Any]] = []
    context: dict[str, Any] = {}


RunStatus = Literal[
    "pending", "running", "converged", "failed", "timeout", "human_intervention"
]


class RunRecord(_BaseModel):
    """Internal record tracking a single agent run."""

    model_config = {"arbitrary_types_allowed": True}

    run_id: str
    status: RunStatus
    goal: str
    agent_id: str
    constraints: list[str] = []
    steps: list[dict[str, Any]] = []
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    result: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Application state (in-process; replace with persistent store for prod)
# ---------------------------------------------------------------------------

_runs: dict[str, RunRecord] = {}
# Each run gets a queue for streaming step events to WebSocket clients
_run_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agent Framework API",
    description=(
        "REST + WebSocket API for the K8s-inspired Agent orchestration framework."
    ),
    version="0.7.0",
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Return service liveness status."""
    return {"status": "ok", "service": "agent-framework"}


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

@app.post("/runs", status_code=202, tags=["runs"])
async def create_run(request: RunRequest) -> dict[str, Any]:
    """Submit a new agent run.

    The run executes asynchronously in the background.  Poll
    ``GET /runs/{run_id}`` for status or connect to
    ``WS /runs/{run_id}/stream`` for real-time step events.

    Returns:
        ``{"run_id": "...", "status": "pending"}``
    """
    run_id = str(uuid.uuid4())

    run = RunRecord(
        run_id=run_id,
        status="pending",
        goal=request.goal,
        agent_id=request.agent_id,
        constraints=request.constraints,
        created_at=datetime.utcnow(),
    )
    _runs[run_id] = run
    _run_queues[run_id] = asyncio.Queue(maxsize=1000)

    asyncio.create_task(_execute_run(run_id, request))

    return {"run_id": run_id, "status": "pending"}


@app.get("/runs", tags=["runs"])
async def list_runs(
    status: RunStatus | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List agent runs, optionally filtered by status.

    Args:
        status: Filter by run status (optional).
        limit: Maximum number of runs to return (default 50).

    Returns:
        ``{"runs": [...], "total": N}``
    """
    runs = list(_runs.values())
    if status:
        runs = [r for r in runs if r.status == status]
    runs = runs[-limit:]  # return most recent
    return {
        "runs": [_run_summary(r) for r in runs],
        "total": len(runs),
    }


@app.get("/runs/{run_id}", tags=["runs"])
async def get_run(run_id: str) -> dict[str, Any]:
    """Get the full status and result of a run.

    Args:
        run_id: Run identifier returned by POST /runs.

    Raises:
        404: If run_id is not found.
    """
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run.model_dump(mode="json")


@app.get("/runs/{run_id}/steps", tags=["runs"])
async def get_run_steps(run_id: str) -> dict[str, Any]:
    """Get all completed steps for a run.

    Args:
        run_id: Run identifier.

    Raises:
        404: If run_id is not found.
    """
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return {
        "run_id": run_id,
        "status": run.status,
        "steps": run.steps,
        "total_steps": len(run.steps),
    }


# ---------------------------------------------------------------------------
# WebSocket streaming
# ---------------------------------------------------------------------------

@app.websocket("/runs/{run_id}/stream")
async def stream_run(websocket: WebSocket, run_id: str) -> None:
    """Stream real-time step events for a run.

    Events are JSON objects with a ``type`` field:
    - ``{"type": "step",  "run_id": "...", "step": {...}}``
    - ``{"type": "done",  "run_id": "...", "status": "converged"}``
    - ``{"type": "error", "run_id": "...", "error": "..."}``
    - ``{"type": "ping"}``  (keepalive every 30 s)

    Args:
        run_id: Run identifier returned by POST /runs.
    """
    await websocket.accept()

    queue = _run_queues.get(run_id)
    if queue is None:
        await websocket.close(code=4004, reason=f"Run '{run_id}' not found")
        return

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(event)
                if event.get("type") in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                # Send keepalive ping
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Workflows (stub)
# ---------------------------------------------------------------------------

@app.post("/workflows", status_code=202, tags=["workflows"])
async def create_workflow(request: WorkflowRequest) -> dict[str, Any]:
    """Submit a multi-step workflow (stub — Phase 8+ implementation).

    Currently returns a placeholder response.

    Args:
        request: Workflow definition.

    Returns:
        ``{"workflow_id": "...", "status": "not_implemented"}``
    """
    return {
        "workflow_id": str(uuid.uuid4()),
        "status": "not_implemented",
        "message": "Workflow execution is planned for Phase 8. Use POST /runs for single-agent tasks.",
    }


# ---------------------------------------------------------------------------
# Background run execution
# ---------------------------------------------------------------------------

async def _execute_run(run_id: str, request: RunRequest) -> None:
    """Execute an agent run in the background and stream step events.

    Updates the RunRecord in _runs and pushes events to _run_queues[run_id].

    Args:
        run_id: The run identifier.
        request: The original RunRequest.
    """
    run = _runs[run_id]
    queue = _run_queues[run_id]

    run.status = "running"
    run.started_at = datetime.utcnow()

    # Build step callback — appends to run.steps and pushes to WebSocket queue
    async def _step_callback(step: Any) -> None:
        step_dict = step.model_dump(mode="json")
        run.steps.append(step_dict)
        event: dict[str, Any] = {
            "type": "step",
            "run_id": run_id,
            "step_number": step_dict.get("step_number"),
            "step": step_dict,
        }
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Slow consumer — drop the event rather than blocking

    from core.agent.agent import Agent
    from core.state.models import AgentConfig, DesiredState
    from core.state.sqlite_store import SQLiteStateStore

    state_store = SQLiteStateStore()
    try:
        config = AgentConfig(
            agent_id=request.agent_id,
            model=request.model,
            safety_max_steps=request.safety_max_steps,
        )

        desired_state = DesiredState(
            goal=request.goal,
            constraints=request.constraints,
            context=request.context,
        )

        agent = Agent(
            config=config,
            state_store=state_store,
        )

        result = await agent.run(
            desired_state,
            step_callback=_step_callback,
            enable_rollback=request.enable_rollback,
        )

        run.status = result.status  # type: ignore[assignment]
        run.result = result.model_dump(mode="json")
        run.completed_at = datetime.utcnow()
        if result.error:
            run.error = result.error

        done_event: dict[str, Any] = {
            "type": "done",
            "run_id": run_id,
            "status": run.status,
            "total_steps": result.total_steps,
            "converged": result.converged,
        }
        try:
            queue.put_nowait(done_event)
        except asyncio.QueueFull:
            pass

    except Exception as exc:
        run.status = "failed"
        run.error = str(exc)
        run.completed_at = datetime.utcnow()

        error_event: dict[str, Any] = {
            "type": "error",
            "run_id": run_id,
            "error": str(exc),
        }
        try:
            queue.put_nowait(error_event)
        except asyncio.QueueFull:
            pass

    finally:
        await state_store.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_summary(run: RunRecord) -> dict[str, Any]:
    """Return a compact summary dict for list endpoints."""
    return {
        "run_id": run.run_id,
        "status": run.status,
        "goal": run.goal[:120],
        "agent_id": run.agent_id,
        "created_at": run.created_at.isoformat(),
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "total_steps": len(run.steps),
        "error": run.error,
    }
