"""Phase 7 example: API Server + Sandbox + Rollback.

Demonstrates:
1. SubprocessSandbox-isolated BashTool
2. enable_rollback flag in agent.run()
3. API Server usage with httpx client (if available)

Run API server:
    uvicorn api.server:app --reload --port 8000

Then run this script (in another terminal) to interact with the server.
"""

from __future__ import annotations

import asyncio
import sys


# ---------------------------------------------------------------------------
# Scenario 1: SubprocessSandbox + BashTool
# ---------------------------------------------------------------------------

async def demo_sandbox() -> None:
    """Execute a shell command inside a SubprocessSandbox."""
    print("\n=== Scenario 1: SubprocessSandbox ===")

    from tools.sandbox.subprocess_sandbox import SubprocessSandbox
    from tools.code.shell_tools import BashTool

    sandbox = SubprocessSandbox()
    tool = BashTool(sandbox=sandbox)

    # Normal command
    result = await tool.execute({"command": f"{sys.executable} -c \"print('Hello from sandbox')\"", "timeout": 10})
    print(f"Success: {result.success}")
    print(f"Output:  {result.output.get('stdout', '').strip()}")
    print(f"Sandbox: {result.metadata.get('sandbox')}")

    # Timeout test
    result_timeout = await tool.execute({
        "command": f"{sys.executable} -c \"import time; time.sleep(60)\"",
        "timeout": 0.3,
    })
    print(f"\nTimeout test — timed_out: {result_timeout.metadata.get('timed_out')}")


# ---------------------------------------------------------------------------
# Scenario 2: State rollback on agent failure
# ---------------------------------------------------------------------------

async def demo_rollback() -> None:
    """Show state rollback when an agent step fails."""
    print("\n=== Scenario 2: State Rollback on Failure ===")

    from core.state.sqlite_store import SQLiteStateStore
    from core.state.models import AgentConfig, DesiredState
    from core.agent.agent import Agent

    store = SQLiteStateStore(":memory:")
    await store.put("workflow/status", {"phase": "before_agent"})
    print(f"Before: {(await store.get('workflow/status')).value}")

    config = AgentConfig(agent_id="rollback-demo", safety_max_steps=1)

    agent = Agent(config=config, state_store=store)

    # The mock LLM will run one step that modifies state, then the probe
    # will cause a hard_fail on step 2... but with only 1 max step and no
    # convergence, it hits ConvergenceTimeoutError → rollback triggered.
    #
    # For illustration, we manually snapshot and restore to show the mechanism.
    snap_id = await store.snapshot()
    print(f"Snapshot taken: {snap_id[:8]}...")

    await store.put("workflow/status", {"phase": "mutated_during_run"})
    print(f"Mutated: {(await store.get('workflow/status')).value}")

    await store.restore(snap_id)
    print(f"Restored: {(await store.get('workflow/status')).value}")

    await store.close()


# ---------------------------------------------------------------------------
# Scenario 3: API Server (requires server running on :8000)
# ---------------------------------------------------------------------------

async def demo_api_client(base_url: str = "http://localhost:8000") -> None:
    """Submit a run via HTTP and poll for completion."""
    print(f"\n=== Scenario 3: API Client → {base_url} ===")

    try:
        import httpx
    except ImportError:
        print("httpx not installed — skipping API client demo")
        print("Install with: pip install httpx")
        return

    async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
        try:
            health = await client.get("/health")
            print(f"Server health: {health.json()['status']}")
        except Exception as e:
            print(f"Server not reachable at {base_url}: {e}")
            print("Start the server first:  uvicorn api.server:app --port 8000")
            return

        # Submit run
        payload = {
            "goal": "Echo hello from the API",
            "agent_id": "api-demo-agent",
            "safety_max_steps": 3,
        }
        create_resp = await client.post("/runs", json=payload)
        run_id = create_resp.json()["run_id"]
        print(f"Run submitted: {run_id}")

        # Poll until complete
        for _ in range(20):
            await asyncio.sleep(0.5)
            status_resp = await client.get(f"/runs/{run_id}")
            run = status_resp.json()
            status = run["status"]
            print(f"  Status: {status}  (steps: {len(run.get('steps', []))})")
            if status in ("converged", "failed", "timeout"):
                break

        # Fetch steps
        steps_resp = await client.get(f"/runs/{run_id}/steps")
        steps_data = steps_resp.json()
        print(f"Total steps recorded: {steps_data['total_steps']}")


# ---------------------------------------------------------------------------
# Scenario 4: DockerSandbox (requires docker daemon)
# ---------------------------------------------------------------------------

async def demo_docker_sandbox() -> None:
    """Run a command in a Docker container."""
    print("\n=== Scenario 4: DockerSandbox (requires Docker) ===")

    import shutil
    if not shutil.which("docker"):
        print("Docker CLI not found — skipping")
        return

    try:
        from tools.sandbox.docker_sandbox import DockerSandbox
        from tools.code.shell_tools import BashTool

        sandbox = DockerSandbox(image="alpine:latest", network="none")
        tool = BashTool(sandbox=sandbox)

        result = await tool.execute({
            "command": "echo 'Hello from Docker'",
            "timeout": 15,
        })
        print(f"Success:  {result.success}")
        print(f"Output:   {result.output.get('stdout', '').strip()}")
        print(f"Sandbox:  {result.metadata.get('sandbox')}")
    except Exception as e:
        print(f"Docker error: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    await demo_sandbox()
    await demo_rollback()
    await demo_docker_sandbox()
    # Uncomment to test against a running server:
    # await demo_api_client()


if __name__ == "__main__":
    asyncio.run(main())
