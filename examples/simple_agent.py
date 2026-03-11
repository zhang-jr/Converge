"""Simple Agent Example.

Demonstrates the basic usage of the Agent Framework.
This example shows:
1. Creating and registering tools
2. Using the ReconcileLoop pattern
3. Custom quality probes for convergence
4. State management

Note: This example uses mock LLM responses. In production,
you would integrate with a real LLM via LiteLLM.
"""

import asyncio
from typing import Any

from core.runtime.reconcile_loop import SimpleReconcileLoop
from core.state.models import DesiredState, LoopContext, StepOutput, ToolCall
from core.state.sqlite_store import SQLiteStateStore
from probes.quality_probe import ProbeResult, QualityProbe
from tools.base import ReadOnlyTool, ToolResult
from tools.registry import ToolRegistry


# =============================================================================
# Example Tools
# =============================================================================


class EchoTool(ReadOnlyTool):
    """A simple tool that echoes its input."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes the input message back"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        message = params.get("message", "")
        return ToolResult(
            success=True,
            output=f"Echo: {message}",
        )


class CalculatorTool(ReadOnlyTool):
    """A simple calculator tool."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Performs basic arithmetic operations (+, -, *, /)"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        operation = params.get("operation", "+")
        a = params.get("a", 0)
        b = params.get("b", 0)

        try:
            if operation == "+":
                result = a + b
            elif operation == "-":
                result = a - b
            elif operation == "*":
                result = a * b
            elif operation == "/":
                if b == 0:
                    return ToolResult(
                        success=False,
                        error="Division by zero",
                    )
                result = a / b
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown operation: {operation}",
                )

            return ToolResult(
                success=True,
                output=result,
                metadata={"operation": operation, "a": a, "b": b},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
            )


# =============================================================================
# Custom Quality Probe
# =============================================================================


class StepLimitProbe(QualityProbe):
    """Probe that converges after a specified number of steps."""

    def __init__(self, max_steps: int = 3):
        self.max_steps = max_steps

    async def evaluate(
        self,
        step_output: StepOutput,
        context: LoopContext,
    ) -> ProbeResult:
        if step_output.step_number >= self.max_steps:
            return ProbeResult(
                verdict="passed",
                confidence=0.95,
                reason=f"Completed {self.max_steps} steps successfully",
                should_converge=True,
            )

        return ProbeResult(
            verdict="passed",
            confidence=0.8,
            reason=f"Step {step_output.step_number}/{self.max_steps} completed",
            should_converge=False,
        )


# =============================================================================
# Example 1: Basic ReconcileLoop
# =============================================================================


async def basic_reconcile_example():
    """Basic example using SimpleReconcileLoop directly."""
    print("=" * 60)
    print("Example 1: Basic ReconcileLoop")
    print("=" * 60)

    store = SQLiteStateStore(":memory:")
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(EchoTool())

    calculations = [
        {"operation": "+", "a": 10, "b": 5},
        {"operation": "*", "a": 3, "b": 7},
        {"operation": "-", "a": 100, "b": 42},
    ]

    async def act_callback(
        diff: dict[str, Any],
        context: LoopContext,
        loop: SimpleReconcileLoop,
    ) -> StepOutput:
        """Simulated agent action - performs calculations."""
        step = context.current_step
        calc_idx = (step - 1) % len(calculations)
        calc = calculations[calc_idx]

        tool = registry.get_tool("calculator", check_permission=False)
        result = await tool.execute(calc)

        await store.put(
            f"result/step_{step}",
            {"calculation": calc, "result": result.output},
            updated_by="example-agent",
        )

        return StepOutput(
            step_number=step,
            action=f"Calculated {calc['a']} {calc['operation']} {calc['b']}",
            reasoning="Performing requested calculation",
            tool_calls=[
                ToolCall(
                    tool_name="calculator",
                    params=calc,
                    result=result.output,
                    success=result.success,
                )
            ],
            state_changes=[f"result/step_{step}"],
            result=result.output,
        )

    loop = SimpleReconcileLoop(
        state_store=store,
        tool_registry=registry,
        quality_probe=StepLimitProbe(max_steps=3),
        act_callback=act_callback,
        agent_id="example-agent",
    )

    result = await loop.run(
        DesiredState(
            goal="Perform a series of calculations",
            constraints=["Use the calculator tool"],
        )
    )

    print(f"\nResult Status: {result.status}")
    print(f"Converged: {result.converged}")
    print(f"Total Steps: {result.total_steps}")
    print(f"Duration: {result.duration_ms:.2f}ms")

    print("\nSteps:")
    for step in result.steps:
        print(f"  Step {step.step_number}: {step.action} = {step.result}")

    print("\nState Store Contents:")
    entries = await store.list()
    for entry in entries:
        print(f"  {entry.key}: {entry.value}")

    await store.close()
    return result


# =============================================================================
# Example 2: State-Driven Convergence
# =============================================================================


class GoalAchievedProbe(QualityProbe):
    """Probe that converges when a target sum is reached in state."""

    def __init__(self, target_sum: float):
        self.target_sum = target_sum

    async def evaluate(
        self,
        step_output: StepOutput,
        context: LoopContext,
    ) -> ProbeResult:
        current_sum = context.state_snapshot.get("running_total", {}).get("value", 0)

        if current_sum >= self.target_sum:
            return ProbeResult(
                verdict="passed",
                confidence=0.99,
                reason=f"Target sum {self.target_sum} reached (current: {current_sum})",
                should_converge=True,
            )

        return ProbeResult(
            verdict="passed",
            confidence=0.7,
            reason=f"Current sum: {current_sum}, target: {self.target_sum}",
            should_converge=False,
        )


async def state_driven_example():
    """Example showing state-driven convergence."""
    print("\n" + "=" * 60)
    print("Example 2: State-Driven Convergence")
    print("=" * 60)

    store = SQLiteStateStore(":memory:")
    await store.put("running_total", {"value": 0})

    async def accumulate_callback(
        diff: dict[str, Any],
        context: LoopContext,
        loop: SimpleReconcileLoop,
    ) -> StepOutput:
        """Add 10 to the running total each step."""
        current = await store.get("running_total")
        current_value = current.value["value"] if current else 0
        new_value = current_value + 10

        await store.put(
            "running_total",
            {"value": new_value},
            expected_version=current.version if current else None,
            updated_by="accumulator-agent",
        )

        return StepOutput(
            step_number=context.current_step,
            action=f"Added 10 to total",
            reasoning=f"Incrementing toward target",
            state_changes=["running_total"],
            result=f"Total is now {new_value}",
        )

    loop = SimpleReconcileLoop(
        state_store=store,
        quality_probe=GoalAchievedProbe(target_sum=50),
        act_callback=accumulate_callback,
        agent_id="accumulator-agent",
        safety_max_steps=10,
    )

    result = await loop.run(
        DesiredState(goal="Accumulate to 50")
    )

    print(f"\nResult Status: {result.status}")
    print(f"Steps taken: {result.total_steps}")

    final_total = await store.get("running_total")
    print(f"Final total: {final_total.value['value'] if final_total else 0}")

    await store.close()
    return result


# =============================================================================
# Example 3: Tool Execution with Error Handling
# =============================================================================


async def tool_execution_example():
    """Example showing tool execution and error handling."""
    print("\n" + "=" * 60)
    print("Example 3: Tool Execution with Error Handling")
    print("=" * 60)

    store = SQLiteStateStore(":memory:")
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    operations = [
        {"operation": "+", "a": 100, "b": 50},
        {"operation": "/", "a": 150, "b": 0},  # Will fail
        {"operation": "*", "a": 10, "b": 10},
    ]
    op_index = 0

    async def operate_callback(
        diff: dict[str, Any],
        context: LoopContext,
        loop: SimpleReconcileLoop,
    ) -> StepOutput:
        nonlocal op_index
        op = operations[op_index % len(operations)]
        op_index += 1

        tool = registry.get_tool("calculator", check_permission=False)
        result = await tool.execute(op)

        return StepOutput(
            step_number=context.current_step,
            action=f"Attempted {op['a']} {op['operation']} {op['b']}",
            reasoning="Executing operation",
            tool_calls=[
                ToolCall(
                    tool_name="calculator",
                    params=op,
                    result=result.output,
                    success=result.success,
                    error=result.error,
                )
            ],
            result="Success" if result.success else f"Failed: {result.error}",
        )

    loop = SimpleReconcileLoop(
        state_store=store,
        quality_probe=StepLimitProbe(max_steps=3),
        act_callback=operate_callback,
        agent_id="operator-agent",
    )

    result = await loop.run(
        DesiredState(goal="Execute operations")
    )

    print(f"\nResult Status: {result.status}")
    print("\nOperations:")
    for step in result.steps:
        status = "OK" if step.tool_calls[0].success else "FAILED"
        print(f"  Step {step.step_number}: {step.action} [{status}]")
        if not step.tool_calls[0].success:
            print(f"    Error: {step.tool_calls[0].error}")

    await store.close()
    return result


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Agent Framework Examples")
    print("=" * 60)
    print("\nThese examples demonstrate the core patterns of the framework")
    print("using mock actions. In production, integrate with LLM via LiteLLM.\n")

    await basic_reconcile_example()
    await state_driven_example()
    await tool_execution_example()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
