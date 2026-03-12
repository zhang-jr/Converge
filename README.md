# Converge

<p align="center">
  <img src="assets/banner.svg" alt="Converge Banner" width="100%"/>
</p>

> Manage Agents like Kubernetes manages containers.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-red)
![Async](https://img.shields.io/badge/Async-asyncio-purple)
![LLM](https://img.shields.io/badge/LLM-LiteLLM-orange)

**Converge** is a production-ready Python framework for orchestrating AI agents at scale, inspired by Kubernetes design philosophy.

You declare *what* you want. Converge figures out *how* — driving agent behavior through self-correcting control loops until the desired state is reached. No hardcoded steps. No silent failures. Full observability by default.

Built for engineers who need reliable, multi-agent systems they can trust in production.

## 设计理念

| Kubernetes | Agent Framework | 说明 |
|------------|-----------------|------|
| Container  | `AgentCall`     | 单次 LLM 调用，无状态 |
| Pod        | `Agent`         | AgentCall + Tools + Memory |
| Deployment | `AgentWorkflow` | 声明期望目标，框架负责收敛 |
| kubelet    | `AgentRuntime`  | 本地执行引擎 |
| Control Loop | `ReconcileLoop` | observe → diff → act → verify → repeat |
| etcd       | `StateStore`    | 单一事实来源 |
| Scheduler  | `AgentScheduler` | 优先级任务调度 |
| RBAC       | `AgentRBAC`     | Tool 权限管理 |

**核心原则：**
- 声明式 API 优先 — 描述想要什么，框架负责怎么做
- 控制循环驱动 — `QualityProbe` 判定收敛，不硬编码步数
- 单一事实来源 — 所有状态只写 `StateStore`
- Tool 副作用声明 — `risk_level` 自动插入 Human-in-the-loop
- 可观测性不可选 — 每步产生结构化 Trace（含 trace_id、reasoning、tool I/O）

## 安装

**Python 3.11+ 必须。**

```bash
pip install -e .
```

可选依赖：

```bash
# OpenTelemetry 支持（对接 Langfuse / Jaeger）
pip install -e ".[otel]"

# PostgreSQL StateStore（生产环境）
pip install -e ".[postgres]"

# 开发工具（pytest、ruff、mypy）
pip install -e ".[dev]"
```

## 快速开始

### 最简示例

```python
import asyncio
from api.declarative import AgentFramework
from tools.base import ReadOnlyTool, ToolResult

class MyTool(ReadOnlyTool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "A simple example tool"

    async def execute(self, params: dict) -> ToolResult:
        return ToolResult(success=True, output="Hello from tool!")

async def main():
    framework = AgentFramework()
    framework.register_tool(MyTool())
    framework.configure_agent(
        agent_id="my-agent",
        tools=["my_tool"],
        system_prompt="You are a helpful assistant.",
    )

    result = await framework.run("Complete a simple task")
    print(f"Status: {result.status}, Steps: {result.total_steps}")

asyncio.run(main())
```

### 声明式 API

```python
from api.declarative import agent, goal, run_agent

result = await run_agent(
    "Analyze and summarize the data",
    agent_config=agent(
        "analyst",
        tools=["read_file", "search_code"],
        model="anthropic/claude-3-5-sonnet-20241022",
        system_prompt="You are a data analysis expert.",
    ),
    constraints=["Only read, do not modify files"],
)
```

### 多 Agent 工作流

```python
from core.workflow.workflow import WorkflowSpec, WorkflowStep
from core.workflow.controller import WorkflowController

spec = WorkflowSpec(
    workflow_id="data-pipeline",
    steps=[
        WorkflowStep(step_id="fetch",   agent_id="fetcher",   goal="Fetch raw data"),
        WorkflowStep(step_id="clean",   agent_id="cleaner",   goal="Clean the data",
                     depends_on=["fetch"]),
        WorkflowStep(step_id="analyze", agent_id="analyzer",  goal="Analyze cleaned data",
                     depends_on=["clean"]),
    ],
    execution_mode="dag",
)

controller = WorkflowController(state_store=store, tool_registry=registry)
result = await controller.run(spec)
```

### Human-in-the-Loop

```python
from core.runtime.human_intervention import CLIHumanInterventionHandler

handler = CLIHumanInterventionHandler()
runtime = AgentRuntime(
    state_store=store,
    tool_registry=registry,
    human_intervention_handler=handler,   # high-risk tool 自动触发
)
```

### 自定义 QualityProbe

```python
from probes.quality_probe import QualityProbe, ProbeResult
from core.state.models import StepOutput, LoopContext

class MyProbe(QualityProbe):
    async def evaluate(self, step_output: StepOutput, context: LoopContext) -> ProbeResult:
        done = "completed" in (step_output.result or "").lower()
        return ProbeResult(
            verdict="passed" if done else "soft_fail",
            confidence=0.9 if done else 0.5,
            reason="Task completed" if done else "Still working...",
            should_converge=done,
        )
```

## 架构分层

```
Application    ← api/declarative.py  —  AgentFramework, goal(), agent()
Orchestration  ← core/workflow/      —  WorkflowController, AgentScheduler
Agent          ← core/agent/         —  Agent, MultiAgentOrchestrator
Runtime        ← core/runtime/       —  AgentRuntime, ReconcileLoop
Infra          ← tools/ memory/ core/state/ observability/ probes/
```

## 模块一览

### 核心运行时

| 模块 | 说明 |
|------|------|
| `core/runtime/reconcile_loop.py` | ReconcileLoop ABC + SimpleReconcileLoop |
| `core/runtime/agent_runtime.py`  | AgentRuntime，集成 Metrics + Human 干预 |
| `core/runtime/scheduler.py`      | AgentScheduler（优先级队列 + 并发限制） |
| `core/runtime/human_intervention.py` | HumanInterventionHandler（CLI / Callback） |

### 状态存储

| 模块 | 说明 |
|------|------|
| `core/state/state_store.py`  | StateStore ABC（乐观锁） |
| `core/state/sqlite_store.py` | SQLite 实现（开发 / 测试） |
| `core/state/postgres_store.py` | PostgreSQL 实现（生产，asyncpg）|
| `core/state/models.py`       | 全部 Pydantic 模型 |

### Tool 体系

| 模块 | 说明 |
|------|------|
| `tools/base.py`     | ToolBase ABC，副作用四元组声明 |
| `tools/registry.py` | ToolRegistry，权限检查 |
| `tools/rbac.py`     | RBACManager，Role，内置 read_only / operator / admin |

### 编排层

| 模块 | 说明 |
|------|------|
| `core/workflow/workflow.py`   | WorkflowSpec，WorkflowStep |
| `core/workflow/controller.py` | sequential / parallel / dag 三种执行模式 |
| `core/agent/multi_agent.py`   | pipeline / supervisor / pool 三种协作模式 |

### 质量探针

| 模块 | 说明 |
|------|------|
| `probes/quality_probe.py`   | QualityProbe ABC + DefaultQualityProbe + CompositeQualityProbe |
| `probes/loop_detector.py`   | LoopDetectorProbe（滑动窗口指纹检测） |
| `probes/confidence_probe.py` | ConfidenceProbe（tool成功率 / 推理质量 / 步骤进展） |

### 记忆系统

| 模块 | 说明 |
|------|------|
| `memory/working.py`        | WorkingMemory（LRU + TTL + 标签过滤） |
| `memory/context_manager.py` | ContextWindowManager（token 预算，LRU 淘汰） |
| `memory/episodic.py`       | EpisodicMemory（SQLite，关键词搜索） |

### 可观测性

| 模块 | 说明 |
|------|------|
| `observability/tracer.py`   | Tracer（JSON 结构化日志）+ OTelTracer（OpenTelemetry） |
| `observability/metrics.py`  | MetricsCollector（in-memory + Prometheus 导出） |
| `observability/audit_log.py` | AuditLog（仅追加 SQLite，actor / action 过滤） |

### 隔离

| 模块 | 说明 |
|------|------|
| `namespace/namespace.py` | Namespace + NamespaceManager（多租户键前缀隔离） |

## Tool 副作用声明

每个 Tool 必须声明四项元信息，框架据此自动决策：

```python
class DeleteFileTool(ToolBase):
    side_effects = ["删除文件系统文件"]
    reversible   = False
    risk_level   = "high"     # → 自动触发 Human-in-the-loop
    idempotent   = True
```

`risk_level` 取值：`"low"` | `"medium"` | `"high"`

## StateStore 乐观锁

```python
entry = await store.get("my-key")

# 乐观锁写入：版本不匹配抛 VersionConflictError
await store.put(
    "my-key",
    {"value": new_value},
    expected_version=entry.version,
    updated_by="my-agent",
)

# 实时变更通知
async for event in store.watch(prefix="tasks/"):
    print(event.key, event.change_type)
```

## PostgreSQL（生产环境）

```python
from core.state.postgres_store import PostgreSQLStateStore

store = await PostgreSQLStateStore.create(
    dsn="postgresql://user:pass@localhost/agentdb"
)
```

支持 `LISTEN/NOTIFY` 实时推送，连接池由 asyncpg 管理。

## 运行示例

```bash
# Phase 1：基础 ReconcileLoop
python examples/simple_agent.py

# Phase 2：多 Agent + Workflow
python examples/multi_agent_workflow.py

# Phase 3：Specialist Agent + Scheduler + Metrics + AuditLog
python examples/specialist_agent.py
```

## 运行测试

```bash
pytest
pytest --cov=. --cov-report=html   # 覆盖率报告
pytest tests/unit/test_state_store.py -v  # 单个模块
```

> PostgreSQL 集成测试需真实 PG 实例，标记为 `@pytest.mark.integration`，默认跳过。

## 错误体系

```
AgentFrameworkError
├── StateStoreError
│   └── VersionConflictError      # 乐观锁冲突
├── ReconcileError
│   ├── ConvergenceTimeoutError   # 超过 safety_max_steps
│   └── LoopDetectedError         # 循环检测触发
├── ToolExecutionError
├── ToolPermissionError
├── HumanInterventionRequired     # 流程控制信号，非错误
└── QualityProbeFailure           # 携带 ProbeResult
```

## 技术栈

| 组件 | 方案 |
|------|------|
| LLM 调用 | LiteLLM（多模型切换，默认 Anthropic） |
| 状态存储 | SQLite（开发）/ PostgreSQL（生产） |
| 序列化 | Pydantic v2 |
| 异步 | asyncio |
| Trace | OpenTelemetry（可选） |
| 测试 | pytest + pytest-asyncio |

## 项目状态

| 阶段 | 状态 | 内容 |
|------|------|------|
| Phase 1 | 完成 | 核心骨架：异常、StateStore、Tool体系、QualityProbe、ReconcileLoop、Agent、声明式API |
| Phase 2 | 完成 | 编排层：Workflow+Controller、MultiAgent、RBAC、Namespace、WorkingMemory、EpisodicMemory |
| Phase 3 | 完成 | 生产就绪：LoopDetector、ConfidenceProbe、Metrics、AuditLog、OTelTracer、Human-in-the-loop、AgentScheduler、PostgreSQL |
| Phase 4 | 计划中 | 向量记忆（ChromaDB/pgvector）、真实LLM接入、性能基准 |

## License

MIT
