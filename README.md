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

| Kubernetes | Converge | 说明 |
|------------|----------|------|
| Container | `AgentCall` | 单次 LLM 调用，无状态 |
| Pod | `Agent` | AgentCall + Tools + Memory 最小部署单元 |
| Deployment | `AgentWorkflow` | 声明期望目标，框架负责收敛 |
| Operator | `SpecialistAgent` | 领域专家模式 |
| kubelet | `AgentRuntime` | 本地执行引擎 |
| Control Loop | `ReconcileLoop` | observe → diff → act → verify → repeat |
| etcd | `StateStore` | 单一事实来源 |
| Namespace | `AgentNamespace` | 多租户隔离 |
| RBAC | `AgentRBAC` | Tool 权限管理 |
| Health Probe | `QualityProbe` | 输出质量 / 置信度 / 循环检测 |
| Scheduler | `AgentScheduler` | 优先级任务路由 |

**核心原则：**
- 声明式 API 优先 — 描述想要什么，框架负责怎么做
- 控制循环驱动 — `QualityProbe` 判定收敛，不硬编码步数
- 单一事实来源 — 所有状态只写 `StateStore`
- Tool 副作用声明 — `risk_level` 自动插入 Human-in-the-loop
- 可观测性不可选 — 每步产生结构化 Trace（含 trace_id、reasoning、tool I/O）
- 失败时停下来 — 置信度低触发人工干预，循环检测立即中断

## 安装

**Python 3.11+ 必须。**

```bash
pip install -e .
```

可选依赖：

```bash
# FastAPI HTTP / WebSocket API 服务器
pip install -e ".[api]"

# OpenTelemetry 支持（对接 Langfuse / Jaeger）
pip install -e ".[otel]"

# PostgreSQL StateStore（生产环境）
pip install -e ".[postgres]"

# 向量记忆（ChromaDB，语义搜索）
pip install -e ".[vector]"

# 开发工具（pytest、ruff、mypy）
pip install -e ".[dev]"
```

> `[docker]` 无需 Python 包，仅需本机安装 Docker CLI 即可使用 `DockerSandbox`。

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
from api.declarative import agent, run_agent

result = await run_agent(
    "Analyze and summarize the data",
    agent_config=agent(
        "analyst",
        tools=["read_file", "search_code"],
        model="anthropic/claude-opus-4-6",
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

### Skill 系统

```python
from api.declarative import AgentFramework
from skills.builtin.code_review import create_code_review_skill

framework = AgentFramework()
framework.load_skill(create_code_review_skill())

# Skill 自动注入工具 + system_prompt + 收敛判定标准
result = await framework.run("Review the authentication module for security issues")
```

自定义 Skill：

```python
from skills.base import SkillBase

class MySkill(SkillBase):
    name = "my_skill"
    description = "Domain-specific capability"
    tools = [my_tool_instance]
    system_prompt_addon = "You are an expert in..."
    convergence_criteria = ["Issue list produced", "Severity ratings assigned"]
```

### 沙箱隔离

```python
from tools.sandbox.subprocess_sandbox import SubprocessSandbox
from tools.sandbox.docker_sandbox import DockerSandbox
from tools.code.shell_tools import BashTool

# 进程级沙箱（跨平台，Unix 支持资源限制）
sandbox = SubprocessSandbox(timeout=30.0)
bash = BashTool(sandbox=sandbox)

# 容器级沙箱（需 Docker CLI）
docker_sandbox = DockerSandbox(
    image="python:3.11-slim",
    timeout=60.0,
    network_disabled=True,
)
bash = BashTool(sandbox=docker_sandbox)
```

### Snapshot / Rollback

```python
# 手动 snapshot
snapshot_id = await store.snapshot(label="before-risky-op")

# 执行操作...

# 出错时还原
await store.restore(snapshot_id)

# ReconcileLoop 自动 rollback（每步 act 前自动打快照）
agent = Agent(config=config, state_store=store, tool_registry=registry)
result = await agent.run(
    goal="Refactor the module",
    enable_rollback=True,   # act 抛异常 → 自动还原上一步状态
)
```

### 步骤回调（流式进度）

```python
async def on_step(step_result):
    print(f"Step {step_result.step_number}: {step_result.action}")

result = await agent.run(
    goal="...",
    step_callback=on_step,
)
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

### HTTP API 服务器

```bash
pip install -e ".[api]"
python -m api.server
```

```python
import httpx, asyncio, websockets, json

async def main():
    async with httpx.AsyncClient() as client:
        # 创建 run
        resp = await client.post("http://localhost:8000/runs", json={
            "goal": "Summarize the codebase structure",
            "agent_id": "analyst",
        })
        run_id = resp.json()["run_id"]

    # WebSocket 实时流式接收步骤事件
    async with websockets.connect(f"ws://localhost:8000/runs/{run_id}/stream") as ws:
        async for msg in ws:
            event = json.loads(msg)
            print(event)

asyncio.run(main())
```

## 架构分层

```
Application    ← api/declarative.py  —  AgentFramework, goal(), agent()
Orchestration  ← core/workflow/      —  WorkflowController, AgentScheduler
Agent          ← core/agent/         —  Agent, MultiAgentOrchestrator
Runtime        ← core/runtime/       —  AgentRuntime, ReconcileLoop
Infra          ← tools/ memory/ skills/ core/state/ observability/ probes/
```

## 模块一览

### 核心运行时

| 模块 | 说明 |
|------|------|
| `core/runtime/reconcile_loop.py` | ReconcileLoop ABC + SimpleReconcileLoop；支持 step_callback、enable_rollback |
| `core/runtime/agent_runtime.py` | AgentRuntime，集成 Metrics + Human 干预 |
| `core/runtime/scheduler.py` | AgentScheduler（优先级队列 + 并发限制） |
| `core/runtime/human_intervention.py` | HumanInterventionHandler（CLI / Callback 两种实现） |

### 状态存储

| 模块 | 说明 |
|------|------|
| `core/state/state_store.py` | StateStore ABC（乐观锁 + snapshot/restore 接口） |
| `core/state/sqlite_store.py` | SQLite 实现（开发 / 测试，含 state_snapshots 表） |
| `core/state/postgres_store.py` | PostgreSQL 实现（生产，asyncpg，LISTEN/NOTIFY） |
| `core/state/models.py` | 全部 Pydantic 模型 |

### Tool 体系

| 模块 | 说明 |
|------|------|
| `tools/base.py` | ToolBase ABC；ReadOnlyTool / StateMutatingTool 便捷基类 |
| `tools/registry.py` | ToolRegistry，权限检查 |
| `tools/rbac.py` | RBACManager，Role；内置 read_only / operator / admin |
| `tools/code/file_tools.py` | ReadFileTool（low）/ WriteFileTool（medium）/ EditFileTool（medium） |
| `tools/code/search_tools.py` | GlobTool / GrepTool |
| `tools/code/shell_tools.py` | BashTool（high，沙箱路由）/ KillShellTool（high） |

### 沙箱

| 模块 | 说明 |
|------|------|
| `tools/sandbox/base.py` | SandboxBase ABC，SandboxResult，ResourceLimits |
| `tools/sandbox/subprocess_sandbox.py` | SubprocessSandbox；Unix 支持 rlimit CPU/内存/文件大小/进程数 |
| `tools/sandbox/docker_sandbox.py` | DockerSandbox；`docker run --rm`，无需 Python docker SDK |

### 编排层

| 模块 | 说明 |
|------|------|
| `core/workflow/workflow.py` | WorkflowSpec，WorkflowStep，WorkflowStepStatus |
| `core/workflow/controller.py` | sequential / parallel / dag 三种执行模式 |
| `core/agent/agent.py` | Agent + AgentReconcileLoop；run(step_callback, enable_rollback) |
| `core/agent/multi_agent.py` | MultiAgentOrchestrator；pipeline / supervisor / pool 三种协作模式 |

### 质量探针

| 模块 | 说明 |
|------|------|
| `probes/quality_probe.py` | QualityProbe ABC + DefaultQualityProbe + CompositeQualityProbe + ConvergenceCriteriaProbe |
| `probes/loop_detector.py` | LoopDetectorProbe（滑动窗口，action + tool 指纹检测） |
| `probes/confidence_probe.py` | ConfidenceProbe（tool 成功率 / 推理质量 / 步骤进展） |

### 记忆系统

| 模块 | 说明 |
|------|------|
| `memory/working.py` | WorkingMemory（LRU + TTL + 标签过滤） |
| `memory/context_manager.py` | ContextWindowManager（token 预算，LRU 淘汰） |
| `memory/episodic.py` | EpisodicMemory（SQLite 关键词）+ VectorEpisodicMemory（ChromaDB 语义搜索） |
| `memory/scratchpad.py` | AgentScratchpad（单次 run 内的短暂记事本，无 TTL/LRU） |

### Skill 系统

| 模块 | 说明 |
|------|------|
| `skills/base.py` | SkillBase（Pydantic，含 tools + system_prompt_addon + convergence_criteria） |
| `skills/registry.py` | SkillRegistry（dict 存储，支持按名列举） |
| `skills/builtin/code_review.py` | 内置 code_review skill |

### 可观测性

| 模块 | 说明 |
|------|------|
| `observability/tracer.py` | Tracer（JSON 结构化日志）+ OTelTracer（OpenTelemetry，可选） |
| `observability/metrics.py` | MetricsCollector（in-memory + Prometheus 导出，全局单例） |
| `observability/audit_log.py` | AuditLog（仅追加 SQLite，按 actor / action 过滤） |

### API 服务器

| 模块 | 说明 |
|------|------|
| `api/server.py` | FastAPI 服务器；POST/GET /runs，WS /runs/{id}/stream，GET /health |
| `api/declarative.py` | AgentFramework（load_skill, configure_agent, run） |

### 多租户

| 模块 | 说明 |
|------|------|
| `namespace/namespace.py` | Namespace + NamespaceManager（键前缀隔离，多租户） |

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

## StateStore

### 乐观锁写入

```python
entry = await store.get("my-key")

# 版本不匹配 → 抛 VersionConflictError
await store.put(
    "my-key",
    {"value": new_value},
    expected_version=entry.version,
    updated_by="my-agent",
)
```

### 实时变更通知

```python
async for event in store.watch(prefix="tasks/"):
    print(event.key, event.change_type)
```

### Snapshot / Restore

```python
# 打快照
snapshot_id = await store.snapshot(label="pre-migration")

# 查看所有快照
snapshots = await store.list_snapshots()

# 还原
await store.restore(snapshot_id)

# 删除快照
await store.delete_snapshot(snapshot_id)
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
# 基础 ReconcileLoop
python examples/simple_agent.py

# 多 Agent + Workflow
python examples/multi_agent_workflow.py

# Specialist Agent + Scheduler + Metrics + AuditLog
python examples/specialist_agent.py

# 沙箱、Rollback、API Server、DockerSandbox
python examples/api_server_example.py
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
│   └── VersionConflictError        # 乐观锁冲突
├── ReconcileError
│   ├── ConvergenceTimeoutError     # 超过 safety_max_steps
│   └── LoopDetectedError           # 循环检测触发
├── ToolExecutionError
├── ToolPermissionError
├── SandboxError                    # 沙箱执行失败（超时 / OOM / 权限）
├── RollbackError                   # Snapshot restore 失败
├── HumanInterventionRequired       # 流程控制信号，非错误
└── QualityProbeFailure             # 携带 ProbeResult
```

## 技术栈

| 组件 | 方案 |
|------|------|
| LLM 调用 | LiteLLM（多模型切换，默认 Anthropic） |
| 状态存储 | SQLite（开发）/ PostgreSQL（生产） |
| 序列化 | Pydantic v2 |
| 异步 | asyncio |
| Trace | OpenTelemetry（可选，对接 Langfuse / Jaeger） |
| 向量记忆 | ChromaDB（可选） |
| API 服务器 | FastAPI + uvicorn（可选） |
| 沙箱 | subprocess（内置）/ Docker（需 Docker CLI） |
| 测试 | pytest + pytest-asyncio |

## License

MIT
