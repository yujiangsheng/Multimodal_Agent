# Pinocchio 架构设计文档

本文档详细描述 Pinocchio 自我进化多模态智能体的内部架构、设计原则和关键实现细节。

> 更新日期: 2025-07 · 版本 0.3.0 · 876 测试全部通过

---

## 目录

1. [设计哲学](#1-设计哲学)
2. [整体架构](#2-整体架构)
3. [统一认知智能体](#3-统一认知智能体)
4. [六阶段认知循环](#4-六阶段认知循环)
5. [双轴记忆系统](#5-双轴记忆系统)
6. [多模态处理](#6-多模态处理)
7. [安全防护层](#7-安全防护层)
8. [工具调用框架](#8-工具调用框架)
9. [流式输出与缓存](#9-流式输出与缓存)
10. [向量语义搜索](#10-向量语义搜索)
11. [性能优化](#11-性能优化)
12. [数据流详解](#12-数据流详解)
13. [扩展子系统](#13-扩展子系统)
14. [扩展指南](#14-扩展指南)

---

## 1. 设计哲学

Pinocchio 的设计基于三个核心原则：

1. **自我进化** — 不仅回答问题，还从每次交互中学习。通过结构化的评估和反思，持续优化策略和知识。
2. **认知启发** — 借鉴认知科学的理论模型（工作记忆、情景记忆、程序记忆），构建类人的学习和推理架构。
3. **统一简洁** — 单一智能体类包含所有认知技能，避免过度工程化。所有组件共享 LLM、记忆和日志基础设施。

---

## 2. 整体架构

### 系统分层

```
┌───────────────────────────────────────────────────────────┐
│  Layer 6 · Interface      CLI / FastAPI (web/) / API      │
├───────────────────────────────────────────────────────────┤
│  Layer 5 · Security       InputGuard · ContextManager     │
├───────────────────────────────────────────────────────────┤
│  Layer 4 · Orchestration  Pinocchio  (orchestrator.py)    │
├───────────────────────────────────────────────────────────┤
│  Layer 3 · Cognition      PinocchioAgent (6-phase loop)   │
│                           ToolRegistry · ToolExecutor     │
├───────────────────────────────────────────────────────────┤
│  Layer 2 · Memory         Episodic · Semantic · Procedural│
│                           WorkingMemory · EmbeddingClient │
│                           ResponseCache                   │
├───────────────────────────────────────────────────────────┤
│  Layer 1 · Infrastructure LLMClient · Logger · Monitor    │
│                           ParallelExecutor                │
└───────────────────────────────────────────────────────────┘
```

### 组件交互

```
┌─────────────────────────────────────────────────────────────┐
│                     用户 (CLI / Web / API)                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Pinocchio (orchestrator.py)                │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │  Session State    │  │  Multimodal Processor Pool       │ │
│  │  · UserModel      │  │  · TextProcessor                 │ │
│  │  · History        │  │  · VisionProcessor               │ │
│  │  · Interaction #  │  │  · AudioProcessor                │ │
│  │                   │  │  · VideoProcessor                │ │
│  └──────────────────┘  └──────────────────────────────────┘ │
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │               PinocchioAgent (unified_agent.py)          ││
│  │                                                          ││
│  │  perceive() → strategize() → execute() → evaluate()     ││
│  │                                    → learn()             ││
│  │                                    → meta_reflect()      ││
│  └──────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │             MemoryManager (memory_manager.py)            ││
│  │                                                          ││
│  │  Content axis:  EpisodicMemory · SemanticMemory          ││
│  │                 · ProceduralMemory                       ││
│  │  Temporal axis: WorkingMemory (volatile)                 ││
│  └──────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌─────────────────┐  ┌──────────────────────────┐          │
│  │   LLMClient      │  │   PinocchioLogger         │         │
│  │   (sync + async)  │  │   (colour-coded)          │         │
│  └─────────────────┘  └──────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
              ┌────────────────┐
              │  Ollama Server  │
              │  (Qwen3-VL)    │
              └────────────────┘
```

### 继承体系

```
BaseAgent (ABC)              ← 抽象基类：提供 LLM / 记忆 / 日志访问
├── PinocchioAgent           ← 统一认知智能体 (6 种技能方法)
├── TextProcessor            ← 文本模态处理
├── VisionProcessor          ← 图像模态处理
├── AudioProcessor           ← 音频模态处理
└── VideoProcessor           ← 视频模态处理
```

---

## 3. 统一认知智能体

`PinocchioAgent` 将六个认知阶段实现为同一个类的方法，而非六个独立的类。

**设计理由：**

| 方面 | 统一设计 | 分离设计 |
|------|----------|----------|
| 代码量 | 1 个文件 ~650 行 | 6 个文件 ~1000 行 |
| 上下文共享 | 自然共享 self | 需要参数传递 |
| 测试 | Mock 1 个类 | Mock 6 个类 |
| 新技能 | 添加方法 | 添加文件 + 注册 |
| 职责分离 | 通过命名约定 | 通过类边界 |

每个技能方法有独立的系统 prompt 常量（`_PERCEIVE_PROMPT`、`_STRATEGIZE_PROMPT` 等），确保 LLM 在每个阶段有清晰的角色定义。

---

## 4. 六阶段认知循环

### 完整流程

```
用户输入
   │
   ▼
[Fast Path Check] ─── 纯文本 ≤500字 ──→ 单次 LLM 调用 → 返回
   │
   │ (复杂输入)
   ▼
[Phase 0.5: PREPROCESS MODALITIES]
   │  并行处理 image/audio/video → 文本描述
   ▼
[Phase 1: PERCEIVE]
   │  分析输入 → PerceptionResult
   │  (task_type, complexity, confidence, similar_episodes)
   ▼
[Phase 2: STRATEGIZE]
   │  选择策略 → StrategyResult
   │  (selected_strategy, fusion_strategy, risk_assessment)
   ▼
[Phase 3: EXECUTE]
   │  生成响应 → AgentMessage
   │  ├── 自动续写 (finish_reason="length" → 最多2轮)
   │  └── 完整性检查 (启发式检测)
   ▼
[Phase 4: EVALUATE]
   │  评估质量 → EvaluationResult
   │  (output_quality, is_complete, incompleteness_details)
   │
   ├── 不完整 → [Phase 4.5: COMPLETION RETRY] (最多1次)
   │               └── continue_response() → 重新 EVALUATE
   ▼
[返回响应给用户]
   │
   └── [后台线程]
       ├── [Phase 5: LEARN]
       │   存储情景记忆 + 语义知识 + 程序模板
       ├── [Phase 6: META-REFLECT] (每 N 次交互)
       │   高阶自我分析 + 改进计划
       └── [Consolidation] (每 10 次交互)
           长期记忆 → 持久记忆 晋升
```

### 快速路径

对于简单的纯文本输入（≤500 字符），跳过 PERCEIVE / STRATEGIZE / EVALUATE，直接通过单次 LLM 调用返回：

- **判定条件：** 无图片/音频/视频，文本长度 ≤ 500 字符
- **上下文：** 从工作记忆获取最近 6 轮对话
- **性能：** 与直接调用 Ollama 基本一致

### 后台学习

Phase 5 (LEARN) 和 Phase 6 (META-REFLECT) 在 daemon 线程中异步执行，不阻塞用户响应返回。这节省了 2-3 次 LLM 调用的延迟。

---

## 5. 双轴记忆系统

### 内容轴 × 时间轴 矩阵

|                | Working (会话) | Long-term (跨会话) | Persistent (永久) |
|----------------|---------------|--------------------|--------------------|
| **Episodic**   | N/A           | 交互轨迹 (JSON)    | 高分情景 (≥8分)    |
| **Semantic**   | N/A           | 知识条目 (JSON)    | 高置信知识 (≥0.85) |
| **Procedural** | N/A           | 策略模板 (JSON)    | 高成功程序 (≥80%)  |
| **Context**    | 对话缓冲      | N/A                | N/A                |

### 性能优化

所有内容轴存储都使用**倒排索引**加速检索：

- `EpisodicMemory`: `task_type` + `modality` 倒排索引 → O(k) 检索
- `SemanticMemory`: `domain` 倒排索引 → O(D) 检索（D = 唯一域数）
- `ProceduralMemory`: `task_type` 倒排索引 → O(k log k) 检索

### 知识合成与晋升

```
情景记忆积累 (同一领域 ≥10 条)
    │
    ▼
[Knowledge Synthesis] → 蒸馏为语义知识条目
    │
    │  (每 10 次交互)
    ▼
[Consolidation] → 高价值条目晋升为持久记忆
                   ├── 高分情景 (outcome_score ≥ 8)
                   ├── 高置信知识 (confidence ≥ 0.85)
                   └── 高成功程序 (usage ≥ 5, rate ≥ 80%)
```

---

## 6. 多模态处理

### 处理流程

```
用户输入 (text + images + audio + video)
    │
    ├── text → 直接传入认知循环
    │
    └── 非文本模态 → _preprocess_modalities()
        │
        ├── [并行模式] ThreadPoolExecutor
        │   ├── VisionProcessor.run() → 图像描述文本
        │   ├── AudioProcessor.run()  → 音频转写/分析文本
        │   └── VideoProcessor.run()  → 视频分析文本
        │
        └── [顺序模式] (单核 / 配置禁用并行)
            └── 依次处理各模态
    │
    ▼
modality_context: dict[str, str]  ← 模态名 → 文本描述
    │
    ▼
传入 PinocchioAgent.execute() 作为额外上下文
```

### 视频处理双路径

```
VideoProcessor.run()
    │
    ├── native_video=True (默认)
    │   └── Qwen3-VL 原生视频理解 (build_video_message)
    │
    └── native_video=False (回退)
        ├── ffmpeg 抽帧 → VisionProcessor 逐帧分析
        ├── ffmpeg 提取音频 → AudioProcessor 音频分析
        └── 融合帧描述 + 音频分析 → 综合视频理解
```

---

## 7. 安全防护层

Pinocchio 提供五层纵深防御，由 `InputGuard` 和相关组件实现。

### 防护管线

```
用户输入
   │
   ▼
[Layer 1] 长度校验
   │  _MAX_INPUT_LENGTH = 32,000 字符
   │  超限 → is_safe=False, 拒绝
   ▼
[Layer 2] 注入模式检测
   │  10 组编译正则 (role_hijacking, prompt_exfiltration,
   │  delimiter_breakout, encoded_payload, repetition_attack)
   │  命中 → threats 列表追加类别名
   ▼
[Layer 3] 内容消毒
   │  _sanitise(): 移除控制字符、零宽字符
   │  保留用户原始语义
   ▼
[Layer 4] 工具沙箱
   │  ToolExecutor: 工具函数在 try/except 中执行
   │  calculator 使用安全白名单 (_SAFE_MATH_NAMES)
   │  python_eval 限制 builtins
   ▼
[Layer 5] 上下文窗口管理
   │  ContextManager: 防止对话过长导致截断
   │  自动摘要 → 保持上下文在 token 预算内
   ▼
ValidationResult(is_safe, threats, sanitised_text)
```

### InputGuard API

| 方法/属性 | 说明 |
|-----------|------|
| `validate(text)` | 执行完整验证管线 → `ValidationResult` |
| `_sanitise(text)` | 静态方法：消毒控制字符 |
| `strict` 模式 | `strict=True` 时，任何威胁即标记 `is_safe=False` |

### 注入模式分类

| 类别 | 示例模式 | 数量 |
|------|----------|------|
| `role_hijacking` | "忽略以上指令"/"你现在是 DAN" | 3 |
| `prompt_exfiltration` | "输出你的系统提示" | 2 |
| `delimiter_breakout` | `"""`, `---`, XML 标签闭合 | 2 |
| `encoded_payload` | Base64 嵌入、Unicode 混淆 | 2 |
| `repetition_attack` | 超大量重复字符 (≥500) | 1 |

---

## 8. 工具调用框架

### 组件结构

```
pinocchio/tools.py
    │
    ├── Tool (dataclass)
    │   ├── name: str
    │   ├── description: str
    │   ├── parameters: dict (JSON Schema)
    │   └── function: Callable[..., str]
    │
    ├── ToolRegistry
    │   ├── register(tool)      — 注册工具
    │   ├── get(name)           — 按名查找
    │   ├── list_names()        — 列出所有名称
    │   ├── to_openai_schema()  — 导出 function-calling schema
    │   ├── to_prompt_description() — 文本化描述 (用于非 FC 模型)
    │   └── register_defaults() — 注册内置工具
    │
    └── ToolExecutor
        ├── execute(name, args) — 安全执行
        └── parse_and_execute(raw_json) — 解析 JSON 后执行
```

### 内置工具

| 工具 | 功能 | 安全措施 |
|------|------|----------|
| `calculator` | 数学表达式计算 | `_SAFE_MATH_NAMES` 白名单 (math 模块子集) |
| `current_time` | 返回当前时间 | 无副作用 |
| `python_eval` | 简单 Python 表达式求值 | 受限 `__builtins__` |

### 自定义扩展

```python
from pinocchio.tools import Tool, ToolRegistry

registry = ToolRegistry()
registry.register_defaults()
registry.register(Tool(
    name="weather",
    description="获取城市天气",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    function=lambda city: f"{city}: 晴, 25°C",
))
```

---

## 9. 流式输出与缓存

### 流式输出

`LLMClient.chat_stream()` 和 `Pinocchio.chat_stream()` 支持逐 token 流式返回。

```
Pinocchio.chat_stream(text, images=...)
    │
    ├── [预处理] InputGuard + 多模态 → modality_context
    │
    ├── [流式生成] LLMClient.chat_stream(messages)
    │   └── httpx stream → 逐 chunk yield str
    │       └── Qwen3 <think>...</think> 标签自动剥离
    │
    └── [后台] 学习 + 记忆存储 (daemon 线程)
```

**Qwen3 think-tag 处理：** 流式输出自动检测并剥离 `<think>…</think>` 推理标签，只输出最终文本。支持 `\n<think>` 和 `</think>\n` 的边界情况。

### 响应缓存 (ResponseCache)

| 属性 | 值 |
|------|-----|
| 淘汰策略 | LRU (Least Recently Used) |
| 过期策略 | TTL (默认 600 秒) |
| 容量 | 默认 256 条 |
| 线程安全 | `threading.Lock` 保护 |
| 键生成 | SHA-256(`messages + extra`) |

```
ResponseCache 工作流
    │
    ├── get(key)
    │   ├── 命中且未过期 → 返回缓存值 + 移至 OrderedDict 尾部
    │   ├── 命中但已过期 → 删除 + 返回 None
    │   └── 未命中 → 返回 None
    │
    └── put(key, value)
        ├── 已存在 → 更新 + 移至尾部
        └── 新条目 → 插入尾部
            └── 超容量 → 弹出最久未用条目 (popitem FIFO)
```

---

## 10. 向量语义搜索

### EmbeddingClient

`EmbeddingClient` 通过 OpenAI 兼容的 embeddings API (默认使用 `nomic-embed-text` 模型) 生成文本向量。

```
配置
    PINOCCHIO_EMBEDDING_MODEL = "nomic-embed-text"  (环境变量)
    base_url = OPENAI_BASE_URL  (与 LLMClient 共享)

EmbeddingClient.embed(text) → list[float]
    │
    └── POST /v1/embeddings
        └── 返回 768 维向量 (nomic-embed-text)
```

### 记忆向量搜索

```
MemoryManager.set_embedding_client(client)
    │
    ├── store_episode(record)
    │   └── 自动调用 embed(record.lessons) → record.embedding
    │
    ├── store_knowledge(entry)
    │   └── 自动调用 embed(entry.knowledge) → entry.embedding
    │
    └── search_by_embedding(query_text, top_k)
        ├── embed(query_text) → query_vec
        ├── 遍历 episodic + semantic 记忆
        │   └── cosine_similarity(query_vec, record.embedding)
        └── 返回 top_k 最相似记忆条目
```

与倒排索引互补：倒排索引用于精确类别匹配，向量搜索用于语义相似度检索。

---

## 11. 性能优化

| 优化 | 机制 | 效果 |
|------|------|------|
| 快速路径 | 短文本跳过重型阶段 | 响应延迟降低 ~70% |
| 后台学习 | LEARN + META-REFLECT 在 daemon 线程执行 | 用户等待减少 2-3 次 LLM 调用 |
| HTTP 连接池 | httpx 连接复用 (20 max, 10 keepalive) | 减少 TCP 握手开销 |
| Qwen3 /no_think | JSON 调用禁用 thinking mode | 减少 ~50% JSON 调用的 token 消耗 |
| 倒排索引 | 记忆存储按 task_type/modality/domain 索引 | 检索从 O(n) 降至 O(k) |
| 并行模态 | 多线程并发处理 image/audio/video | 多模态输入延迟降低 ~60% |
| 空响应重试 | LLM 空响应自动升温重试 (最多3次) | 减少空响应率 |
| 响应缓存 | ResponseCache: LRU + TTL (SHA-256 key) | 重复查询零 LLM 开销 |
| 向量搜索 | EmbeddingClient + cosine similarity | 语义记忆召回率显著提升 |
| 上下文管理 | ContextManager 自动摘要 | 避免超窗口截断，保持对话连贯 |
| 流式输出 | chat_stream 逐 token 返回 | 首 token 延迟大幅降低 |

---

## 12. 数据流详解

### 核心数据类型

```
MultimodalInput    → PERCEIVE  → PerceptionResult
                   → STRATEGIZE → StrategyResult
                   → EXECUTE    → AgentMessage
                   → EVALUATE   → EvaluationResult
                   → LEARN      → LearningResult
                   → META-REFLECT → MetaReflectionResult
```

### 记忆数据类型

```
EpisodicRecord   — 交互轨迹 (episode_id, task_type, outcome_score, lessons)
SemanticEntry    — 知识条目 (entry_id, domain, knowledge, confidence)
ProceduralEntry  — 策略模板 (entry_id, task_type, name, steps, success_rate)
WorkingMemoryItem — 会话条目 (item_id, category, content, relevance)
```

所有记忆数据类支持 `to_dict()` / `from_dict()` JSON 往返序列化。

---

## 13. 扩展子系统

v0.3.0 引入了 7 个扩展子系统，全部集成在编排器中。

### 13.1 任务规划 (Planning)

**模块**: `pinocchio/planning/`

| 类 | 职责 |
|---|---|
| `TaskPlanner` | Plan-and-Solve 风格多步分解，LLM 生成结构化 JSON 计划 |
| `TaskPlan` / `TaskStep` | 计划与步骤的数据结构（含状态追踪、依赖关系） |
| `ReActExecutor` | Thought → Action → Observation 迭代循环 |
| `ReActTrace` / `ReActStep` | 推理过程的完整执行轨迹 |

**设计要点**:
- 编排器在 STRATEGIZE 阶段调用 `planner.should_plan(complexity, task_type)` 判断是否需要分解
- 复杂度 ≥ 3 或特定任务类型（analysis, code_generation, multimodal_reasoning）自动触发
- 支持失败后 `replan()` 重新规划

### 13.2 代码沙箱 (Sandbox)

**模块**: `pinocchio/sandbox/`

| 类 | 职责 |
|---|---|
| `CodeSandbox` | 隔离子进程安全执行 Python 代码 |
| `ExecutionResult` | 执行结果（stdout, stderr, exit_code, success, timed_out） |

**安全机制**:
- **静态检查**: 拒绝 `eval()`, `exec()`, `__import__`, `compile()`
- **Import 白名单**: 阻止 `os`, `subprocess`, `socket`, `shutil` 等 30+ 模块
- **文件系统保护**: 拦截写模式 `open()`，只允许读操作
- **超时**: 默认 15 秒，硬性上限 60 秒
- **隔离环境**: 子进程清空 `PATH`，限制在临时目录运行

### 13.3 RAG 知识库

**模块**: `pinocchio/rag/`

| 类 | 职责 |
|---|---|
| `DocumentStore` | SQLite 持久化管理，文档分块，混合搜索 |
| `DocumentChunk` | 分块数据结构（含元数据、嵌入向量、相关度分数） |

**检索策略**:
1. 首先尝试 **向量搜索**（cosine similarity），需要 embedding client
2. 回退到 **关键字搜索**（SQLite LIKE，多词 OR 匹配）
3. 按相关度降序排列，返回 top-k 结果

**分块策略**: 段落优先切分，约 500 token/块，128 token 重叠窗口

### 13.4 MCP 协议

**模块**: `pinocchio/mcp/`

| 类 | 职责 |
|---|---|
| `MCPClient` | JSON-RPC 2.0 底层客户端 |
| `MCPToolBridge` | 发现 MCP 工具 → 注册到 Pinocchio ToolRegistry |
| `MCPToolSpec` | 工具规范数据结构 |

**协议实现**:
- HTTP 传输（JSON-RPC 2.0 over POST）
- `tools/list` — 发现工具
- `tools/call` — 调用工具
- 每个远程工具注册为 `mcp_<tool_name>` 前缀

### 13.5 Agent Graph

**模块**: `pinocchio/graph/`

| 类 | 职责 |
|---|---|
| `AgentGraph` | DAG 定义（节点 + 有向边） |
| `GraphNode` | 处理步骤节点（handler, retry, metadata） |
| `GraphEdge` | 有向边（可选条件谓词） |
| `GraphExecutor` | DAG 拓扑排序 → 按层级并行执行 |
| `NodeResult` | 节点执行结果 |

**执行策略**:
- 拓扑排序保证依赖顺序
- 同层节点用 ThreadPoolExecutor 并行执行
- 条件边：仅当 `condition(NodeResult)` 返回 True 时才走该路径
- 节点支持重试（`retry` 参数）

### 13.6 多智能体协作

**模块**: `pinocchio/collaboration/`

| 类 | 职责 |
|---|---|
| `AgentTeam` | 团队管理 + 协调执行 |
| `TeamMember` | 成员定义（role, specialty, handler） |
| `TeamMessage` | 成员间消息 |
| `TeamResult` | 协作结果（含所有贡献） |

**协调模式**:
1. `_decompose()` — LLM 将任务拆分为成员分配
2. 按顺序执行（每个成员接收前序成员的上下文）
3. `_synthesize()` — LLM 将所有贡献合成为最终输出

### 13.7 结构化追踪

**模块**: `pinocchio/tracing/`

| 类 | 职责 |
|---|---|
| `Tracer` | 全局追踪器，管理 Trace 生命周期 |
| `Trace` | 端到端交互记录（含多个 Span） |
| `Span` | 单个操作的计时 + 属性 + 事件 |
| `SpanStatus` | OK / ERROR / SKIPPED |
| `SpanEvent` | Span 内的时间戳事件 |

**集成方式**: 编排器在认知循环的每个阶段自动创建 Span，记录耗时、属性和工具调用事件。

---

## 14. 扩展指南

### 添加新认知技能

1. 在 `unified_agent.py` 中定义系统 prompt 常量
2. 添加技能方法（参考现有方法的模式）
3. 在 `orchestrator.py` 的认知循环中集成
4. 在 `schemas.py` 中定义结果数据类（如需要）

### 添加新模态处理器

1. 创建 `pinocchio/multimodal/new_processor.py`，继承 `BaseAgent`
2. 在 `enums.py` 中注册 `Modality` 和 `AgentRole` 枚举值
3. 在 `LLMClient` 中添加 `build_xxx_message()` 方法
4. 在 `orchestrator.py` 的 `_preprocess_modalities()` 中注册

### 添加新记忆内容轴

1. 创建 `pinocchio/memory/new_memory.py`（参考 `episodic_memory.py`）
2. 在 `schemas.py` 中定义数据类
3. 在 `MemoryManager` 中添加属性和操作方法

### 注册自定义工具

```python
from pinocchio.tools import Tool, ToolRegistry

registry = ToolRegistry()
registry.register_defaults()           # 加载内置三件套
registry.register(Tool(
    name="search_web",
    description="搜索互联网",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "搜索词"}},
        "required": ["query"],
    },
    function=my_search_function,        # Callable[..., str]
))

# 通过 Pinocchio 编排器注入
pinocchio = Pinocchio(tool_registry=registry)
```

### 切换 LLM 后端

修改环境变量即可切换到任何 OpenAI 兼容 API：

```bash
export PINOCCHIO_MODEL="gpt-4o"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OLLAMA_API_KEY="sk-..."
```

---

## 参考

- Kahneman, D. (2011). *Thinking, Fast and Slow* — 快速路径 vs 完整认知循环的设计灵感
- Tulving, E. (1972). *Episodic and Semantic Memory* — 双轴记忆系统的理论基础
- Anderson, J. R. (1993). *Rules of the Mind* — 程序记忆的认知模型
