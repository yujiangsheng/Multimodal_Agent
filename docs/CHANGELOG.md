# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.4.0] — 2025-07

### Added: 可靠性与安全加固 (Reliability & Security Hardening)
- **CircuitBreaker** — LLM 客户端新增熔断器，连续失败时自动熔断避免雪崩（`LLMClient`）
- **InputGuard Unicode 规范化** — 防御 Unicode 混淆攻击（零宽字符、同形异义字等）
- **Tracer TTL** — 追踪记录自动过期清理，防止内存泄漏
- **RAG 路径遍历防护** — `DocumentStore.ingest()` 拒绝 `../` 路径穿越攻击
- **OutputGuard** — 新增输出安全守卫：PII 脱敏、内容策略拦截、幻觉标记

### Added: 线程安全与并发改进
- **Session 切换加锁** — `switch_session()` 全程持有 `_lock`，避免并发状态损坏
- **Team 并行执行线程安全** — `AgentTeam` 并行模式使用细粒度锁保护共享状态
- **Team 并行模式 Review** — 并行执行后同样运行综合审查轮

### Changed: 行为修正
- **Token 计量修正** — `LLMClient.chat()` 仅在最终成功尝试记录 token 用量，重试不重复计数
- **Graph 前驱检查** — `GraphExecutor._execute_node()` 检查前驱节点是否失败，失败则跳过
- **JSON 空响应警告** — `_parse_json_response()` 返回空字典时记录 warning 日志

### Changed: 测试
- 测试用例从 876 → **1042**（36 个测试文件）
- 新增 `test_parametrized.py`（参数化测试覆盖音频格式、枚举遍历等）
- 新增 `test_robustness.py`（LLM 异常场景、极端输入压力测试）
- 拆分 Round 6 测试到对应模块测试文件中

### Fixed
- 修复 `CodeSandbox` 在 macOS 下的进程隔离兼容性问题
- 修复 `ResponseCache` TTL 边界条件下的竞态问题

---

## [0.3.0] — 2025-07

### Added: 扩展子系统 (7 Subsystems)
- **任务规划** (`pinocchio/planning/`) — Plan-and-Solve 多步分解 + ReAct 推理循环
- **代码沙箱** (`pinocchio/sandbox/`) — 隔离子进程安全执行 Python 代码
- **RAG 知识库** (`pinocchio/rag/`) — SQLite 持久化文档分块 + 向量/关键字混合检索
- **MCP 协议** (`pinocchio/mcp/`) — JSON-RPC 2.0 连接外部 MCP 工具服务器
- **Agent Graph** (`pinocchio/graph/`) — DAG 工作流引擎，拓扑排序 + 条件路由 + 并行执行
- **多智能体协作** (`pinocchio/collaboration/`) — 团队协调模式（分解/分配/执行/综合）
- **结构化追踪** (`pinocchio/tracing/`) — OpenTelemetry 风格 Trace/Span 系统

### Added: 工具生态
- 工具数量从 3 → 16（新增 web_fetch, shell_command, file_reader, file_writer, json_query, text_summarizer, regex_search, hash_digest, base64_codec, uuid_generator, env_info, http_request, timestamp_convert）
- web_fetch 内置 SSRF 防护（私有 IP 段拦截）
- shell_command 使用命令白名单 + timeout 保护

### Added: LLM 多供应商
- `PinocchioConfig.from_provider()` 工厂方法
- 8 种预设：Ollama / OpenAI / DeepSeek / Dashscope / Groq / Together / Anthropic / SiliconFlow

### Added: 文档与示例
- `examples/` 目录：8 个独立使用示例
- `docs/CHANGELOG.md`（本文件）
- README 新增故障排查、FAQ、Changelog 章节
- ARCHITECTURE.md 新增 7 个子系统设计文档

### Changed
- 编排器集成所有 7 个子系统（planner, react, sandbox, rag, mcp, graph, team, tracer）
- 测试覆盖从 608 → 876 个用例（33 个测试文件）
- `pyproject.toml` 添加 `project.urls`（Homepage, Repository, Issues, Docs, Changelog）

### Fixed
- `Tool(func=...)` → `Tool(function=...)` 参数名修正
- `ReActExecutor` 构造函数参数顺序修正

---

## [0.2.0] — 2025-03

### Added
- 统一智能体重构：6 个子智能体合并为单一 `PinocchioAgent`
- 流式输出：`chat_stream()` 逐 token 返回 + SSE Web 推送
- 工具调用框架：`ToolRegistry` + `ToolExecutor` + 内置 3 工具
- 向量语义搜索：`EmbeddingClient` + nomic-embed-text
- 响应完整性保障：自动续写 + 启发式检查 + finish_reason 检测
- Prompt 注入防御：`InputGuard` 多层检测
- 上下文管理：`ContextManager` 智能摘要 + token 预算
- 响应缓存：`ResponseCache` 线程安全 LRU + TTL
- 硬件感知并行：`ResourceMonitor` + `ParallelExecutor`
- Web Demo：FastAPI + SSE + 多模态上传

### Changed
- 608 个测试全部通过，>95% 覆盖率

---

## [0.1.0] — 2025-02

### Added
- 初始版本
- 6 阶段认知循环：PERCEIVE → STRATEGIZE → EXECUTE → EVALUATE → LEARN → META-REFLECT
- 双轴记忆系统：内容轴 × 时间轴
- 多模态支持：文本 / 图像 / 音频 / 视频
- CLI 交互入口
- 293 个测试全部通过
