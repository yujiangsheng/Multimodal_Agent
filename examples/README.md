# Pinocchio 使用示例 / Usage Examples

本目录包含 Pinocchio 各子系统的独立使用示例，覆盖核心 API 和全部扩展子系统。

## 示例索引

| 示例文件 | 子系统 | 说明 | LLM 要求 |
|----------|--------|------|----------|
| [basic_chat.py](basic_chat.py) | 核心 | 基础对话、流式输出、多模态输入、状态查看 | 需要 |
| [planning_react.py](planning_react.py) | 规划 | 任务规划与 ReAct 推理循环 | 需要 |
| [sandbox_execution.py](sandbox_execution.py) | 沙箱 | 代码沙箱安全执行（隔离子进程） | 离线可运行 |
| [rag_knowledge.py](rag_knowledge.py) | RAG | 知识库：文档导入、分块、混合检索 | 离线可运行 |
| [agent_graph.py](agent_graph.py) | 图工作流 | DAG 拓扑排序 + 条件路由 + 并行执行 | 离线可运行 |
| [team_collaboration.py](team_collaboration.py) | 协作 | 多智能体团队协调 | 需要 |
| [tracing_observability.py](tracing_observability.py) | 追踪 | Trace/Span 结构化追踪 + JSON 导出 | 离线可运行 |
| [provider_presets.py](provider_presets.py) | 配置 | LLM 多供应商预设一键切换 | 离线可运行 |

## 运行方式

```bash
cd Multimodal_Agent

# 方式一：设置 PYTHONPATH（推荐）
PYTHONPATH=. python3 examples/basic_chat.py

# 方式二：安装为可编辑包
pip install -e .
python3 examples/basic_chat.py
```

> **注意**: 标记"需要"LLM 的示例需要运行 Ollama（`ollama serve && ollama pull qwen3-vl:4b`）。
> 标记"离线可运行"的示例无需 LLM 后端，可直接执行验证 API 和数据结构。

## 快速尝试顺序

如果你是第一次使用 Pinocchio，建议按以下顺序尝试：

1. `provider_presets.py` — 了解配置系统（离线）
2. `sandbox_execution.py` — 体验代码沙箱（离线）
3. `tracing_observability.py` — 了解追踪系统（离线）
4. `basic_chat.py` — 核心对话 API（需要 Ollama）
5. `rag_knowledge.py` → `planning_react.py` → `agent_graph.py` → `team_collaboration.py`
