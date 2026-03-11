# Pinocchio 使用示例 / Usage Examples

本目录包含 Pinocchio 各子系统的独立使用示例。

| 示例文件 | 说明 |
|----------|------|
| [basic_chat.py](basic_chat.py) | 基础对话、流式输出、多模态输入 |
| [planning_react.py](planning_react.py) | 任务规划与 ReAct 推理循环 |
| [sandbox_execution.py](sandbox_execution.py) | 代码沙箱安全执行 |
| [rag_knowledge.py](rag_knowledge.py) | RAG 知识库：文档导入与检索 |
| [agent_graph.py](agent_graph.py) | DAG 工作流引擎 |
| [team_collaboration.py](team_collaboration.py) | 多智能体协作 |
| [tracing_observability.py](tracing_observability.py) | 结构化追踪与可观测性 |
| [provider_presets.py](provider_presets.py) | LLM 多供应商预设切换 |

## 运行方式

```bash
cd Multimodal_Agent

# 方式一：设置 PYTHONPATH（推荐）
PYTHONPATH=. python3 examples/basic_chat.py

# 方式二：安装为可编辑包
pip install -e .
python3 examples/basic_chat.py
```

> **注意**: 大多数示例需要运行 Ollama（`ollama serve`）。
> 标记为 `# 离线可运行` 的示例无需 LLM 后端。
