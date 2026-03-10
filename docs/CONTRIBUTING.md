# 贡献指南

感谢你对 Pinocchio 项目的关注！欢迎以任何形式参与贡献。

---

## 开发环境搭建

### 1. 克隆并安装

```bash
git clone https://github.com/yujansen/Multimodal_Agent.git
cd Multimodal_Agent

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装（含开发依赖）
pip install -e ".[dev]"

# 安装 Web 依赖（可选）
pip install -e ".[web]"
```

### 2. 启动 Ollama（运行集成测试时需要）

```bash
ollama serve
ollama pull qwen3-vl:4b
```

> **注意：** 单元测试全部使用 mock，不需要运行 Ollama。

---

## 代码风格

项目使用 [Ruff](https://docs.astral.sh/ruff/) 进行 lint 和格式化：

```bash
# 检查
ruff check pinocchio/ tests/

# 自动修复
ruff check --fix pinocchio/ tests/

# 格式化
ruff format pinocchio/ tests/
```

### 代码规范

- **类型注解**：使用 PEP 604 语法（`str | None`），所有文件头部添加 `from __future__ import annotations`
- **Docstring**：所有公开类和方法必须有 docstring（Google 或 NumPy 风格）
- **命名**：类名 PascalCase，方法名 snake_case，常量 UPPER_SNAKE_CASE
- **私有方法**：以 `_` 前缀标记内部方法

---

## 测试

### 运行测试

```bash
# 全部测试
pytest

# 带覆盖率
pytest --cov=pinocchio --cov-report=term-missing

# 指定文件
pytest tests/test_agents.py -v

# 仅匹配特定测试名
pytest -k "test_perceive"
```

### 编写测试

- 所有测试文件放在 `tests/` 目录下，文件名以 `test_` 开头
- 使用 `conftest.py` 中的共享 fixture（`mock_llm`, `mock_memory`, `mock_logger`）
- **永远不调用真实 LLM** — 所有 LLM 调用必须 mock
- 异步测试使用 `pytest-asyncio`（已配置 `asyncio_mode="auto"`）

### 测试结构

```
tests/
├── conftest.py              # 共享 fixtures
├── test_agents.py           # 统一智能体 6 种技能
├── test_cognitive_loop.py   # 认知循环边界条件
├── test_integration.py      # 全链路集成
├── test_memory*.py          # 记忆系统 (正常 + 边界)
├── test_multimodal.py       # 模态处理器
├── test_streaming.py        # 流式输出 chat_stream
├── test_tools.py            # 工具注册 / 执行 / 沙箱
├── test_embedding.py        # 向量嵌入 + 语义搜索
├── test_input_guard.py      # 输入安全 (注入检测、消毒)
├── test_context_manager.py  # 上下文窗口管理
├── test_response_cache.py   # 响应缓存 (LRU + TTL)
├── test_orchestrator_deep.py# 编排器深度测试
├── test_stress_integration.py# 压力 / 跨切面
└── ...                      # 其余按模块命名
```

> 当前共 **26 个测试文件、608 个测试用例**，全部 100% 通过。

---

## 项目结构

```
pinocchio/
├── agents/
│   ├── base_agent.py       # 抽象基类
│   └── unified_agent.py    # 统一认知智能体 (6 种技能)
├── memory/                 # 双轴记忆系统
├── models/                 # 枚举 + 数据类
├── multimodal/             # 模态处理器
├── tools.py                # 工具注册 + 安全执行
└── utils/
    ├── llm_client.py       # LLMClient + EmbeddingClient
    ├── logger.py            # 彩色日志 (自动 TTY 检测)
    ├── input_guard.py       # 输入安全 (注入检测)
    ├── context_manager.py   # 上下文窗口管理
    ├── response_cache.py    # 响应缓存 (LRU + TTL)
    ├── parallel_executor.py # 并行执行器
    └── resource_monitor.py  # 资源监控
```

详细架构参见 [ARCHITECTURE.md](ARCHITECTURE.md)。

---

## 提交规范

### Commit 消息格式

```
<type>: <简短描述>

<详细说明>（可选）
```

**type 类型：**

| Type | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 文档更新 |
| `refactor` | 重构（不改变行为） |
| `test` | 测试相关 |
| `perf` | 性能优化 |
| `chore` | 构建/工具链变更 |

### 示例

```
feat: add audio emotion detection to AudioProcessor

Add AudioProcessor.detect_emotion() method that uses Qwen3-VL's
native audio understanding to classify speaker emotions.
```

---

## Pull Request 流程

1. **Fork** 并创建分支：`git checkout -b feat/my-feature`
2. **编写代码** + **测试**
3. **确保全部测试通过**：`pytest`
4. **检查代码风格**：`ruff check pinocchio/ tests/`
5. **提交** 并推送到你的 fork
6. **创建 PR**，描述更改内容和动机

### PR 检查清单

- [ ] 所有测试通过
- [ ] 新功能有对应测试
- [ ] 代码通过 ruff 检查
- [ ] 公开 API 有 docstring
- [ ] 更新 README.md（如果有用户可见的变化）

---

## 常见开发任务

### 添加新技能（认知阶段）

参见 [ARCHITECTURE.md #添加新认知技能](ARCHITECTURE.md#添加新认知技能)

### 添加新模态处理器

参见 [ARCHITECTURE.md #添加新模态处理器](ARCHITECTURE.md#添加新模态处理器)

### 调试认知循环

启用 verbose 模式查看每个阶段的输入输出：

```python
agent = Pinocchio(verbose=True)
agent.chat("你的问题")
```

日志输出会用不同颜色标记每个阶段：
- 🟦 ORCHESTRATOR（青色）
- 🟨 PERCEPTION（黄色）
- 🟪 STRATEGY（品红）
- 🟩 EXECUTION（绿色）
- 🟦 EVALUATION（蓝色）
- 🟨 LEARNING（粗体黄色）

---

## 联系方式

如有问题或建议，欢迎通过 [GitHub Issues](https://github.com/yujansen/Multimodal_Agent/issues) 提出。
