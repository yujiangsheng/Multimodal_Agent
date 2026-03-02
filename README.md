# 🎭 Pinocchio — 多模态自我进化智能体

[![Author](https://img.shields.io/badge/Author-Jansen%20Yu-blue)](https://github.com/yujansen)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> *"每一次对话都让我变得更好。"*

Pinocchio 是一个具备**持续自我学习与自我改进能力**的多模态智能体系统。它通过结构化的六阶段认知循环，在每次交互后提炼经验、消化教训、优化策略，实现真正的持续进化。

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| **6 阶段认知循环** | PERCEIVE → STRATEGIZE → EXECUTE → EVALUATE → LEARN → META-REFLECT |
| **多模态原生** | 文本 / 图像 / 音频 / 视频，通过 Qwen2.5-Omni 统一处理 |
| **三层记忆系统** | 情景记忆 + 语义记忆 + 程序记忆，JSON 文件持久化，倒排索引加速 |
| **自我进化** | 从每次交互中提取教训，优化策略，积累可复用的程序模板 |
| **元反思** | 周期性高阶反思，检测认知偏差，生成改进计划 |
| **硬件感知并行** | 自动检测 CPU / GPU / RAM，动态调整并行度 |
| **异步支持** | `AsyncLLMClient` + `async_chat()` 适用于高吞吐场景 |

---

## 📁 项目结构

```
Multimodal_Agent/
├── config.py                  # 全局配置（PinocchioConfig 数据类）
├── main.py                    # CLI 交互入口
├── pyproject.toml             # 包管理与工具配置
├── requirements.txt           # 核心依赖
├── pinocchio_system_prompt.md # 系统人设 prompt
│
├── pinocchio/                 # 核心包
│   ├── __init__.py
│   ├── orchestrator.py        # 顶层编排器 — 驱动认知循环
│   │
│   ├── agents/                # 6 个认知循环子智能体
│   │   ├── base_agent.py          # 抽象基类（共享 LLM / 记忆 / 日志）
│   │   ├── perception_agent.py    # Phase 1: 感知 — 输入分析与分类
│   │   ├── strategy_agent.py      # Phase 2: 策略 — 方案选择与风险评估
│   │   ├── execution_agent.py     # Phase 3: 执行 — 生成用户响应
│   │   ├── evaluation_agent.py    # Phase 4: 评估 — 质量评分与问题识别
│   │   ├── learning_agent.py      # Phase 5: 学习 — 提取教训、更新记忆
│   │   └── meta_reflection_agent.py # Phase 6: 元反思 — 高阶自我分析
│   │
│   ├── memory/                # 三层记忆系统
│   │   ├── memory_manager.py      # 统一门面（facade）
│   │   ├── episodic_memory.py     # 情景记忆 — 交互历史
│   │   ├── semantic_memory.py     # 语义记忆 — 蒸馏知识
│   │   └── procedural_memory.py   # 程序记忆 — 可复用策略
│   │
│   ├── models/                # 数据模型
│   │   ├── enums.py               # 枚举（Modality, TaskType, AgentRole...）
│   │   └── schemas.py             # 数据类（记忆记录、认知结果、消息）
│   │
│   ├── multimodal/            # 模态处理器
│   │   ├── text_processor.py      # 文本理解 & 生成
│   │   ├── vision_processor.py    # 图像理解 & VQA & OCR
│   │   ├── audio_processor.py     # 音频转写 & 情感分析
│   │   └── video_processor.py     # 视频理解（原生 + ffmpeg 回退）
│   │
│   └── utils/                 # 基础工具
│       ├── llm_client.py          # LLM API 封装（同步 + 异步）
│       ├── logger.py              # 彩色结构化日志
│       ├── resource_monitor.py    # CPU / GPU / RAM 检测
│       └── parallel_executor.py   # 资源感知并行执行器
│
├── scripts/                   # 辅助脚本
│   └── generate_demo_video.py     # 演示视频生成器
│
└── tests/                     # 测试套件（390+ 测试，99% 覆盖率）
    ├── conftest.py
    ├── test_agents.py
    ├── test_cognitive_loop.py
    ├── test_integration.py
    ├── test_memory.py
    ├── test_memory_edge.py
    ├── test_models.py
    ├── test_multimodal.py
    ├── test_orchestrator_deep.py
    ├── test_resource_parallel.py
    ├── test_stress_integration.py
    └── test_utils.py
```

---

## 🚀 快速开始

### 1. 安装

```bash
# 克隆项目
git clone https://github.com/yujansen/Multimodal_Agent.git
cd Multimodal_Agent

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装（可编辑模式，含开发依赖）
pip install -e ".[dev]"
```

### 2. 启动 Ollama 后端

```bash
# 安装 Ollama（macOS）
brew install ollama

# 拉取 Qwen2.5-Omni 模型
ollama pull qwen2.5-omni

# 启动服务（默认监听 localhost:11434）
ollama serve
```

### 3. 运行

```bash
python main.py
```

交互示例：

```
🧑 You: 请用简单的语言解释量子纠缠

── Phase 1: PERCEIVE 感知 ──────────────────
[PERCEPTION] Task type: question_answering | Complexity: 3 | Confidence: high

── Phase 2: STRATEGIZE 策略 ────────────────
[STRATEGY] Strategy: structured_explanation | Novel: false

── Phase 3: EXECUTE 执行 ────────────────────
[EXECUTION] Executing strategy: structured_explanation

── Phase 4: EVALUATE 评估 ──────────────────
[EVALUATION] Quality: 8/10 | Strategy: 8/10 | Status: complete

── Phase 5: LEARN 学习 ──────────────────────
[LEARNING] Stored episode abc12345 (score: 8/10)

🤖 Pinocchio: 量子纠缠就像是两个粒子之间的一种"超距关联"...
```

---

## 📖 使用方式

### Python API

```python
from pinocchio import Pinocchio

# 使用默认配置
agent = Pinocchio()

# 纯文本对话
response = agent.chat("用 Python 写一个快速排序")
print(response)

# 图片理解
response = agent.chat(
    "这张图片里有什么？",
    image_paths=["photo.jpg"],
)

# 音频转写
response = agent.chat(
    "总结这段会议录音的要点",
    audio_paths=["meeting.wav"],
)

# 视频分析
response = agent.chat(
    "描述这个视频的主要内容",
    video_paths=["lecture.mp4"],
)

# 多模态混合输入
response = agent.chat(
    "对比这张图片和这段视频的内容",
    image_paths=["photo.jpg"],
    video_paths=["clip.mp4"],
)

# 查看智能体状态
import json
print(json.dumps(agent.status(), ensure_ascii=False, indent=2))
```

### 异步调用

```python
import asyncio
from pinocchio import Pinocchio

agent = Pinocchio()

async def main():
    response = await agent.async_chat("Hello, Pinocchio!")
    print(response)

asyncio.run(main())
```

### 自定义配置

```python
from config import PinocchioConfig
from pinocchio import Pinocchio

cfg = PinocchioConfig()

agent = Pinocchio(
    model=cfg.model,
    api_key=cfg.api_key,
    base_url=cfg.base_url,
    data_dir=cfg.data_dir,
    verbose=cfg.verbose,
    max_workers=cfg.max_workers,
    parallel_modalities=cfg.parallel_modalities,
    meta_reflect_interval=cfg.meta_reflect_interval,
)
```

### 环境变量

```bash
export PINOCCHIO_MODEL="qwen2.5-omni"
export OPENAI_BASE_URL="http://localhost:11434/v1"
export PINOCCHIO_DATA_DIR="./my_data"
export PINOCCHIO_MAX_WORKERS=4
export PINOCCHIO_PARALLEL=true
```

---

## 🧠 架构设计

### 认知循环

```
用户输入  ──►  ┌──────────┐   ┌──────────┐   ┌──────────┐
               │ PERCEIVE │──►│STRATEGIZE│──►│ EXECUTE  │
               │ 感知分析 │   │ 策略选择 │   │ 生成响应 │
               └──────────┘   └──────────┘   └────┬─────┘
                                                   │
用户输出  ◄──  ┌────────────┐  ┌──────────┐  ┌────▼─────┐
               │META-REFLECT│◄─│  LEARN   │◄─│ EVALUATE │
               │元反思(周期) │  │ 提取教训 │  │ 质量评估 │
               └────────────┘  └──────────┘  └──────────┘
```

每个阶段由独立的子智能体执行，通过 Orchestrator 统一调度。所有阶段共享同一个 LLM 实例和三层记忆系统。

### 三层记忆系统

| 记忆类型 | 人类类比 | 存储内容 | 索引方式 | 持久化 |
|----------|----------|----------|----------|--------|
| **情景记忆** | "我记得那次…" | 交互痕迹（意图、策略、分数、教训） | task_type / modality 倒排索引 | JSON |
| **语义记忆** | "我知道…" | 从多次经验蒸馏的通用知识 | domain 倒排索引 | JSON |
| **程序记忆** | "我会…" | 可复用的策略模板（步骤、成功率） | task_type 倒排索引 | JSON |

**知识合成机制**：当某个领域的情景记忆达到阈值（默认 10 条），系统会标记该领域进行知识蒸馏，由 LearningAgent 将具体经验提炼为通用语义知识。

### 多模态处理

```
             ┌── TextProcessor ──┐
             │                   │
用户输入 ──► ├── VisionProcessor ├──► 融合上下文 ──► ExecutionAgent
             │                   │
             ├── AudioProcessor ──┤
             │                   │
             └── VideoProcessor ──┘
```

- **并行模式**：多线程并发处理各模态（`ThreadPoolExecutor`）
- **顺序回退**：单核环境自动降级为顺序处理
- **融合策略**：`early_fusion` / `late_fusion` / `hybrid_fusion`
- **视频处理**：优先原生 Qwen2.5-Omni 视频输入，回退至 ffmpeg 抽帧 + 分析

---

## 🧪 测试

```bash
# 运行全部测试
pytest

# 带覆盖率报告
pytest --cov=pinocchio --cov-report=term-missing

# 仅运行快速测试
pytest -m "not slow"

# 指定测试文件
pytest tests/test_agents.py -v
```

当前状态：**390+ 测试，99% 代码覆盖率**

---

## ⚙️ 配置参考

| 参数 | 类型 | 默认值 | 环境变量 | 说明 |
|------|------|--------|----------|------|
| `model` | `str` | `qwen2.5-omni` | `PINOCCHIO_MODEL` | LLM 模型名称 |
| `api_key` | `str` | `ollama` | `OLLAMA_API_KEY` | API 密钥 |
| `base_url` | `str` | `http://localhost:11434/v1` | `OPENAI_BASE_URL` | API 基础 URL |
| `temperature` | `float` | `0.7` | — | 生成温度 |
| `max_tokens` | `int` | `4096` | — | 最大生成 tokens |
| `data_dir` | `str` | `data` | `PINOCCHIO_DATA_DIR` | 记忆持久化目录 |
| `meta_reflect_interval` | `int` | `5` | — | 元反思触发间隔 |
| `max_workers` | `int\|None` | auto | `PINOCCHIO_MAX_WORKERS` | 最大并行线程数 |
| `parallel_modalities` | `bool` | `True` | `PINOCCHIO_PARALLEL` | 并行处理多模态 |

---

## 🛠️ 开发指南

### 代码风格

项目使用 [Ruff](https://docs.astral.sh/ruff/) 进行格式化和 lint：

```bash
pip install ruff
ruff check pinocchio/
ruff format pinocchio/
```

### 添加新的子智能体

1. 在 `pinocchio/agents/` 下创建新文件，继承 `BaseAgent`
2. 设置 `role = AgentRole.YOUR_ROLE`（需先在 `enums.py` 注册）
3. 实现 `run(**kwargs)` 方法
4. 在 `pinocchio/agents/__init__.py` 中注册导出
5. 在 `orchestrator.py` 的认知循环中集成调用
6. 编写对应的测试用例

### 添加新的模态处理器

1. 在 `pinocchio/multimodal/` 下创建新文件，继承 `BaseAgent`
2. 在 `enums.py` 中添加新的 `Modality` 和 `AgentRole` 枚举值
3. 在 `LLMClient` 中添加 `build_xxx_message()` 构建方法
4. 在 `orchestrator.py` 的 `_preprocess_modalities()` 中注册
5. 在 `pinocchio/multimodal/__init__.py` 中导出

---

## 👤 作者与维护者

**Jansen Yu** — [@yujansen](https://github.com/yujansen)

## 📄 许可证

[MIT License](LICENSE)
