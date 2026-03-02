#!/usr/bin/env python3
"""Generate a multimodal agent DEMO video.

This script:
1. Creates synthetic test assets (images, text files)
2. Runs the Pinocchio agent through real multimodal scenarios
3. Records each interaction and renders an animated terminal-style demo video

Requires: Pillow, opencv-python-headless, Ollama running with qwen3-vl:8b
Produces: multimodal_demo.mp4
"""

from __future__ import annotations

import json
import math
import os
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Add project to path ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Video constants ──────────────────────────────────────────
WIDTH, HEIGHT = 1920, 1080
FPS = 30

# Catppuccin Mocha palette
BG           = (30, 30, 46)
SURFACE      = (49, 50, 68)
TEXT         = (205, 214, 244)
GREEN        = (166, 227, 161)
RED          = (243, 139, 168)
YELLOW       = (249, 226, 175)
BLUE         = (137, 180, 250)
LAVENDER     = (180, 190, 254)
MAUVE        = (203, 166, 247)
PEACH        = (250, 179, 135)
DIM          = (108, 112, 134)
TEAL         = (148, 226, 213)

MARGIN       = 40
LINE_HEIGHT  = 28
FONT_SIZE    = 20
H1_SIZE      = 44
H2_SIZE      = 28
H3_SIZE      = 22

OUTPUT_PATH  = ROOT / "multimodal_demo.mp4"

# ── Font helper ──────────────────────────────────────────────
# We need TWO font families:
#   1. A CJK-capable font for ALL user-facing text (Chinese + ASCII both render fine)
#   2. Keep a monospace font only for purely-ASCII code blocks if desired
#
# STHeiti Medium / Hiragino Sans GB cover CJK + Latin + emoji fallback.
# They are proportional but look great at terminal-style sizes.

_font_cache: dict[tuple[int, bool, bool], ImageFont.FreeTypeFont] = {}


def get_font(size: int = FONT_SIZE, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    """Return a font that supports Chinese characters.

    Parameters
    ----------
    size  : font size in pixels
    bold  : prefer bold weight
    mono  : prefer monospace (ASCII-only, for code blocks)
    """
    key = (size, bold, mono)
    if key in _font_cache:
        return _font_cache[key]

    # ── CJK-capable candidates (preferred for everything with Chinese) ──
    cjk_candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        # Linux fallbacks
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    cjk_bold_candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ]

    # ── Monospace candidates (ASCII-only, for code rendering) ──
    mono_candidates = [
        "/System/Library/Fonts/SFMono-Regular.otf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.dfont",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]

    if mono:
        candidates = mono_candidates
    elif bold:
        candidates = cjk_bold_candidates + cjk_candidates
    else:
        candidates = cjk_candidates

    for p in candidates:
        if Path(p).exists():
            try:
                f = ImageFont.truetype(p, size)
                _font_cache[key] = f
                return f
            except Exception:
                continue

    # Last resort
    f = ImageFont.load_default()
    _font_cache[key] = f
    return f


def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def make_frame() -> tuple[Image.Image, ImageDraw.Draw]:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    return img, ImageDraw.Draw(img)


def wrap_text(text: str, max_chars: int = 90) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        wrapped = textwrap.wrap(paragraph, width=max_chars) or [""]
        lines.extend(wrapped)
    return lines


# ── Asset generation ─────────────────────────────────────────

ASSETS_DIR = ROOT / "demo_assets"


def create_test_image_chart() -> str:
    """Generate a synthetic bar-chart image for vision testing."""
    path = ASSETS_DIR / "sales_chart.png"
    img = Image.new("RGB", (800, 600), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = get_font(18)
    font_big = get_font(24, bold=True)

    # Title
    draw.text((250, 20), "Quarterly Sales Report 2025", fill=(30, 30, 30), font=font_big)

    # Bars
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    values = [150, 280, 210, 340]
    colors = [(66, 133, 244), (52, 168, 83), (251, 188, 4), (234, 67, 53)]
    bar_w, max_h, base_y = 100, 350, 530

    for i, (q, v, c) in enumerate(zip(quarters, values, colors)):
        x = 150 + i * 160
        h = int(v / max(values) * max_h)
        draw.rectangle([x, base_y - h, x + bar_w, base_y], fill=c)
        draw.text((x + 30, base_y - h - 25), f"${v}K", fill=(30, 30, 30), font=font)
        draw.text((x + 35, base_y + 10), q, fill=(80, 80, 80), font=font)

    # Y-axis label
    draw.text((30, 250), "Revenue ($K)", fill=(80, 80, 80), font=font)
    img.save(path)
    return str(path)


def create_test_image_scene() -> str:
    """Generate a synthetic scene image with shapes for vision testing."""
    path = ASSETS_DIR / "scene.png"
    img = Image.new("RGB", (800, 600), (135, 206, 235))  # sky blue
    draw = ImageDraw.Draw(img)

    # Ground
    draw.rectangle([0, 400, 800, 600], fill=(34, 139, 34))

    # Sun
    draw.ellipse([620, 40, 730, 150], fill=(255, 215, 0))

    # House
    draw.rectangle([200, 280, 400, 450], fill=(139, 69, 19))  # wall
    draw.polygon([(180, 280), (420, 280), (300, 180)], fill=(178, 34, 34))  # roof
    draw.rectangle([270, 340, 330, 450], fill=(101, 67, 33))  # door
    draw.rectangle([220, 310, 260, 350], fill=(173, 216, 230))  # window
    draw.rectangle([340, 310, 380, 350], fill=(173, 216, 230))  # window

    # Tree
    draw.rectangle([550, 300, 575, 420], fill=(101, 67, 33))
    draw.ellipse([510, 200, 620, 320], fill=(0, 100, 0))

    # Text in image
    font = get_font(16)
    draw.text((20, 560), "Synthetic Scene for VQA Testing", fill=(255, 255, 255), font=font)

    img.save(path)
    return str(path)


def create_test_image_code() -> str:
    """Generate an image of code (OCR test)."""
    path = ASSETS_DIR / "code_screenshot.png"
    img = Image.new("RGB", (800, 500), (40, 42, 54))
    draw = ImageDraw.Draw(img)
    font = get_font(16)

    code_lines = [
        ("def fibonacci(n):", BLUE),
        ('    """Return the n-th Fibonacci number."""', GREEN),
        ("    if n <= 1:", TEXT),
        ("        return n", TEXT),
        ("    return fibonacci(n - 1) + fibonacci(n - 2)", TEXT),
        ("", TEXT),
        ("# Test", DIM),
        ("for i in range(10):", MAUVE),
        ('    print(f"F({i}) = {fibonacci(i)}")', YELLOW),
    ]
    y = 30
    for line, color in code_lines:
        draw.text((30, y), line, fill=color, font=font)
        y += 28

    img.save(path)
    return str(path)


# -- Text test assets ---
def create_text_assets() -> dict[str, str]:
    """Return text prompts for different text-processing demo scenarios."""
    return {
        "summarization": (
            "人工智能（AI）正在加速改变全球各行各业。在医疗领域，AI辅助诊断系统能够在几秒钟内分析"
            "医学影像，准确率已经超越人类放射科医生。在自动驾驶领域，特斯拉、Waymo等公司的无人驾驶"
            "汽车已经在多个城市上路测试。在教育领域，个性化AI导师可以根据每个学生的学习进度和风格"
            "调整教学内容。然而，AI的快速发展也引发了关于就业、隐私和伦理的广泛讨论。许多专家呼吁"
            "建立更完善的AI监管框架，以确保技术的发展能够造福全人类。"
        ),
        "translation": "The quick brown fox jumps over the lazy dog. This sentence is a pangram — it contains every letter of the English alphabet at least once.",
        "code_generation": "写一个Python函数，实现二分查找算法，要求有完整的docstring和类型注解。",
    }


# ── Demo scenario dataclass ─────────────────────────────────

@dataclass
class DemoScene:
    title: str
    icon: str
    modality: str  # "text" | "vision" | "audio" | "video" | "multi"
    description: str
    user_input: str
    chat_kwargs: dict[str, Any] = field(default_factory=dict)
    agent_response: str = ""
    elapsed: float = 0.0
    thumbnail: Image.Image | None = None


# ── Run actual agent interactions ────────────────────────────

def run_demo_scenarios() -> list[DemoScene]:
    """Run through all demo scenarios, calling the real Pinocchio agent."""
    from pinocchio import Pinocchio

    # Use qwen3-vl for vision, it handles text well too
    MODEL = "qwen3-vl:8b"
    print(f"🤖 Initialising Pinocchio with model: {MODEL}")
    agent = Pinocchio(
        model=MODEL,
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        data_dir=str(ROOT / "demo_data"),
        verbose=True,
    )

    # ── Create assets ──
    print("🎨 Generating test assets …")
    chart_path = create_test_image_chart()
    scene_path = create_test_image_scene()
    code_img_path = create_test_image_code()
    texts = create_text_assets()

    scenes: list[DemoScene] = []

    # ── Scene 1: Text Summarisation ──
    print("\n📝 Scene 1: Text Summarization")
    s1 = DemoScene(
        title="文本理解 — 自动摘要",
        icon="📝",
        modality="text",
        description="将长篇中文段落精炼为要点摘要",
        user_input=f"请用三个要点总结以下内容：\n{texts['summarization']}",
    )
    t0 = time.time()
    try:
        s1.agent_response = agent.chat(s1.user_input)
    except Exception as e:
        s1.agent_response = f"[Error: {e}]"
        traceback.print_exc()
    s1.elapsed = time.time() - t0
    scenes.append(s1)
    print(f"   ✅ Done in {s1.elapsed:.1f}s")

    # ── Scene 2: Translation ──
    print("\n🌐 Scene 2: Translation")
    s2 = DemoScene(
        title="文本处理 — 翻译",
        icon="🌐",
        modality="text",
        description="英译中，保持语言风格和修辞",
        user_input=f"请将以下英文翻译成中文，保持语言流畅：\n{texts['translation']}",
    )
    t0 = time.time()
    try:
        s2.agent_response = agent.chat(s2.user_input)
    except Exception as e:
        s2.agent_response = f"[Error: {e}]"
    s2.elapsed = time.time() - t0
    scenes.append(s2)
    print(f"   ✅ Done in {s2.elapsed:.1f}s")

    # ── Scene 3: Code Generation ──
    print("\n💻 Scene 3: Code Generation")
    s3 = DemoScene(
        title="代码生成 — 二分查找",
        icon="💻",
        modality="text",
        description="根据自然语言描述生成带注解的Python代码",
        user_input=texts["code_generation"],
    )
    t0 = time.time()
    try:
        s3.agent_response = agent.chat(s3.user_input)
    except Exception as e:
        s3.agent_response = f"[Error: {e}]"
    s3.elapsed = time.time() - t0
    scenes.append(s3)
    print(f"   ✅ Done in {s3.elapsed:.1f}s")

    # ── Scene 4: Image Understanding (Chart) ──
    print("\n📊 Scene 4: Vision — Chart Analysis")
    s4 = DemoScene(
        title="图像理解 — 图表分析",
        icon="📊",
        modality="vision",
        description="分析柱状图并提取数据与趋势",
        user_input="请分析这张销售报表图表，描述数据趋势和关键发现。",
        chat_kwargs={"image_paths": [chart_path]},
    )
    s4.thumbnail = Image.open(chart_path).copy()
    t0 = time.time()
    try:
        s4.agent_response = agent.chat(s4.user_input, **s4.chat_kwargs)
    except Exception as e:
        s4.agent_response = f"[Error: {e}]"
        traceback.print_exc()
    s4.elapsed = time.time() - t0
    scenes.append(s4)
    print(f"   ✅ Done in {s4.elapsed:.1f}s")

    # ── Scene 5: Image Understanding (Scene VQA) ──
    print("\n🏠 Scene 5: Vision — Scene Understanding")
    s5 = DemoScene(
        title="图像理解 — 场景问答",
        icon="🏠",
        modality="vision",
        description="回答关于场景图像的自然语言问题",
        user_input="这张图片里有什么？请详细描述场景中的每个元素。",
        chat_kwargs={"image_paths": [scene_path]},
    )
    s5.thumbnail = Image.open(scene_path).copy()
    t0 = time.time()
    try:
        s5.agent_response = agent.chat(s5.user_input, **s5.chat_kwargs)
    except Exception as e:
        s5.agent_response = f"[Error: {e}]"
    s5.elapsed = time.time() - t0
    scenes.append(s5)
    print(f"   ✅ Done in {s5.elapsed:.1f}s")

    # ── Scene 6: OCR — Code Screenshot ──
    print("\n🔍 Scene 6: Vision — OCR Code Extraction")
    s6 = DemoScene(
        title="OCR识别 — 代码截图",
        icon="🔍",
        modality="vision",
        description="从代码截图中提取文字并解释代码逻辑",
        user_input="请识别这张截图中的代码，并解释它的功能。",
        chat_kwargs={"image_paths": [code_img_path]},
    )
    s6.thumbnail = Image.open(code_img_path).copy()
    t0 = time.time()
    try:
        s6.agent_response = agent.chat(s6.user_input, **s6.chat_kwargs)
    except Exception as e:
        s6.agent_response = f"[Error: {e}]"
    s6.elapsed = time.time() - t0
    scenes.append(s6)
    print(f"   ✅ Done in {s6.elapsed:.1f}s")

    # ── Scene 7: Multi-turn Conversation (Memory) ──
    print("\n🧠 Scene 7: Multi-turn Memory")
    s7 = DemoScene(
        title="多轮对话 — 记忆持续性",
        icon="🧠",
        modality="text",
        description="测试智能体跨轮次记忆与上下文理解",
        user_input="我叫小明，我是一名数据科学家，最近在研究大语言模型。你能记住这些信息吗？",
    )
    t0 = time.time()
    try:
        s7.agent_response = agent.chat(s7.user_input)
    except Exception as e:
        s7.agent_response = f"[Error: {e}]"
    s7.elapsed = time.time() - t0
    scenes.append(s7)
    print(f"   ✅ Done in {s7.elapsed:.1f}s")

    # Follow-up that tests memory
    print("\n🧠 Scene 7b: Memory Recall")
    s7b = DemoScene(
        title="多轮对话 — 记忆回忆",
        icon="🧠",
        modality="text",
        description="验证智能体能否回忆之前的对话内容",
        user_input="请问我叫什么名字？我的职业是什么？我在研究什么？",
    )
    t0 = time.time()
    try:
        s7b.agent_response = agent.chat(s7b.user_input)
    except Exception as e:
        s7b.agent_response = f"[Error: {e}]"
    s7b.elapsed = time.time() - t0
    scenes.append(s7b)
    print(f"   ✅ Done in {s7b.elapsed:.1f}s")

    # ── Scene 8: Agent Self-Reflection ──
    print("\n🪞 Scene 8: Self-Reflection Status")
    s8 = DemoScene(
        title="自我反思 — 状态报告",
        icon="🪞",
        modality="text",
        description="查看认知循环的自我评估与改进方向",
        user_input="",
    )
    t0 = time.time()
    try:
        status = agent.status()
        s8.agent_response = json.dumps(status, ensure_ascii=False, indent=2)
    except Exception as e:
        s8.agent_response = f"[Error: {e}]"
    s8.elapsed = time.time() - t0
    scenes.append(s8)
    print(f"   ✅ Done in {s8.elapsed:.1f}s")

    return scenes


# ── Cache helpers ────────────────────────────────────────────
CACHE_PATH = ROOT / "demo_cache.json"


def save_cache(scenes: list[DemoScene]) -> None:
    """Persist scene responses to JSON so we can re-render without re-running LLM."""
    data = []
    for s in scenes:
        data.append({
            "title": s.title,
            "icon": s.icon,
            "modality": s.modality,
            "description": s.description,
            "user_input": s.user_input,
            "agent_response": s.agent_response,
            "elapsed": s.elapsed,
            "has_thumbnail": s.thumbnail is not None,
        })
    CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 Cache saved to {CACHE_PATH}")


def load_cache() -> list[DemoScene] | None:
    """Load cached scene data if available."""
    if not CACHE_PATH.exists():
        return None
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        # Re-create assets for thumbnails
        chart_path = create_test_image_chart()
        scene_path = create_test_image_scene()
        code_img_path = create_test_image_code()
        thumb_map = {
            "图像理解 — 图表分析": chart_path,
            "图像理解 — 场景问答": scene_path,
            "OCR识别 — 代码截图": code_img_path,
        }

        scenes: list[DemoScene] = []
        for d in data:
            s = DemoScene(
                title=d["title"],
                icon=d["icon"],
                modality=d["modality"],
                description=d["description"],
                user_input=d["user_input"],
                agent_response=d["agent_response"],
                elapsed=d["elapsed"],
            )
            if d.get("has_thumbnail") and s.title in thumb_map:
                s.thumbnail = Image.open(thumb_map[s.title]).copy()
            scenes.append(s)
        print(f"📦 Loaded {len(scenes)} scenes from cache")
        return scenes
    except Exception as e:
        print(f"⚠️  Cache load failed: {e}")
        return None


# ── Video rendering ──────────────────────────────────────────

class VideoRenderer:
    """Render demo scenes into an MP4 video."""

    def __init__(self) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, FPS, (WIDTH, HEIGHT))
        assert self.writer.isOpened(), "Failed to open VideoWriter"

    def close(self) -> None:
        self.writer.release()

    def _write_frame(self, img: Image.Image) -> None:
        self.writer.write(pil_to_cv(img))

    def _hold(self, img: Image.Image, seconds: float) -> None:
        for _ in range(int(FPS * seconds)):
            self._write_frame(img)

    # ── Title screen ──

    def render_title(self, duration: float = 4.0) -> None:
        font_h1 = get_font(H1_SIZE, bold=True)
        font_h2 = get_font(H2_SIZE)
        font_sm = get_font(FONT_SIZE)

        for i in range(int(FPS * duration)):
            img, draw = make_frame()
            a = min(1.0, i / (FPS * 0.8))

            title = "🤖 Pinocchio 多模态智能体演示"
            c1 = tuple(int(v * a) for v in BLUE)
            bx = draw.textbbox((0, 0), title, font=font_h1)
            draw.text(((WIDTH - bx[2] + bx[0]) // 2, HEIGHT // 2 - 120), title, fill=c1, font=font_h1)

            sub = "Multimodal Self-Evolving Agent  ·  Live Demo with qwen3-vl:8b"
            c2 = tuple(int(v * a) for v in TEXT)
            bx2 = draw.textbbox((0, 0), sub, font=font_h2)
            draw.text(((WIDTH - bx2[2] + bx2[0]) // 2, HEIGHT // 2 - 50), sub, fill=c2, font=font_h2)

            features = [
                "📝 文本理解与生成    📊 图表与图像分析    🔍 OCR代码识别",
                "🧠 多轮记忆持续性    🪞 六阶段认知循环    ⚡ 并行多模态处理",
            ]
            c3 = tuple(int(v * a) for v in LAVENDER)
            for j, feat in enumerate(features):
                bx3 = draw.textbbox((0, 0), feat, font=font_sm)
                draw.text(((WIDTH - bx3[2] + bx3[0]) // 2, HEIGHT // 2 + 30 + j * 36), feat, fill=c3, font=font_sm)

            date_str = "2026-03-01  ·  Powered by Ollama + Pinocchio Framework"
            c4 = tuple(int(v * a) for v in DIM)
            bx4 = draw.textbbox((0, 0), date_str, font=font_sm)
            draw.text(((WIDTH - bx4[2] + bx4[0]) // 2, HEIGHT - 80), date_str, fill=c4, font=font_sm)

            self._write_frame(img)

    # ── Scene transition ──

    def render_transition(self, scene: DemoScene, scene_num: int, total: int) -> None:
        """Brief transition card for each scene."""
        font_h2 = get_font(H2_SIZE, bold=True)
        font_sm = get_font(FONT_SIZE)

        for i in range(int(FPS * 1.5)):
            img, draw = make_frame()
            a = min(1.0, i / (FPS * 0.4))

            # Tinted left bar
            draw.rectangle([0, 0, 8, HEIGHT], fill=BLUE)

            # Scene number
            num_text = f"Scene {scene_num}/{total}"
            c_dim = tuple(int(v * a) for v in DIM)
            draw.text((MARGIN, 40), num_text, fill=c_dim, font=font_sm)

            # Title with icon
            title = f"{scene.icon}  {scene.title}"
            c_blue = tuple(int(v * a) for v in BLUE)
            draw.text((MARGIN, HEIGHT // 2 - 60), title, fill=c_blue, font=font_h2)

            # Description
            c_text = tuple(int(v * a) for v in TEXT)
            draw.text((MARGIN, HEIGHT // 2), scene.description, fill=c_text, font=font_sm)

            # Modality badge
            badge_colors = {
                "text": TEAL, "vision": PEACH, "audio": MAUVE, "video": RED, "multi": YELLOW,
            }
            badge_col = badge_colors.get(scene.modality, TEXT)
            badge = f"[{scene.modality.upper()}]"
            c_badge = tuple(int(v * a) for v in badge_col)
            draw.text((MARGIN, HEIGHT // 2 + 40), badge, fill=c_badge, font=font_sm)

            self._write_frame(img)

    # ── Main scene rendering ──

    def render_scene(self, scene: DemoScene, scene_num: int) -> None:
        """Render one full demo scene with typing effect."""
        font = get_font(FONT_SIZE)
        font_bold = get_font(FONT_SIZE, bold=True)
        font_h3 = get_font(H3_SIZE, bold=True)
        font_small = get_font(16)

        # Header bar height
        header_h = 50
        # Chat area
        chat_top = header_h + 10

        # Prepare user message lines
        user_lines = wrap_text(scene.user_input or "(status 命令)", max_chars=100)

        # Prepare agent response lines
        resp_lines = wrap_text(scene.agent_response, max_chars=100)
        # Limit to avoid too many frames for very long responses
        if len(resp_lines) > 30:
            resp_lines = resp_lines[:28] + ["...", f"(共 {len(wrap_text(scene.agent_response, 100))} 行)"]

        # === Phase 1: Type out user message ===
        total_user_chars = sum(len(l) for l in user_lines)
        chars_per_frame = max(2, total_user_chars // (FPS * 2))  # type in ~2s

        typed = 0
        while typed <= total_user_chars:
            img, draw = make_frame()
            self._draw_header(draw, scene, scene_num, font_bold, font_small)

            # Thumbnail for vision scenes
            thumb_offset = 0
            if scene.thumbnail:
                thumb_offset = self._draw_thumbnail(img, draw, scene.thumbnail, chat_top + 10)

            y = chat_top + 10 + thumb_offset
            # User label
            draw.text((MARGIN, y), "🧑 User:", fill=TEAL, font=font_bold)
            y += LINE_HEIGHT + 4

            # Type user text
            chars_left = typed
            for line in user_lines:
                if chars_left <= 0:
                    break
                visible = line[:chars_left]
                draw.text((MARGIN + 20, y), visible, fill=TEXT, font=font)
                chars_left -= len(line)
                y += LINE_HEIGHT

            # Cursor blink
            if (typed // 3) % 2 == 0:
                draw.text((MARGIN + 20 + len(user_lines[min(len(user_lines)-1, 0)]) * 5, y - LINE_HEIGHT), "▊", fill=GREEN, font=font)

            self._draw_footer(draw, scene, font_small)
            self._write_frame(img)
            typed += chars_per_frame

        # Hold on complete user message briefly
        img_user_done, draw_user_done = make_frame()
        self._draw_header(draw_user_done, scene, scene_num, font_bold, font_small)
        y = chat_top + 10
        thumb_offset = 0
        if scene.thumbnail:
            thumb_offset = self._draw_thumbnail(img_user_done, draw_user_done, scene.thumbnail, y)
        y += thumb_offset
        draw_user_done.text((MARGIN, y), "🧑 User:", fill=TEAL, font=font_bold)
        y += LINE_HEIGHT + 4
        for line in user_lines:
            draw_user_done.text((MARGIN + 20, y), line, fill=TEXT, font=font)
            y += LINE_HEIGHT
        self._draw_footer(draw_user_done, scene, font_small)
        self._hold(img_user_done, 0.8)

        # === Phase 2: "Thinking" animation ===
        thinking_frames = int(FPS * min(2.0, scene.elapsed))
        for i in range(thinking_frames):
            img, draw = make_frame()
            self._draw_header(draw, scene, scene_num, font_bold, font_small)
            y = chat_top + 10
            thumb_offset = 0
            if scene.thumbnail:
                thumb_offset = self._draw_thumbnail(img, draw, scene.thumbnail, y)
            y += thumb_offset
            draw.text((MARGIN, y), "🧑 User:", fill=TEAL, font=font_bold)
            y += LINE_HEIGHT + 4
            for line in user_lines:
                draw.text((MARGIN + 20, y), line, fill=TEXT, font=font)
                y += LINE_HEIGHT

            y += LINE_HEIGHT
            draw.text((MARGIN, y), "🤖 Pinocchio:", fill=PEACH, font=font_bold)
            y += LINE_HEIGHT + 4

            # Animated dots
            dots = "·" * ((i // 8) % 4 + 1)
            phases = ["感知中", "策略规划", "执行中", "评估中", "学习中", "反思中"]
            phase = phases[(i // 15) % len(phases)]
            draw.text((MARGIN + 20, y), f"🔄 认知循环 — {phase} {dots}", fill=YELLOW, font=font)

            # Timer
            elapsed_show = min(scene.elapsed, i / FPS)
            draw.text((WIDTH - 150, y), f"{elapsed_show:.1f}s", fill=DIM, font=font_small)

            self._draw_footer(draw, scene, font_small)
            self._write_frame(img)

        # === Phase 3: Stream agent response ===
        total_resp_chars = sum(len(l) for l in resp_lines)
        resp_chars_per_frame = max(3, total_resp_chars // (FPS * 5))  # 5s for response

        typed = 0
        while typed <= total_resp_chars:
            img, draw = make_frame()
            self._draw_header(draw, scene, scene_num, font_bold, font_small)

            y = chat_top + 10
            thumb_offset = 0
            if scene.thumbnail:
                thumb_offset = self._draw_thumbnail(img, draw, scene.thumbnail, y)
            y += thumb_offset

            # User message (compact version if space is tight)
            draw.text((MARGIN, y), "🧑 User:", fill=TEAL, font=font_bold)
            y += LINE_HEIGHT + 4
            user_show = user_lines[:3]
            if len(user_lines) > 3:
                user_show = user_lines[:2] + [f"… ({len(user_lines)} 行)"]
            for line in user_show:
                draw.text((MARGIN + 20, y), line, fill=DIM, font=font)
                y += LINE_HEIGHT

            y += LINE_HEIGHT
            draw.text((MARGIN, y), "🤖 Pinocchio:", fill=PEACH, font=font_bold)
            y += LINE_HEIGHT + 4

            # Stream response text
            max_visible_lines = (HEIGHT - y - 80) // LINE_HEIGHT
            chars_left = typed
            visible_resp_lines: list[str] = []
            for line in resp_lines:
                if chars_left <= 0:
                    break
                visible = line[:chars_left]
                visible_resp_lines.append(visible)
                chars_left -= len(line)

            # Scroll if too many lines
            start_line = max(0, len(visible_resp_lines) - max_visible_lines)
            for line in visible_resp_lines[start_line:]:
                if y > HEIGHT - 80:
                    break
                # Syntax highlighting for code
                if line.strip().startswith(("def ", "class ", "import ", "from ")):
                    draw.text((MARGIN + 20, y), line, fill=BLUE, font=font)
                elif line.strip().startswith("#"):
                    draw.text((MARGIN + 20, y), line, fill=DIM, font=font)
                elif line.strip().startswith(("return ", "if ", "for ", "while ")):
                    draw.text((MARGIN + 20, y), line, fill=MAUVE, font=font)
                elif '"""' in line or "'''" in line:
                    draw.text((MARGIN + 20, y), line, fill=GREEN, font=font)
                else:
                    draw.text((MARGIN + 20, y), line, fill=TEXT, font=font)
                y += LINE_HEIGHT

            self._draw_footer(draw, scene, font_small)
            self._write_frame(img)
            typed += resp_chars_per_frame

        # === Phase 4: Hold final frame ===
        self._hold(img, 2.0)

    def _draw_header(self, draw: ImageDraw.Draw, scene: DemoScene, num: int,
                     font_bold: ImageFont.FreeTypeFont, font_small: ImageFont.FreeTypeFont) -> None:
        draw.rectangle([0, 0, WIDTH, 46], fill=SURFACE)
        draw.text((MARGIN, 12), f"{scene.icon} {scene.title}", fill=BLUE, font=font_bold)

        badge_colors = {"text": TEAL, "vision": PEACH, "audio": MAUVE, "video": RED}
        bc = badge_colors.get(scene.modality, TEXT)
        badge = f"[{scene.modality.upper()}]"
        draw.text((WIDTH - 200, 14), badge, fill=bc, font=font_small)

    def _draw_footer(self, draw: ImageDraw.Draw, scene: DemoScene,
                     font_small: ImageFont.FreeTypeFont) -> None:
        draw.rectangle([0, HEIGHT - 36, WIDTH, HEIGHT], fill=SURFACE)
        draw.text((MARGIN, HEIGHT - 28), f"⏱ {scene.elapsed:.1f}s", fill=DIM, font=font_small)
        draw.text((WIDTH // 2 - 100, HEIGHT - 28),
                  "Pinocchio · qwen3-vl:8b · Cognitive Loop", fill=DIM, font=font_small)

    def _draw_thumbnail(self, img: Image.Image, draw: ImageDraw.Draw,
                        thumb: Image.Image, y_start: int) -> int:
        """Paste a scaled thumbnail on the right side. Returns height offset."""
        max_w, max_h = 350, 250
        tw, th = thumb.size
        scale = min(max_w / tw, max_h / th)
        new_w, new_h = int(tw * scale), int(th * scale)
        resized = thumb.resize((new_w, new_h), Image.Resampling.LANCZOS)
        x = WIDTH - new_w - MARGIN
        img.paste(resized, (x, y_start))
        # Border
        draw.rectangle([x - 2, y_start - 2, x + new_w + 2, y_start + new_h + 2],
                       outline=SURFACE, width=2)
        return 0  # No vertical offset — thumb is positioned to the right

    # ── Summary screen ──

    def render_summary(self, scenes: list[DemoScene], duration: float = 5.0) -> None:
        font_h1 = get_font(H1_SIZE, bold=True)
        font_h2 = get_font(H2_SIZE)
        font = get_font(FONT_SIZE)

        total_time = sum(s.elapsed for s in scenes)
        n_scenes = len(scenes)
        n_vision = sum(1 for s in scenes if s.modality == "vision")
        n_text = sum(1 for s in scenes if s.modality == "text")

        for i in range(int(FPS * duration)):
            img, draw = make_frame()
            a = min(1.0, i / (FPS * 0.6))

            c1 = tuple(int(v * a) for v in BLUE)
            c2 = tuple(int(v * a) for v in TEXT)
            c3 = tuple(int(v * a) for v in GREEN)
            c4 = tuple(int(v * a) for v in DIM)
            c5 = tuple(int(v * a) for v in LAVENDER)

            draw.text((MARGIN, 40), "📋 演示总结", fill=c1, font=font_h1)

            y = 120
            stats = [
                (f"总场景数: {n_scenes}", TEAL),
                (f"文本场景: {n_text}  |  视觉场景: {n_vision}", LAVENDER),
                (f"总耗时: {total_time:.1f}s  |  平均: {total_time/max(n_scenes,1):.1f}s/场景", GREEN),
            ]
            for text, color in stats:
                c = tuple(int(v * a) for v in color)
                draw.text((MARGIN, y), text, fill=c, font=font_h2)
                y += 44

            y += 20
            draw.text((MARGIN, y), "场景回顾:", fill=c4, font=font)
            y += 36

            for j, scene in enumerate(scenes):
                if y > HEIGHT - 80:
                    break
                line = f"  {scene.icon}  {scene.title}  —  {scene.elapsed:.1f}s"
                ok = "✅" if "[Error" not in scene.agent_response else "❌"
                line = f"{ok} {line}"
                draw.text((MARGIN, y), line, fill=c2, font=font)
                y += LINE_HEIGHT

            foot = "Pinocchio Multimodal Agent · Live Demo · 2026-03-01"
            bxf = draw.textbbox((0, 0), foot, font=font)
            draw.text(((WIDTH - bxf[2] + bxf[0]) // 2, HEIGHT - 60), foot, fill=c4, font=font)

            self._write_frame(img)


# ── Main ─────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("🎬 Pinocchio 多模态智能体演示视频生成器")
    print("=" * 60)

    # Step 1: Try cache first, otherwise run LLM interactions
    render_only = "--render-only" in sys.argv
    scenes = None
    if render_only:
        scenes = load_cache()
        if scenes is None:
            print("⚠️  No cache found, falling back to full run")

    if scenes is None:
        scenes = run_demo_scenarios()
        save_cache(scenes)

    # Step 2: Render video
    print("\n🎬 Rendering video …")
    renderer = VideoRenderer()

    renderer.render_title(duration=4.0)

    for i, scene in enumerate(scenes, 1):
        print(f"   🎞  Rendering scene {i}/{len(scenes)}: {scene.title}")
        renderer.render_transition(scene, i, len(scenes))
        renderer.render_scene(scene, i)

    renderer.render_summary(scenes, duration=5.0)
    renderer.close()

    print(f"\n{'=' * 60}")
    print(f"✅ 演示视频已生成: {OUTPUT_PATH}")
    print(f"   分辨率: {WIDTH}×{HEIGHT} @ {FPS}fps")
    total_s = sum(s.elapsed for s in scenes)
    print(f"   场景数: {len(scenes)}")
    print(f"   LLM总耗时: {total_s:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
