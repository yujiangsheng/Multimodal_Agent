"""Pinocchio — Multimodal Self-Evolving Agent

Interactive CLI entry-point.  Starts a REPL loop where the user types
messages and Pinocchio responds through a full 6-phase cognitive loop
(PERCEIVE → STRATEGIZE → EXECUTE → EVALUATE → LEARN → META-REFLECT).

Usage
-----
    python main.py

Built-in commands (typed at the prompt):
    ``quit`` / ``exit`` / ``q``  — exit the session
    ``status``                   — print agent state as JSON
    ``reset``                    — clear session (persistent memory kept)

Multimodal input:
    Attach files with ``--image``, ``--audio``, or ``--video`` flags::

        请描述这张图片 --image photo.jpg
        总结这段录音 --audio meeting.wav
        分析这个视频 --video lecture.mp4

Optional environment variables:
    PINOCCHIO_MODEL       — LLM model name (default: qwen3-vl:4b)
    OPENAI_BASE_URL       — Custom API base URL (default: http://localhost:11434/v1)
    PINOCCHIO_DATA_DIR    — Directory for persistent memory (default: data)
"""

from __future__ import annotations

import json
import re
import sys

from config import PinocchioConfig
from pinocchio import Pinocchio


def _parse_input(raw: str) -> tuple[str, list[str], list[str], list[str]]:
    """Parse user input, extracting ``--image`` / ``--audio`` / ``--video`` flags.

    Returns ``(text, image_paths, audio_paths, video_paths)``.
    """
    images: list[str] = []
    audios: list[str] = []
    videos: list[str] = []

    pattern = r'--(image|audio|video)\s+(\S+)'
    for match in re.finditer(pattern, raw):
        flag, path = match.group(1), match.group(2)
        {"image": images, "audio": audios, "video": videos}[flag].append(path)

    text = re.sub(pattern, '', raw).strip()
    return text, images, audios, videos


def main() -> None:
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
        num_ctx=cfg.num_ctx,
    )

    print(agent.greet())
    print()
    print("输入消息与 Pinocchio 对话 (输入 'quit' 退出, 'status' 查看状态)")
    print("支持多模态: 请描述图片 --image photo.jpg --audio clip.wav")
    print("─" * 60)

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！Pinocchio 会记住今天学到的一切。👋")
            break

        if user_input.lower() == "status":
            status = agent.status()
            print(json.dumps(status, ensure_ascii=False, indent=2))
            continue

        if user_input.lower() == "reset":
            agent.reset()
            print("会话已重置（持久化记忆保留）。")
            continue

        # Parse multimodal flags
        text, images, audios, videos = _parse_input(user_input)

        response = agent.chat(
            text or None,
            image_paths=images or None,
            audio_paths=audios or None,
            video_paths=videos or None,
        )
        print(f"\n🤖 Pinocchio: {response}")


if __name__ == "__main__":
    main()
