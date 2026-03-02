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

Optional environment variables:
    PINOCCHIO_MODEL       — LLM model name (default: qwen2.5-omni)
    OPENAI_BASE_URL       — Custom API base URL (default: http://localhost:11434/v1)
    PINOCCHIO_DATA_DIR    — Directory for persistent memory (default: data)
"""

from __future__ import annotations

import sys

from config import PinocchioConfig
from pinocchio import Pinocchio


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
    )

    print(agent.greet())
    print()
    print("输入消息与 Pinocchio 对话 (输入 'quit' 退出, 'status' 查看状态)")
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
            import json
            status = agent.status()
            print(json.dumps(status, ensure_ascii=False, indent=2))
            continue

        if user_input.lower() == "reset":
            agent.reset()
            print("会话已重置（持久化记忆保留）。")
            continue

        # ── Run the cognitive loop ──
        response = agent.chat(user_input)
        print(f"\n🤖 Pinocchio: {response}")


if __name__ == "__main__":
    main()
