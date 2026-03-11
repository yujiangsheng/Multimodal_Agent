"""基础对话示例 — 演示 Pinocchio 核心 API。

包括：纯文本对话、流式输出、多模态输入、会话状态查看。

运行前请确保 Ollama 正在运行：
    ollama serve
    ollama pull qwen3-vl:4b
"""

from pinocchio import Pinocchio


def basic_chat():
    """最简单的对话用法。"""
    agent = Pinocchio()

    # 单轮对话
    response = agent.chat("用三句话介绍量子计算")
    print("=== 基本对话 ===")
    print(response)
    print()


def streaming_chat():
    """流式输出 — 逐 token 返回。"""
    agent = Pinocchio()

    print("=== 流式输出 ===")
    for chunk in agent.chat_stream("讲一个关于 AI 的短故事"):
        print(chunk, end="", flush=True)
    print("\n")


def multimodal_chat():
    """多模态输入 — 图像 / 音频 / 视频。"""
    agent = Pinocchio()

    # 图片理解
    # response = agent.chat("描述这张图片", image_paths=["photo.jpg"])

    # 音频转写
    # response = agent.chat("总结录音要点", audio_paths=["meeting.wav"])

    # 多模态混合
    # response = agent.chat("对比图片和视频",
    #                       image_paths=["a.jpg"],
    #                       video_paths=["b.mp4"])

    print("=== 多模态 ===")
    print("取消注释上方代码并提供真实文件路径即可使用")
    print()


def check_status():
    """查看智能体状态。"""
    import json

    agent = Pinocchio()
    agent.chat("Hello!")

    print("=== 智能体状态 ===")
    print(json.dumps(agent.status(), ensure_ascii=False, indent=2))
    print()


if __name__ == "__main__":
    basic_chat()
    # streaming_chat()
    # multimodal_chat()
    # check_status()
