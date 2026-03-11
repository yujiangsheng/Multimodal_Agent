"""LLM 多供应商预设切换示例。

演示 PinocchioConfig.from_provider() 的用法。
# 离线可运行（仅展示配置，不需要连接 LLM）
"""

from config import PinocchioConfig, PROVIDER_PRESETS


def list_providers():
    """列出所有可用的预设供应商。"""
    print("=== 可用供应商预设 ===")
    for name, preset in PROVIDER_PRESETS.items():
        print(f"  {name:20s} → {preset.base_url}")
    print()


def switch_provider():
    """一行代码切换 LLM 后端。"""
    print("=== 切换供应商 ===")

    # 默认 Ollama
    cfg_ollama = PinocchioConfig.from_provider("ollama")
    print(f"Ollama:    model={cfg_ollama.model}, url={cfg_ollama.base_url}")

    # OpenAI
    cfg_openai = PinocchioConfig.from_provider("openai", model="gpt-4o")
    print(f"OpenAI:    model={cfg_openai.model}, url={cfg_openai.base_url}")

    # DeepSeek
    cfg_ds = PinocchioConfig.from_provider("deepseek", model="deepseek-chat")
    print(f"DeepSeek:  model={cfg_ds.model}, url={cfg_ds.base_url}")

    # 阿里云 Dashscope
    cfg_dash = PinocchioConfig.from_provider("dashscope", model="qwen-max")
    print(f"Dashscope: model={cfg_dash.model}, url={cfg_dash.base_url}")

    # Groq
    cfg_groq = PinocchioConfig.from_provider("groq", model="llama-3.3-70b-versatile")
    print(f"Groq:      model={cfg_groq.model}, url={cfg_groq.base_url}")

    print()


def custom_provider():
    """自定义供应商配置。"""
    print("=== 自定义配置 ===")

    cfg = PinocchioConfig(
        model="my-custom-model",
        base_url="http://my-server:8000/v1",
        api_key="my-api-key",
        temperature=0.5,
        max_tokens=4096,
    )
    print(f"Custom: model={cfg.model}, url={cfg.base_url}")

    # 使用配置创建 Pinocchio 实例
    # from pinocchio import Pinocchio
    # agent = Pinocchio(
    #     model=cfg.model,
    #     api_key=cfg.api_key,
    #     base_url=cfg.base_url,
    # )
    print()


if __name__ == "__main__":
    list_providers()
    switch_provider()
    custom_provider()
