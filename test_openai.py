#!/usr/bin/env python3
"""
OpenAI 兼容 API 测试脚本
测试沙盒中注入的 API_KEY 和 ENDPOINT 是否正常工作
"""

import os
import yaml
from pathlib import Path
from openai import OpenAI

# ─────────────────────────────────────────────
# 1. 加载配置（优先读取 ~/.genspark_llm.yaml，回退到环境变量）
# ─────────────────────────────────────────────
def load_config():
    config_path = Path.home() / ".genspark_llm.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        api_key  = cfg["openai"]["api_key"]
        base_url = cfg["openai"]["base_url"]
        print(f"✅ 配置来源：~/.genspark_llm.yaml")
    else:
        api_key  = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        print(f"✅ 配置来源：环境变量")

    print(f"   BASE_URL : {base_url}")
    print(f"   API_KEY  : {api_key[:20]}...\n")
    return api_key, base_url


# ─────────────────────────────────────────────
# 2. 普通请求（Non-streaming）
# ─────────────────────────────────────────────
def test_normal(client, model="gpt-5"):
    print(f"{'='*50}")
    print(f"🧪 测试 1：普通请求  (model={model})")
    print(f"{'='*50}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个简洁的助手，回答控制在 50 字以内。"},
            {"role": "user",   "content": "用一句话介绍你自己。"}
        ]
    )
    content = response.choices[0].message.content
    print(f"🤖 回复：{content}")
    print(f"📊 Token 使用：{response.usage}\n")


# ─────────────────────────────────────────────
# 3. 流式请求（Streaming）
# ─────────────────────────────────────────────
def test_streaming(client, model="gpt-5"):
    print(f"{'='*50}")
    print(f"🧪 测试 2：流式请求  (model={model})")
    print(f"{'='*50}")

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个简洁的助手，回答控制在 80 字以内。"},
            {"role": "user",   "content": "请列举 3 个 Python 的优点。"}
        ],
        stream=True
    )

    print("🤖 流式回复：", end="", flush=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
    print("\n")


# ─────────────────────────────────────────────
# 4. 多模型测试
# ─────────────────────────────────────────────
def test_models(client):
    models = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]
    print(f"{'='*50}")
    print(f"🧪 测试 3：多模型测试")
    print(f"{'='*50}")

    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "说「你好」"}],
                max_tokens=20
            )
            reply = response.choices[0].message.content.strip()
            print(f"  ✅ {model:<20} → {reply}")
        except Exception as e:
            print(f"  ❌ {model:<20} → 错误: {e}")
    print()


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 OpenAI 兼容 API 测试开始\n")

    api_key, base_url = load_config()

    client = OpenAI(api_key=api_key, base_url=base_url)

    test_normal(client)
    test_streaming(client)
    test_models(client)

    print("🎉 所有测试完成！")
