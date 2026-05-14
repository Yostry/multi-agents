"""
多 API Key 自动切换提供器

支持 DeepSeek + Kimi 双路 LLM 提供商，当一路 API 配额耗尽（429/rate limit）时
自动切换到下一路。也支持单路 Embedding 的智谱 API。

用法:
    from amesim_multi_agent.model_providers import create_run_config
    run_config = create_run_config()
"""

from __future__ import annotations

import asyncio
import logging
import os

from openai import APIStatusError, AsyncOpenAI, RateLimitError

from agents import (
    Agent,
    Model,
    ModelProvider,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunConfig,
    set_tracing_disabled,
)

logger = logging.getLogger(__name__)

# ============================================================
# 提供商配置
# ============================================================

# DeepSeek (优先级 1)
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY") or "sk-1cd545d1cb534ab3876440fb86689315"

# Kimi 付费平台 (优先级 2)
KIMI_BASE_URL = "https://api.kimi.com/coding/v1"
KIMI_MODEL = "kimi-for-coding"
KIMI_API_KEY = os.environ.get("KIMI_API_KEY") or "sk-Yv31cVy8ih0ZXN2BCEv9MFOy1XrrYMZOTsGS8CgVyM59WJzy"

# ============================================================
# 全局设置
# ============================================================

# 关闭追踪 (不依赖 OpenAI 平台)
set_tracing_disabled(disabled=True)

# Kimi Code API 要求特定 User-Agent 进行订阅认证
from agents.models.chatcmpl_helpers import HEADERS_OVERRIDE  # noqa: E402
HEADERS_OVERRIDE.set({"User-Agent": "claude-code/0.1.0"})


# ============================================================
# 配额耗尽错误检测
# ============================================================

def _is_quota_exhausted_error(error: Exception) -> bool:
    """判断是否为配额耗尽/限流错误，需要切换到下一个 API Key。

    识别规则:
    - openai.RateLimitError (HTTP 429)
    - APIStatusError 且 status_code 为 429 或 403(某些平台的配额=403)
    - 错误消息包含 quota/rate limit/exhausted/超出/额度 等关键词
    """
    # RateLimitError 是 APIStatusError 子类，status_code=429
    if isinstance(error, RateLimitError):
        return True

    if isinstance(error, APIStatusError):
        if error.status_code == 429:
            return True
        if error.status_code == 403:
            # 403 可能是配额耗尽(某些平台)，检查消息
            msg = str(error.body if hasattr(error, "body") else error).lower()
            quota_keywords = [
                "quota", "rate limit", "exhausted", "exceeded",
                "超出", "额度", "不足", "欠费", "balance", "insufficient",
                "billing", "充值",
            ]
            if any(kw in msg for kw in quota_keywords):
                return True

    return False


# ============================================================
# FallbackModel — 多 Key 自动切换的 Model 包装器
# ============================================================

class FallbackModel(Model):
    """包装多个 OpenAIChatCompletionsModel，遇配额耗尽自动切换。

    每个 Agent 调用链共享同一个 FallbackModel 实例，
    实例级别的 _exhausted 集合确保已耗尽的 Key 不会被当前链路重试。
    """

    def __init__(self, models: list[OpenAIChatCompletionsModel], names: list[str]):
        self._models = models
        self._names = names
        self._exhausted: set[int] = set()

    def _active_index(self) -> int | None:
        """返回第一个未耗尽的 model 索引，若全部耗尽返回 None。"""
        for i in range(len(self._models)):
            if i not in self._exhausted:
                return i
        return None

    def _current_model(self) -> OpenAIChatCompletionsModel | None:
        idx = self._active_index()
        return self._models[idx] if idx is not None else None

    # ---- 核心: 包装 get_response ----

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list,
        model_settings: ModelSettings,
        tools: list,
        output_schema,
        handoffs: list,
        tracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt=None,
    ):
        last_error: Exception | None = None

        while True:
            idx = self._active_index()
            if idx is None:
                # 全部耗尽
                print("\n[API] ⚠ 所有 API Key 均已耗尽！")
                break

            model = self._models[idx]
            name = self._names[idx]

            try:
                return await model.get_response(
                    system_instructions=system_instructions,
                    input=input,
                    model_settings=model_settings,
                    tools=tools,
                    output_schema=output_schema,
                    handoffs=handoffs,
                    tracing=tracing,
                    previous_response_id=previous_response_id,
                    conversation_id=conversation_id,
                    prompt=prompt,
                )
            except Exception as e:
                if _is_quota_exhausted_error(e):
                    self._exhausted.add(idx)
                    remaining = len(self._models) - len(self._exhausted)
                    print(f"\n[API] 🔄 {name} 配额耗尽 → 切换中... (剩余 {remaining} 个可用 Key)")
                    last_error = e
                    continue
                # 非配额错误，直接抛出
                raise

        raise last_error or RuntimeError("所有 API Key 均已耗尽")

    # ---- 核心: 包装 stream_response ----

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list,
        model_settings: ModelSettings,
        tools: list,
        output_schema,
        handoffs: list,
        tracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt=None,
    ):
        last_error: Exception | None = None

        while True:
            idx = self._active_index()
            if idx is None:
                print("\n[API] ⚠ 所有 API Key 均已耗尽！")
                break

            model = self._models[idx]
            name = self._names[idx]

            try:
                async for event in model.stream_response(
                    system_instructions=system_instructions,
                    input=input,
                    model_settings=model_settings,
                    tools=tools,
                    output_schema=output_schema,
                    handoffs=handoffs,
                    tracing=tracing,
                    previous_response_id=previous_response_id,
                    conversation_id=conversation_id,
                    prompt=prompt,
                ):
                    yield event
                return  # 成功完成
            except Exception as e:
                if _is_quota_exhausted_error(e):
                    self._exhausted.add(idx)
                    remaining = len(self._models) - len(self._exhausted)
                    print(f"\n[API] 🔄 {name} 配额耗尽 → 切换中... (剩余 {remaining} 个可用 Key)")
                    last_error = e
                    continue
                raise

        raise last_error or RuntimeError("所有 API Key 均已耗尽")


# ============================================================
# MultiKeyModelProvider — 支持多 API Key 的 ModelProvider
# ============================================================

class MultiKeyModelProvider(ModelProvider):
    """支持多个 API Key 的 ModelProvider，当一个耗尽时自动切换到下一个。

    优先级顺序即为 providers 列表顺序:
      1. DeepSeek (deepseek-chat)
      2. Kimi    (kimi-for-coding)
    """

    def __init__(self) -> None:
        # 构建所有 providers 的 client+model
        self._providers: list[dict] = []

        # --- DeepSeek ---
        deepseek_client = AsyncOpenAI(
            base_url=DEEPSEEK_BASE_URL,
            api_key=DEEPSEEK_API_KEY,
        )
        self._providers.append({
            "name": "DeepSeek",
            "model_name": DEEPSEEK_MODEL,
            "client": deepseek_client,
        })

        # --- Kimi ---
        kimi_client = AsyncOpenAI(
            base_url=KIMI_BASE_URL,
            api_key=KIMI_API_KEY,
            default_headers={"User-Agent": "claude-code/0.1.0"},
        )
        self._providers.append({
            "name": "Kimi",
            "model_name": KIMI_MODEL,
            "client": kimi_client,
        })

    def get_model(self, model_name: str | None) -> Model:
        """返回一个 FallbackModel，包含所有 provider 的 model 实例。"""
        models = [
            OpenAIChatCompletionsModel(
                model=p["model_name"],
                openai_client=p["client"],
            )
            for p in self._providers
        ]
        names = [p["name"] for p in self._providers]
        return FallbackModel(models=models, names=names)


# ============================================================
# 全局单例 & 便捷函数
# ============================================================

_multi_provider: MultiKeyModelProvider | None = None


def get_multi_provider() -> MultiKeyModelProvider:
    """获取全局多 Key ModelProvider 单例。"""
    global _multi_provider
    if _multi_provider is None:
        _multi_provider = MultiKeyModelProvider()
    return _multi_provider


def create_run_config() -> RunConfig:
    """创建使用多 Key 自动切换的 RunConfig。

    优先级: DeepSeek → Kimi
    任一 Key 配额耗尽时自动切换到下一个。
    """
    return RunConfig(
        model_provider=get_multi_provider(),
        model_settings=ModelSettings(
            extra_body={"thinking": {"type": "disabled"}},
        ),
    )


# ---- 向后兼容: 保留 kimi_provider 接口 ----

def create_kimi_agent(
    name: str,
    instructions: str,
    tools: list | None = None,
    handoffs: list | None = None,
    model_name: str | None = None,
) -> Agent:
    """快捷创建 Agent（使用多 Key 提供器，向后兼容）。

    Args:
        name: Agent 名称
        instructions: 系统指令
        tools: 工具列表
        handoffs: handoff 列表
        model_name: 模型名（未指定时使用 DeepSeek 默认模型）

    Returns:
        配置好多 Key 模型的 Agent
    """
    return Agent(
        name=name,
        instructions=instructions,
        tools=tools or [],
        handoffs=handoffs or [],
        model=get_multi_provider().get_model(model_name or DEEPSEEK_MODEL),
    )


def create_kimi_run_config() -> RunConfig:
    """创建 RunConfig（向后兼容，实际使用多 Key 提供器）。"""
    return create_run_config()


# 兼容别名
KIMI_DEFAULT_MODEL = DEEPSEEK_MODEL
KIMI_MODEL_NAME = DEEPSEEK_MODEL
KIMI_API_KEY_REF = KIMI_API_KEY
