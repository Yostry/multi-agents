"""
Kimi (Moonshot) 模型适配器 — 向后兼容层

本模块已升级为多 Key 自动切换架构。
实际实现位于 model_providers.py，本模块保留所有公开接口以保证向后兼容。

多 Key 优先级:
  1. DeepSeek (deepseek-chat)
  2. Kimi    (kimi-for-coding)

任一 API Key 配额耗尽时自动切换到下一个。

用法（不变）:
    from amesim_multi_agent.kimi_provider import create_kimi_agent, create_kimi_run_config
    agent = create_kimi_agent(name="MyAgent", instructions="...")
    run_config = create_kimi_run_config()
"""
from __future__ import annotations

# 从新模块导入所有公开接口
from .model_providers import (
    # 便捷函数
    create_kimi_agent,
    create_kimi_run_config,
    create_run_config,
    get_multi_provider,
    # Provider 类
    MultiKeyModelProvider,
    FallbackModel,
    # 配置常量
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_API_KEY,
    KIMI_BASE_URL,
    KIMI_MODEL,
    KIMI_API_KEY,
    KIMI_DEFAULT_MODEL,
    KIMI_MODEL_NAME,
    KIMI_API_KEY_REF,
    # 工具函数
    _is_quota_exhausted_error,
)

# 兼容旧代码的模块级引用
KimiModelProvider = MultiKeyModelProvider  # 旧名称别名
kimi_provider = get_multi_provider()  # 全局单例
