#!/usr/bin/env python3
"""
Amesim 多 Agent 自动化仿真入口 (兼容旧入口)

重定向到新的 6-Agent 编排器。

用法:
    uv run python -m amesim_multi_agent.runner "构建一个质量-弹簧-阻尼系统"
"""
from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 重定向到新的编排器
from .orchestrator_runner import main as _orchestrator_main

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python -m amesim_multi_agent.runner '<需求描述>'")
        print()
        print("示例:")
        print('  uv run python -m amesim_multi_agent.runner "构建一个质量块连弹簧再连阻尼器的系统"')
        print('  uv run python -m amesim_multi_agent.runner "构建一个朗肯循环发电模型"')
        sys.exit(1)

    user_request = " ".join(sys.argv[1:])
    asyncio.run(_orchestrator_main(user_request))
