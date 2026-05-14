"""
Orchestrator Agent — 主管 Agent 工厂函数

职责: 需求理解 → 物理拓扑规划 → TopologyGraph 有向图
"""

from __future__ import annotations

from agents import Agent

from ...model_providers import get_multi_provider, DEEPSEEK_MODEL
from ..prompts.orchestrator_prompt import ORCHESTRATOR_SYSTEM_PROMPT


def create_tri_orchestrator_agent() -> Agent:
    """创建主管 Agent (三agent架构)。

    Tools:
      - web_search: 互联网知识检索 (中英文专业源)
      - query_experience_memory: 查询历史成功案例
      - search_component: 验证元件在 KB 中的存在性
    """
    return Agent(
        name="Orchestrator",
        instructions=ORCHESTRATOR_SYSTEM_PROMPT,
        model=get_multi_provider().get_model(DEEPSEEK_MODEL),
        tools=[
            # Tools are attached at pipeline runtime via tool injection
            # to allow test mode (mock tools) vs production mode (live tools)
        ],
    )


def create_orchestrator_with_tools(
    web_search,
    experience_query,
    kb_search,
    ask_user=None,
) -> Agent:
    """创建带完整工具集的主管 Agent。

    Args:
        web_search: 互联网搜索函数 (function_tool)
        experience_query: 经验记忆查询函数
        kb_search: 知识库搜索函数 (统一搜索)
        ask_user: 向用户提问函数 (可选, 测试时可用 mock)
    """
    tools = [web_search, experience_query, kb_search]
    if ask_user:
        tools.append(ask_user)

    return Agent(
        name="Orchestrator",
        instructions=ORCHESTRATOR_SYSTEM_PROMPT,
        model=get_multi_provider().get_model(DEEPSEEK_MODEL),
        tools=tools,
    )
