"""
Connector Agent — 连接 Agent 工厂函数

职责: Phase 1 核心元件锚定 (重推理) + Phase 2 次要元件螺旋展开 (轻推理)
"""

from __future__ import annotations

from agents import Agent

from ...model_providers import get_multi_provider, DEEPSEEK_MODEL
from ..prompts.connector_prompts import (
    CONNECTOR_PHASE1_SYSTEM_PROMPT,
    CONNECTOR_PHASE2_SYSTEM_PROMPT,
)


def create_connector_phase1_agent() -> Agent:
    """创建连接 Agent Phase 1 (核心元件选择, 重推理)。

    Tools 由 pipeline 运行时注入, 包含:
      - batch_search_and_select (selector_tools)
      - search_component (selector_tools)
      - get_component_detail (selector_tools)
      - recommend_submodel (selector_tools)
    """
    return Agent(
        name="Connector_Phase1",
        instructions=CONNECTOR_PHASE1_SYSTEM_PROMPT,
        model=get_multi_provider().get_model(DEEPSEEK_MODEL),
        tools=[],
    )


def create_connector_phase2_agent() -> Agent:
    """创建连接 Agent Phase 2 (螺旋展开, 轻推理)。

    Tools 由 pipeline 运行时注入, 包含:
      - batch_query_ports (bridge_tools)
      - verify_all_components_exist (bridge_tools)
      - detect_dangling_ports (bridge_tools)
      - find_port_connections (bridge_tools)
      - find_bridge_paths (bridge_tools)
      - find_junction_component (bridge_tools)
      - validate_connection_plan (bridge_tools)
      - find_dangling_terminal_candidates (bridge_tools)
      - get_component_detail (selector_tools)
    """
    return Agent(
        name="Connector_Phase2",
        instructions=CONNECTOR_PHASE2_SYSTEM_PROMPT,
        model=get_multi_provider().get_model(DEEPSEEK_MODEL),
        tools=[],
    )


def create_connector_phase1_with_tools(
    batch_search_and_select,
    search_component,
    get_component_detail,
    recommend_submodel,
) -> Agent:
    """创建带完整工具集的连接 Agent Phase 1。"""
    return Agent(
        name="Connector_Phase1",
        instructions=CONNECTOR_PHASE1_SYSTEM_PROMPT,
        model=get_multi_provider().get_model(DEEPSEEK_MODEL),
        tools=[
            batch_search_and_select,
            search_component,
            get_component_detail,
            recommend_submodel,
        ],
    )


def create_connector_phase2_with_tools(
    batch_query_ports,
    verify_all_components_exist,
    detect_dangling_ports,
    find_port_connections,
    find_bridge_paths,
    find_junction_component,
    validate_connection_plan,
    find_dangling_terminal_candidates,
    get_component_detail,
) -> Agent:
    """创建带完整工具集的连接 Agent Phase 2。"""
    return Agent(
        name="Connector_Phase2",
        instructions=CONNECTOR_PHASE2_SYSTEM_PROMPT,
        model=get_multi_provider().get_model(DEEPSEEK_MODEL),
        tools=[
            batch_query_ports,
            verify_all_components_exist,
            detect_dangling_ports,
            find_port_connections,
            find_bridge_paths,
            find_junction_component,
            validate_connection_plan,
            find_dangling_terminal_candidates,
            get_component_detail,
        ],
    )
