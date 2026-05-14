"""
Tri-Agent Amesim 建模系统

三 Agent 架构:
    Orchestrator Agent  (主管: 需求理解 → 拓扑规划 → TopologyGraph)
        ↓
    Connector Agent    (连接: 核心元件 → 螺旋展开 → TorsionBarJSON)
        ↓
    Parameter-Runner Agent (参数运行: 子模型 → 参数 → 构建 → 仿真 → 提取)

独立测试入口:
    python -m amesim_multi_agent.tri_agent.test_orchestrator "需求描述"
    python -m amesim_multi_agent.tri_agent.test_connector topology_graph.json
    python -m amesim_multi_agent.tri_agent.test_runner model.json

完整流水线:
    python -m amesim_multi_agent.tri_agent.pipeline "需求描述"
"""

from .topology_graph import (
    TopologyNode,
    TopologyEdge,
    TopologyGraph,
    Subsystem,
    ComponentEntry,
    ConnectionEntry,
    TorsionBarModel,
)
from .experience_memory import ExperienceMemory
from .agents import (
    create_tri_orchestrator_agent,
    create_orchestrator_with_tools,
    create_connector_phase1_agent,
    create_connector_phase2_agent,
    create_connector_phase1_with_tools,
    create_connector_phase2_with_tools,
    create_parameter_runner_agent,
    create_parameter_runner_with_tools,
)

__all__ = [
    # Data models
    "TopologyNode",
    "TopologyEdge",
    "TopologyGraph",
    "Subsystem",
    "ComponentEntry",
    "ConnectionEntry",
    "TorsionBarModel",
    # Memory
    "ExperienceMemory",
    # Agent factories
    "create_tri_orchestrator_agent",
    "create_orchestrator_with_tools",
    "create_connector_phase1_agent",
    "create_connector_phase2_agent",
    "create_connector_phase1_with_tools",
    "create_connector_phase2_with_tools",
    "create_parameter_runner_agent",
    "create_parameter_runner_with_tools",
]
