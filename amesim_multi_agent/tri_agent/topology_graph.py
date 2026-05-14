"""
TopologyGraph — 物理拓扑有向图数据模型

主管 Agent 产出, 连接 Agent 消费。内存数据流, 不序列化为 JSON 文件。
所有 label 使用英文。

节点: 物理功能单元 (未绑定 Amesim 元件)
边:   物质流/能量流通道
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# 拓扑节点
# ============================================================

@dataclass
class TopologyNode:
    """有向图中的物理功能单元。"""

    node_id: str = ""                          # 唯一 ID, 如 "n1", "n2"
    label: str = ""                            # 英文功能名, 如 "LN2_Storage_Tank"
    role: str = ""                             # 从预定义词库选择的 role, 如 "thermal_storage"
    domain: str = ""                           # 物理域: two_phase_flow/mechanical_1d/thermal/...
    topological_role: str = ""                 # source | sink | storage | transfer | sensor | actuator
    is_core: bool = False                      # 是否核心元件 (不可被修复操作删除)
    suggested_library: Optional[str] = None    # 建议库: libtpf/libmec/libth/...
    functional_description: str = ""           # 功能描述 (英文)
    extracted_attrs: dict = field(default_factory=dict)    # 从用户描述提取的属性
    constraints: list[str] = field(default_factory=list)   # 设计约束
    unresolved: list[dict] = field(default_factory=list)   # 未解决的模糊点
    quantity: int = 1                          # 所需数量
    subsystem: str = ""                        # 所属子系统 (复杂模型分解时)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "role": self.role,
            "domain": self.domain,
            "topological_role": self.topological_role,
            "is_core": self.is_core,
            "suggested_library": self.suggested_library,
            "functional_description": self.functional_description,
            "extracted_attrs": self.extracted_attrs,
            "constraints": self.constraints,
            "unresolved": self.unresolved,
            "quantity": self.quantity,
            "subsystem": self.subsystem,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TopologyNode":
        return cls(
            node_id=d.get("node_id", ""),
            label=d.get("label", ""),
            role=d.get("role", ""),
            domain=d.get("domain", ""),
            topological_role=d.get("topological_role", ""),
            is_core=d.get("is_core", False),
            suggested_library=d.get("suggested_library"),
            functional_description=d.get("functional_description", ""),
            extracted_attrs=d.get("extracted_attrs", {}),
            constraints=d.get("constraints", []),
            unresolved=d.get("unresolved", []),
            quantity=d.get("quantity", 1),
            subsystem=d.get("subsystem", ""),
        )


# ============================================================
# 拓扑边
# ============================================================

@dataclass
class TopologyEdge:
    """连接两个拓扑节点的物质流或能量流通道。"""

    edge_id: str = ""                          # 唯一 ID, 如 "e1"
    source_node_id: str = ""                   # 从哪个节点
    target_node_id: str = ""                   # 到哪个节点
    flow_type: str = ""                        # mass_flow/heat_flow/mechanical_rotation/signal/...
    port_match: str = ""                       # 端口匹配约束, 如 "hflow→hflow"
    label: str = ""                            # 英文描述, 如 "pressurized_flow"
    domain: str = ""                           # 物理域
    is_cross_domain: bool = False              # 是否跨域连接 (需桥接元件)
    bridge_suggestion: Optional[str] = None     # 建议的桥接方案

    def to_dict(self) -> dict:
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "flow_type": self.flow_type,
            "port_match": self.port_match,
            "label": self.label,
            "domain": self.domain,
            "is_cross_domain": self.is_cross_domain,
            "bridge_suggestion": self.bridge_suggestion,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TopologyEdge":
        return cls(
            edge_id=d.get("edge_id", ""),
            source_node_id=d.get("source_node_id", ""),
            target_node_id=d.get("target_node_id", ""),
            flow_type=d.get("flow_type", ""),
            port_match=d.get("port_match", ""),
            label=d.get("label", ""),
            domain=d.get("domain", ""),
            is_cross_domain=d.get("is_cross_domain", False),
            bridge_suggestion=d.get("bridge_suggestion"),
        )


# ============================================================
# 子系统 (复杂模型分解)
# ============================================================

@dataclass
class Subsystem:
    """复杂模型分解后的子系统。"""

    subsys_id: str                             # 如 "sub_h2_supply"
    name: str                                  # 如 "Hydrogen_Supply"
    domain: str = ""                           # 主导物理域
    nodes: list[TopologyNode] = field(default_factory=list)
    edges: list[TopologyEdge] = field(default_factory=list)
    interfaces: list[dict] = field(default_factory=list)  # 对外接口端口

    def to_dict(self) -> dict:
        return {
            "subsys_id": self.subsys_id,
            "name": self.name,
            "domain": self.domain,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "interfaces": self.interfaces,
        }


# ============================================================
# 有向图
# ============================================================

@dataclass
class TopologyGraph:
    """主管 Agent 产出的完整物理拓扑有向图。

    在三个 Agent 间以内存数据流传递, 不序列化。
    支持序列化到 JSON 用于调试和持久化 (PipelineContext)。
    """

    model_name: str = ""                       # ASCII 安全的英文模型名
    user_request: str = ""                     # 原始用户需求文本
    system_description: str = ""               # 系统整体功能描述 (英文)
    physical_domains: list[str] = field(default_factory=list)
    nodes: list[TopologyNode] = field(default_factory=list)
    edges: list[TopologyEdge] = field(default_factory=list)
    subsystems: list[Subsystem] = field(default_factory=list)
    notes: str = ""                            # 额外建模指导

    # ---- 查询方法 ----

    @property
    def core_nodes(self) -> list[TopologyNode]:
        """返回所有核心节点 (is_core=True)。"""
        return [n for n in self.nodes if n.is_core]

    @property
    def non_core_nodes(self) -> list[TopologyNode]:
        """返回所有非核心节点。"""
        return [n for n in self.nodes if not n.is_core]

    def get_node(self, node_id: str) -> TopologyNode | None:
        """按 ID 查找节点。"""
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def get_edges_from(self, node_id: str) -> list[TopologyEdge]:
        """返回从指定节点出发的所有边。"""
        return [e for e in self.edges if e.source_node_id == node_id]

    def get_edges_to(self, node_id: str) -> list[TopologyEdge]:
        """返回指向指定节点的所有边。"""
        return [e for e in self.edges if e.target_node_id == node_id]

    def is_orphan(self, node_id: str) -> bool:
        """检查节点是否孤立 (无任何边连接)。"""
        has_out = any(e.source_node_id == node_id for e in self.edges)
        has_in = any(e.target_node_id == node_id for e in self.edges)
        return not has_out and not has_in

    @property
    def cross_domain_edges(self) -> list[TopologyEdge]:
        """返回所有跨域边。"""
        return [e for e in self.edges if e.is_cross_domain]

    @property
    def component_count(self) -> int:
        """估算所需元件数 (含 quantity 累加)。"""
        return sum(n.quantity for n in self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def complexity(self) -> str:
        """简单复杂度评估。"""
        ct = self.component_count
        if ct <= 4:
            return "basic"
        elif ct <= 12:
            return "intermediate"
        else:
            return "advanced"

    # ---- 序列化 ----

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "user_request": self.user_request,
            "system_description": self.system_description,
            "physical_domains": self.physical_domains,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "subsystems": [s.to_dict() for s in self.subsystems],
            "notes": self.notes,
            "complexity": self.complexity,
            "component_count": self.component_count,
            "edge_count": self.edge_count,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "TopologyGraph":
        return cls(
            model_name=d.get("model_name", ""),
            user_request=d.get("user_request", ""),
            system_description=d.get("system_description", ""),
            physical_domains=d.get("physical_domains", []),
            nodes=[TopologyNode.from_dict(n) for n in d.get("nodes", [])],
            edges=[TopologyEdge.from_dict(e) for e in d.get("edges", [])],
            subsystems=[Subsystem(**s) for s in d.get("subsystems", [])],
            notes=d.get("notes", ""),
        )

    @classmethod
    def from_json(cls, s: str) -> "TopologyGraph":
        return cls.from_dict(json.loads(s))

    # ---- 可视化 ----

    def to_mermaid(self) -> str:
        """生成 Mermaid 流程图源码。"""
        lines = ["graph LR"]
        for node in self.nodes:
            marker = "(( ))" if node.is_core else "[ ]"
            role_tag = f"{node.role}" if node.role else node.label
            domain_tag = node.domain.replace("_", " ")
            lines.append(
                f'    {node.node_id}{marker}"{node.label}<br/>{role_tag}<br/>{domain_tag}"'
            )
        for edge in self.edges:
            arrow = "==>" if edge.is_cross_domain else "-->"
            flow = edge.flow_type or edge.label
            lines.append(
                f"    {edge.source_node_id} {arrow}|{flow}| {edge.target_node_id}"
            )
        return "\n".join(lines)

    def summary(self) -> str:
        """人类可读的摘要。"""
        lines = [
            f"TopologyGraph: {self.model_name}",
            f"  Complexity: {self.complexity}",
            f"  Nodes: {len(self.nodes)} ({len(self.core_nodes)} core)",
            f"  Edges: {self.edge_count} ({len(self.cross_domain_edges)} cross-domain)",
            f"  Domains: {', '.join(self.physical_domains)}",
            f"  Subsystems: {len(self.subsystems)}",
        ]
        if self.notes:
            lines.append(f"  Notes: {self.notes}")
        return "\n".join(lines)


# ============================================================
# TorsionBar JSON 格式 (连接 Agent ↔ 参数运行 Agent)
# ============================================================

@dataclass
class ComponentEntry:
    """TorsionBar JSON 中的一个元件条目。"""
    icon_name: str
    alias: str
    position: tuple[int, int]
    submodel: str = ""
    submodel_path: str = ""
    rotations: int = 0
    flip: bool = False
    layer: int = 0                             # 所属扩展层 (0=核心)


@dataclass
class ConnectionEntry:
    """TorsionBar JSON 中的一个连接条目。"""
    from_component: str
    from_port: int
    to_component: str
    to_port: int
    type: str = "line"                         # "line" | "direct"
    line_alias: str = ""
    waypoints: list[tuple[int, int]] = field(default_factory=list)
    line_submodel: Optional[str] = None
    line_submodel_path: Optional[str] = None


@dataclass
class TorsionBarModel:
    """层层填充的完整模型定义 (TorsionBar 格式)。

    连接 Agent → 参数运行 Agent 之间的数据接口。
    保存为 JSON 文件用于迁移、缓存和对比。
    """

    model_name: str
    source: str = ""                           # 来源 demo 路径 (如果是克隆)
    components: list[ComponentEntry] = field(default_factory=list)
    connections: list[ConnectionEntry] = field(default_factory=list)
    bridge_components: list[dict] = field(default_factory=list)
    terminal_components: list[dict] = field(default_factory=list)
    non_default_params: dict[str, str] = field(default_factory=dict)  # {varname@alias: value}
    parameter_assignments: list[dict] = field(default_factory=list)   # 参数设置详情
    topology_graph_snapshot: Optional[dict] = None   # 来源拓扑图快照

    # ---- 查询 ----

    def get_component(self, alias: str) -> ComponentEntry | None:
        for c in self.components:
            if c.alias == alias:
                return c
        return None

    def is_complete(self) -> bool:
        """检查模型是否完整 (可构建)。"""
        return bool(self.model_name and self.components and self.connections)

    # ---- 序列化 (匹配 Solutions_max_TorsionBar.json) ----

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "model_name": self.model_name,
            "circuit": {
                "name": self.model_name,
                "components": [
                    {k: v for k, v in {
                        "icon_name": c.icon_name,
                        "alias": c.alias,
                        "position": list(c.position),
                        "submodel": c.submodel or None,
                        "submodel_path": c.submodel_path or None,
                        "rotations": c.rotations,
                        "flip": c.flip,
                    }.items() if v is not None}
                    for c in self.components
                ],
                "connections": [
                    {k: v for k, v in {
                        "from_component": c.from_component,
                        "from_port": c.from_port,
                        "to_component": c.to_component,
                        "to_port": c.to_port,
                        "type": c.type,
                        "line_alias": c.line_alias or None,
                        "waypoints": [list(w) for w in c.waypoints] if c.waypoints else None,
                        "line_submodel": c.line_submodel,
                        "line_submodel_path": c.line_submodel_path,
                    }.items() if v is not None}
                    for c in self.connections
                ],
            },
            "non_default_params": self.non_default_params,
            "bridge_components": self.bridge_components,
            "terminal_components": self.terminal_components,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, s: str) -> "TorsionBarModel":
        d = json.loads(s)
        circuit = d.get("circuit", {})
        return cls(
            model_name=d.get("model_name", circuit.get("name", "")),
            source=d.get("source", ""),
            components=[
                ComponentEntry(
                    icon_name=c["icon_name"],
                    alias=c["alias"],
                    position=tuple(c["position"]),
                    submodel=c.get("submodel", ""),
                    submodel_path=c.get("submodel_path", ""),
                    rotations=c.get("rotations", 0),
                    flip=c.get("flip", False),
                )
                for c in circuit.get("components", [])
            ],
            connections=[
                ConnectionEntry(
                    from_component=c["from_component"],
                    from_port=c["from_port"],
                    to_component=c["to_component"],
                    to_port=c["to_port"],
                    type=c.get("type", "line"),
                    line_alias=c.get("line_alias", ""),
                    waypoints=[tuple(w) for w in c.get("waypoints", [])] if c.get("waypoints") else [],
                    line_submodel=c.get("line_submodel"),
                    line_submodel_path=c.get("line_submodel_path"),
                )
                for c in circuit.get("connections", [])
            ],
            non_default_params=d.get("non_default_params", {}),
            bridge_components=d.get("bridge_components", []),
            terminal_components=d.get("terminal_components", []),
        )

    def summary(self) -> str:
        return (
            f"TorsionBarModel '{self.model_name}': "
            f"{len(self.components)} components, "
            f"{len(self.connections)} connections, "
            f"{len(self.non_default_params)} non-default params"
        )
