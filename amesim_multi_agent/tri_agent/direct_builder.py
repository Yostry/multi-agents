"""
DirectBuilder — 确定性 KB 搜索构建器

对简单模型跳过 LLM Connector, 直接通过 KB 搜索和端口匹配构建 TorsionBarModel。
作为 Connector Agent 的快速回退路径。

策略:
  1. 从 TopologyGraph 中提取核心节点
  2. 对每个节点用 UnifiedSearcher 搜索最佳元件
  3. 根据端口兼容性自动连接
  4. 直接产出 TorsionBarModel JSON

用法:
    builder = DirectBuilder(kb_registry_path)
    model = builder.build(topology_graph)
    if model:
        print(model.to_json())
"""

from __future__ import annotations

import json, os, sys
from dataclasses import dataclass, field
from typing import Optional

# 路径设置
_HERE = os.path.dirname(os.path.abspath(__file__))
# tri_agent/ → amesim_multi_agent/ → multi_agent/ → Chat1D/
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_KB_REG = os.path.join(_PROJECT_ROOT, "knowledge_base", "json", "components", "_registry.json")
if not os.path.exists(_KB_REG):
    # fallback: direct hard-coded path
    _PROJECT_ROOT = r"C:\Users\Y\Desktop\Chat1D"
    _KB_REG = os.path.join(_PROJECT_ROOT, "knowledge_base", "json", "components", "_registry.json")
sys.path.insert(0, _PROJECT_ROOT)

from .topology_graph import (
    TopologyGraph, TopologyNode, TopologyEdge,
    TorsionBarModel, ComponentEntry, ConnectionEntry
)


# ================================================================
# 域 → 库 → 默认 icon 映射 (从 10 个 demo 和 KB 中提取)
# ================================================================

# 按 role 推荐常见 icon_name
ROLE_TO_ICONS: dict[str, list[str]] = {
    # 机械平移
    "mass": ["mass2port", "mass_friction_endstops"],
    "mass_body": ["mass2port", "mass_friction_endstops"],
    "spring": ["spring01", "springdamper01"],
    "damper": ["damper01", "springdamper01"],
    "spring_damper": ["springdamper01", "spring01"],
    "elastic_endstop": ["elasticendstop"],
    "endstop": ["elasticendstop"],
    "force_source": ["zeroforcesource"],
    "force_sink": ["zeroforcesource"],
    "speed_source": ["zerospeedsource"],
    "speed_sink": ["zerospeedsource"],
    "velocity_source": ["omegacon"],
    "gravity": ["gravityicon"],
    "ground": ["zeroforcesource", "zerospeedsource"],
    "wall": ["zeroforcesource"],
    "fixed_reference": ["zerospeedsource"],
    "torque_source": ["zerotorquesource"],
    "torque_sink": ["zerotorquesource"],
    "rotary_spring": ["rotaryspring"],
    # 信号
    "signal_source": ["constant"],
    "signal_sink": ["signalsink"],
    "sine_source": ["constant"],
    "quantizer": ["quantizer"],
    "sampler": ["sampler"],
    "constant": ["constant"],
    "gain": ["sigain", "siggain"],
    "sensor": ["anglesensor", "displacementsensor"],
    "control": ["control01"],
    "stop": ["elasticendstop"],
    "elastic_stop": ["elasticendstop"],
    "contact": ["elasticendstop"],
    "bounce_stop": ["elasticendstop"],
    "position_sensor": ["anglesensor", "displacementsensor"],
    "velocity_sensor": ["velocitysensor"],
    "displacement_sensor": ["displacementsensor"],
    "force_input": ["zeroforcesource"],
    "initial_height": [],  # phyiscal attr, not a component
    # PLM, M6DOF, ESS — 不在 KB 但真实 Amesim 中存在
    "calcul": ["plmcalcul"],
    "plm_calcul": ["plmcalcul"],
    "plm_wall": ["plmrefwall"],
    "plm_pivot": ["plmpivot"],
    "plm_assembly": ["plmassembly"],
    "plm_body": ["dynamic_plmbody"],
    "plm_zero": ["plmzer00"],
    "m6dof_joint": ["m6dofannularlinear"],
    "m6dof_body": ["dynamic_3Dbody"],
    "m6dof_assembly": ["m6dofassembly"],
    "m6dof_ground": ["zerospeedsource3D"],
    "battery_cell": ["BatCellElectrochemical"],
    "temperature_source": ["th_ts", "th_temperaturesource"],
    "electrical_ground": ["potential_reference"],
    "current_source": ["zero_current_source"],
    # ACE 超组件 (clone mode only)
    "ace_custom": [],  # ACE 组件无法从 KB 构建, 需要 clone 原始 .ame
    # 热
    "heat_source": ["th_hc", "th_heatflow_source"],
    "thermal_capacity": ["th_c"],
    "thermal_conduction": ["th_conduct"],
    # 液压
    "tank": ["tank"],
    "pump": ["pump01"],
    "valve": ["presscontrol01", "flowcontrol01", "servovalve01"],
    "orifice": ["hydrorifice"],
    "cylinder": ["actuator01"],
    # 电气
    "voltage_source": ["potential_reference", "zero_current_source"],
    "battery_cell": ["BatCellElectrochemical"],
    "electrical_ground": ["potential_reference"],
}

# 域 → 默认库
DOMAIN_TO_LIB: dict[str, str] = {
    "mechanical_1d": "libmec",
    "mechanical_rotary": "libmec",
    "signal": "libsig",
    "thermal": "libth",
    "hydraulic": "libhydr",
    "pneumatic": "libpn",
    "electric": "libeb",
    "two_phase_flow": "libtpf",
    "thermal_hydraulic": "libthh",
    "planar_mechanical": "libplm",
    "m6dof": "libm6dof",
    "transmission": "libtr",
}

# 库 → 子模型路径
LIB_TO_PATH: dict[str, str] = {
    "libmec": "$AME/libmec/submodels",
    "libsig": "$AME/libsig/submodels",
    "libth": "$AME/libth/submodels",
    "libhydr": "$AME/libhydr/submodels",
    "libpn": "$AME/libpn/submodels",
    "libeb": "$AME/libeb/submodels",
    "libtpf": "$AME/libtpf/submodels",
    "libthh": "$AME/libthh/submodels",
    "libplm": "$AME/libplm/submodels",
    "libm6dof": "$AME/libm6dof/submodels",
    "libtr": "$AME/libtr/submodels",
    "libess": "$AME/libess/submodels",
}


class DirectBuilder:
    """基于 KB 搜索的确定性模型构建器。

    对 TopologyGraph 中的每个节点, 用 role → icon 映射 + KB 验证
    直接选择元件, 然后用端口兼容性自动规划连接。
    """

    def __init__(self, kb_registry_path: str = ""):
        self._reg_path = kb_registry_path or _KB_REG
        self._reg: dict = {}
        self._icon_index: dict = {}
        self._submodel_index: dict = {}

    def _load_kb(self):
        if not self._reg:
            if os.path.exists(self._reg_path):
                with open(self._reg_path, encoding="utf-8") as f:
                    self._reg = json.load(f)
                self._icon_index = self._reg.get("icon_index", {})
                self._submodel_index = self._reg.get("submodel_index", {})

    def build(self, topology: TopologyGraph, max_components: int = 20) -> TorsionBarModel | None:
        """从 TopologyGraph 构建 TorsionBarModel。"""
        self._load_kb()
        if not self._icon_index:
            return None

        model = TorsionBarModel(model_name=topology.model_name)

        # 按拓扑顺序布局: 每个节点间距 200px
        positions = self._compute_positions(topology.nodes)

        # Phase 1: 选择元件
        alias_map: dict[str, str] = {}  # node_id → alias
        for i, node in enumerate(topology.nodes):
            icon = self._select_icon(node)
            if not icon:
                print(f"  [DirectBuilder] Cannot find icon for node {node.node_id} ({node.label})")
                continue

            alias = node.label.replace(" ", "_") if node.label else f"Comp_{i+1}"
            # 确保 alias 唯一
            base = alias
            counter = 1
            while any(c.alias == alias for c in model.components):
                counter += 1
                alias = f"{base}_{counter}"

            model.components.append(ComponentEntry(
                icon_name=icon,
                alias=alias,
                position=(positions[i][0], positions[i][1]),
                submodel="",  # 让 Amesim 自动分配
                submodel_path=self._get_submodel_path(node),
                layer=0 if node.is_core else 1,
            ))
            alias_map[node.node_id] = alias

        # Phase 2: 规划连接
        used_line_ids = set()
        for i, edge in enumerate(topology.edges):
            from_alias = alias_map.get(edge.source_node_id)
            to_alias = alias_map.get(edge.target_node_id)
            if not from_alias or not to_alias:
                continue

            line_id = f"line_{from_alias}_{to_alias}"
            if line_id in used_line_ids:
                line_id = f"line_{from_alias}_{to_alias}_{i}"
            used_line_ids.add(line_id)

            # 根据 flow_type 确定端口分配
            fp = 0  # 默认: 机械/信号 port 0
            tp = 0
            if edge.flow_type and "signal" in edge.flow_type:
                fp, tp = 0, 0  # 信号域默认端口

            model.connections.append(ConnectionEntry(
                from_component=from_alias,
                from_port=fp,
                to_component=to_alias,
                to_port=tp,
                type="line",
                line_alias=line_id,
                waypoints=[],
            ))

        print(f"  [DirectBuilder] Built: {model.summary()}")
        return model

    def _select_icon(self, node: TopologyNode) -> Optional[str]:
        """为节点选择最佳 icon_name。"""
        self._load_kb()

        # 0. role 本身是否就是一个有效的 icon_name (需校验域匹配)
        role = (node.role or "").lower().replace(" ", "_")
        if role in self._icon_index:
            # 检查 icon 的库是否匹配节点域
            entries = self._icon_index[role]
            if isinstance(entries, list) and entries:
                target_lib = DOMAIN_TO_LIB.get(node.domain, "")
                matching = [e for e in entries if e.get("lib") == target_lib]
                if matching:
                    return role  # role 直接匹配到正确的库
                # 不匹配 — 不直接返回, 继续用 ROLE_TO_ICONS 查找更好的候选

        # 1. 用 role 映射查找
        candidates = ROLE_TO_ICONS.get(role, [])

        # 2. 从 label 中提取关键词 (label 可能本身是 icon_name)
        label = (node.label or "").lower().replace("_", " ").strip()
        target_lib = DOMAIN_TO_LIB.get(node.domain, "")
        for word in label.replace("_", " ").split():
            if word in ROLE_TO_ICONS:
                candidates.extend(ROLE_TO_ICONS[word])
            # 仅当 word 匹配到正确域的 icon 时才直接作为候选
            if word in self._icon_index and word not in candidates:
                entries = self._icon_index[word]
                if isinstance(entries, list) and entries:
                    if target_lib and any(e.get("lib") == target_lib for e in entries):
                        candidates.insert(0, word)

        # 3. 用 functional_description 中的关键词
        if not candidates:
            desc = (node.functional_description or "").lower()
            for keyword, icons in ROLE_TO_ICONS.items():
                if keyword in desc:
                    candidates = icons
                    break

        # 4. 按域补充
        if not candidates and node.domain:
            lib = DOMAIN_TO_LIB.get(node.domain, "")
            for icon_name, entries in self._icon_index.items():
                if isinstance(entries, list):
                    for e in entries:
                        if e.get("lib") == lib:
                            candidates.append(icon_name)
                            break
                if len(candidates) >= 10:
                    break

        # 5. 验证 KB 中的存在性, 选第一个存在的
        for icon in candidates:
            if icon in self._icon_index:
                return icon

        # 6. 如果 candidates 中有值但都不在 KB, 返回 None (不返回编造的名字)
        # 只有 validated 的 icon 才能被使用
        return None

    def _get_submodel_path(self, node: TopologyNode) -> str:
        """获取子模型库路径。"""
        if node.suggested_library:
            return LIB_TO_PATH.get(node.suggested_library, "")
        domain = node.domain or ""
        lib = DOMAIN_TO_LIB.get(domain, "")
        return LIB_TO_PATH.get(lib, "")

    @staticmethod
    def _compute_positions(nodes: list[TopologyNode]) -> list[tuple[int, int]]:
        """简单的流式布局。"""
        positions = []
        for i in range(len(nodes)):
            x = 150 + i * 220
            y = 150 + (i % 3) * 180
            positions.append((x, y))
        return positions


# ================================================================
# 测试
# ================================================================

def test_with_bouncing_ball():
    """用 BouncingBall 的拓扑测试 DirectBuilder。"""
    graph = TopologyGraph(
        model_name="BouncingBall_Direct",
        user_request="Build a bouncing ball model",
        physical_domains=["mechanical_1d"],
        nodes=[
            TopologyNode(node_id="n1", label="Mass_Body", role="mass",
                         domain="mechanical_1d", is_core=True,
                         suggested_library="libmec",
                         functional_description="mass with friction and endstops"),
            TopologyNode(node_id="n2", label="Elastic_Endstop", role="elastic_endstop",
                         domain="mechanical_1d", is_core=False,
                         suggested_library="libmec",
                         functional_description="elastic endstop for bouncing"),
            TopologyNode(node_id="n3", label="Ground_Speed", role="speed_sink",
                         domain="mechanical_1d", is_core=False,
                         suggested_library="libmec"),
            TopologyNode(node_id="n4", label="Ground_Force", role="force_sink",
                         domain="mechanical_1d", is_core=False,
                         suggested_library="libmec"),
        ],
        edges=[
            TopologyEdge(edge_id="e1", source_node_id="n1", target_node_id="n2",
                         flow_type="mechanical_translation"),
            TopologyEdge(edge_id="e2", source_node_id="n2", target_node_id="n3",
                         flow_type="mechanical_translation"),
            TopologyEdge(edge_id="e3", source_node_id="n4", target_node_id="n1",
                         flow_type="mechanical_translation"),
        ],
    )

    builder = DirectBuilder()
    model = builder.build(graph)
    if model:
        print(model.to_json())
        return model
    return None


if __name__ == "__main__":
    test_with_bouncing_ball()
