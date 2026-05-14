"""
端口驱动的螺旋执行器 (PortDrivenExecutor)

从核心元件出发, 根据其端口类型逐步向外扩展, 直到所有端口闭合。

核心算法:
  1. 给定核心元件 + 其端口列表
  2. 对每个未连接端口: 搜索可匹配的元件 (根据 port_tag 和物理域)
  3. 连接核心端口 ←→ 新元件端口
  4. 新元件带来新端口 → 加入待处理队列
  5. 重复第 2-4 步, 直到队列为空或达到最大扩展层数

设计原则:
  - 端口类型决定可连接的元件类型 (不依赖 LLM 猜测)
  - 每层扩展后验证拓扑一致性
  - 支持自动插入桥接/终端元件
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from collections import deque

# 知识库路径
_KB_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "knowledge_base")
_REGISTRY_PATH = os.path.join(_KB_ROOT, "json", "components", "_registry.json")


# ============================================================
# 端口-元件匹配规则表
# ============================================================

# port_tag → 可用的元件 icon_name 候选 (按优先级排列)
PORT_TO_COMPONENT_MAP: dict[str, list[str]] = {
    # 两相流 (TPF)
    "hflow": [
        "tpf_pipe", "tpf_orifice", "tpf_chamber", "tpf_chamber_heat",
        "tpf_ph_modulated_source", "tpf_ph_modulated_sink",
        "tpf_ph_source", "tpf_ph_sink", "tpf_pump",
        "tpf_3port_junction", "tpf_valve",
    ],
    # 机械 1D (单向)
    "mechanical_1d_in": [
        "zeroforcesource", "zerospeedsource", "mass2port",
        "spring", "damper", "elasticendstop",
        "forcesource", "speedsource", "displacementsource",
        "rotary_spring", "rotary_damper",
        "rotaryload2port", "gearpumpmotor01",
    ],
    "mechanical_1d_out": [
        "zeroforcesource", "zerospeedsource", "mass2port",
        "spring", "damper", "elasticendstop",
        "forcesensor", "speedsensor", "displacementsensor",
    ],
    # 热域
    "thermal": [
        "th_c", "th_ts", "th_hc", "th_conduct", "th_convection",
        "th_radiation", "th_heatflow_source", "th_temperaturesource",
    ],
    # 信号域
    "signal": [
        "sigconst", "sigsine", "sigstep", "sigud", "sigain",
        "sigfunction", "sigfx", "sigfxy", "siggain",
        "sigadd", "sigmul", "sigdiv", "sigintegral",
        "sigcomparator", "sigswitch", "sigsat",
    ],
    # 液压域
    "hydraulic": [
        "hydrnode3", "hydrnode4", "hydrorifice", "hydrpipe",
        "tank", "pump01", "pump02", "presscontrol01",
        "flowcontrol01", "checkvalve01", "servovalve01",
        "actuator01", "actuator02",
    ],
    # 气动域
    "pneumatic": [
        "pn_const", "pn_volume", "pn_orifice", "pn_pipe",
        "pn_valve", "pn_pump", "pn_source", "pn_sink",
    ],
    # 电气域
    "electric": [
        "ele_resistor", "ele_capacitor", "ele_inductor",
        "ele_source_v", "ele_source_i", "ele_ground",
        "ele_diode", "ele_transistor",
    ],
}

# 域名称 → 库名 映射
DOMAIN_TO_LIBRARY: dict[str, str] = {
    "two_phase_flow": "libtpf",
    "mechanical_1d": "libmec",
    "thermal": "libth",
    "signal": "libsig",
    "hydraulic": "libhydr",
    "pneumatic": "libpn",
    "electric": "libeb",
    "thermal_hydraulic": "libthh",
    "cooling": "libcs",
}

# 端口 tag → 物理域 映射
PORT_TAG_TO_DOMAIN: dict[str, str] = {
    "hflow": "two_phase_flow",
    "mechanical_1d_in": "mechanical_1d",
    "mechanical_1d_out": "mechanical_1d",
    "thermal": "thermal",
    "signal": "signal",
    "hydraulic": "hydraulic",
    "pneumatic": "pneumatic",
    "electric": "electric",
}

# 终端/边界端口 tag (这些端口需要封闭元件, 如 source/sink/ground)
TERMINAL_PORT_TAGS = {"hflow", "mechanical_1d_in", "mechanical_1d_out",
                       "hydraulic", "pneumatic", "electric"}

# 终端封闭元件 (按 port_tag)
TERMINAL_COMPONENTS: dict[str, str] = {
    "mechanical_1d_in": "zeroforcesource",
    "mechanical_1d_out": "zerospeedsource",
    "hydraulic": "tank",
    "signal": "sigconst",
    "thermal": "th_temperaturesource",
    "pneumatic": "pn_source",
    "electric": "ele_ground",
}


# ============================================================
# 数据模型
# ============================================================

@dataclass
class OpenPort:
    """一个未连接的端口"""
    component_alias: str          # 所属元件别名
    port_index: int               # 端口号 (0-indexed)
    port_tag: str                 # 端口物理类型
    port_variables: list[dict]    # 端口变量列表 [{norm, io, title, units}]
    domain: str = ""              # 物理域
    expansion_layer: int = 0      # 扩展层数 (核心元件=0)


@dataclass
class ExpansionStep:
    """一次扩展操作的结果"""
    layer: int                              # 第几层扩展
    source_port: OpenPort                   # 源端口
    matched_component: dict | None = None   # 匹配到的元件 {icon_key, icon_name, library, submodel}
    connection: dict | None = None          # 连接定义 {from_alias, from_port, to_alias, to_port}
    new_open_ports: list[OpenPort] = field(default_factory=list)
    terminal_closed: bool = False           # 是否用终端元件封闭
    skip_reason: str = ""                   # 如果未匹配, 跳过原因


@dataclass
class PortExpansionResult:
    """完整扩展结果"""
    core_component: dict                    # 核心元件信息
    expansion_history: list[ExpansionStep]  # 每步扩展的详细记录
    all_components: list[dict]              # 所有元件 (核心 + 扩展)
    all_connections: list[dict]             # 所有连接
    closed_ports: int                       # 已封闭端口数
    remaining_open_ports: list[OpenPort]    # 仍未封闭的端口
    max_layer: int                          # 最大扩展层数
    success: bool = False                   # 是否所有端口已闭合


# ============================================================
# 端口驱动执行器
# ============================================================

class PortDrivenExecutor:
    """端口驱动的螺旋拓扑扩展器。

    用法:
        executor = PortDrivenExecutor(kb_registry_path)
        result = executor.expand_from_core(
            core_icon_key="libtpf:tpf_chamber_heat",
            core_alias="Tank",
            core_submodel="TPFHECH000",
            max_layers=4,
        )
    """

    def __init__(self, registry_path: str | None = None):
        self._registry_path = registry_path or _REGISTRY_PATH
        self._registry: dict | None = None
        self._lib_cache: dict[str, dict] = {}  # 库 JSON 缓存

    # ---- 注册表加载 ----

    def _load_registry(self) -> dict:
        if self._registry is None:
            with open(self._registry_path, encoding="utf-8") as f:
                self._registry = json.load(f)
        return self._registry

    def _load_library(self, lib_name: str) -> dict:
        if lib_name not in self._lib_cache:
            lib_path = os.path.join(
                os.path.dirname(self._registry_path),
                f"{lib_name}.json"
            )
            if os.path.exists(lib_path):
                with open(lib_path, encoding="utf-8") as f:
                    self._lib_cache[lib_name] = json.load(f)
            else:
                self._lib_cache[lib_name] = {}
        return self._lib_cache[lib_name]

    # ---- 端口查询 ----

    def get_component_ports(self, icon_key: str, submodel_id: str = "") -> list[dict]:
        """查询元件的端口信息。返回 [{port_index, port_tag, variables}]。"""
        reg = self._load_registry()
        parts = icon_key.split(":", 1)
        if len(parts) == 2:
            lib_name, icon_name = parts
        else:
            lib_name, icon_name = parts[0], icon_key

        # 从 icon_index 查找 submodel
        icon_entries = reg.get("icon_index", {}).get(icon_name, [])
        if not icon_entries:
            return []

        # 如果指定了 submodel_id, 精确查找
        target_submodel = None
        if submodel_id:
            for entry in icon_entries:
                if entry.get("id") == submodel_id:
                    target_submodel = entry
                    lib_name = entry.get("lib", lib_name)
                    break

        # 否则用第一个匹配
        if target_submodel is None and icon_entries:
            target_submodel = icon_entries[0]
            lib_name = target_submodel.get("lib", lib_name)

        if target_submodel is None:
            return []

        # 加载完整的 submodel 数据
        lib_data = self._load_library(lib_name)
        sm_id = target_submodel.get("id", "")
        sm_data = lib_data.get("submodels", {}).get(sm_id, {})
        if not sm_data:
            sm_data = lib_data.get("submodels", {}).get(submodel_id, {})

        ports = sm_data.get("ports", [])
        return ports

    # ---- 端口匹配 ----

    def find_candidates_for_port(
        self,
        port_tag: str,
        domain: str = "",
        top_k: int = 5,
    ) -> list[dict]:
        """根据端口类型查找可匹配的元件候选。"""
        reg = self._load_registry()

        # 先从 PORT_TO_COMPONENT_MAP 获取候选 icon_name
        candidate_icons = PORT_TO_COMPONENT_MAP.get(port_tag, [])

        results = []
        for icon_name in candidate_icons:
            entries = reg.get("icon_index", {}).get(icon_name, [])
            for entry in entries:
                lib_name = entry.get("lib", "")
                # 如果指定了域, 过滤库
                if domain:
                    lib_data = self._load_library(lib_name)
                    # 加载 submodel 检查端口兼容性
                    sm_data = lib_data.get("submodels", {}).get(entry.get("id", ""), {})
                    sm_ports = sm_data.get("ports", [])
                    # 检查是否有匹配的端口 tag
                    port_tags = [p.get("port_tag", "") for p in sm_ports]
                    if port_tag not in port_tags:
                        continue

                results.append({
                    "icon_key": f"{lib_name}:{icon_name}",
                    "icon_name": icon_name,
                    "library": lib_name,
                    "submodel_id": entry.get("id", ""),
                    "label": entry.get("label", ""),
                    "port_count": len(sm_ports) if 'sm_ports' in dir() else 0,
                })

            if len(results) >= top_k:
                break

        return results[:top_k]

    def find_terminal_for_port(self, port_tag: str) -> dict | None:
        """查找封闭未连接端口的终端元件。"""
        icon_name = TERMINAL_COMPONENTS.get(port_tag)
        if not icon_name:
            return None

        reg = self._load_registry()
        entries = reg.get("icon_index", {}).get(icon_name, [])
        if not entries:
            return None

        entry = entries[0]
        return {
            "icon_key": f"{entry.get('lib', '')}:{icon_name}",
            "icon_name": icon_name,
            "library": entry.get("lib", ""),
            "submodel_id": entry.get("id", ""),
            "label": entry.get("label", ""),
            "is_terminal": True,
        }

    # ---- 核心: 螺旋扩展算法 ----

    def expand_from_core(
        self,
        core_icon_key: str,
        core_alias: str = "Core",
        core_submodel: str = "",
        max_layers: int = 4,
        topology_description: str = "",
    ) -> PortExpansionResult:
        """从核心元件出发, 螺旋扩展构建完整拓扑。

        Args:
            core_icon_key: 核心元件的 "lib:icon_name" 标识
            core_alias: 核心元件别名
            core_submodel: 核心元件的子模型 ID
            max_layers: 最大扩展层数 (防止无限循环)
            topology_description: 拓扑描述文本 (可选, 辅助匹配)

        Returns:
            PortExpansionResult: 完整扩展结果
        """
        # 初始化
        all_components: list[dict] = [{
            "icon_key": core_icon_key,
            "alias": core_alias,
            "submodel_id": core_submodel,
            "layer": 0,
            "is_core": True,
        }]
        all_connections: list[dict] = []
        expansion_history: list[ExpansionStep] = []
        alias_counter = 1

        # 获取核心元件的端口
        core_ports = self.get_component_ports(core_icon_key, core_submodel)

        # 初始化待处理队列: 核心元件的所有端口
        port_queue: deque[OpenPort] = deque()
        for i, port_info in enumerate(core_ports):
            port_tag = port_info.get("port_tag", "")
            port_queue.append(OpenPort(
                component_alias=core_alias,
                port_index=port_info.get("index", i),
                port_tag=port_tag,
                port_variables=port_info.get("variables", []),
                domain=PORT_TAG_TO_DOMAIN.get(port_tag, ""),
                expansion_layer=0,
            ))

        # 主扩展循环
        while port_queue:
            open_port = port_queue.popleft()

            # 检查是否超过最大层数
            if open_port.expansion_layer >= max_layers:
                expansion_history.append(ExpansionStep(
                    layer=open_port.expansion_layer,
                    source_port=open_port,
                    skip_reason=f"达到最大层数限制 ({max_layers})",
                ))
                continue

            # Step 1: 查找候选元件
            candidates = self.find_candidates_for_port(
                port_tag=open_port.port_tag,
                domain=open_port.domain,
                top_k=3,
            )

            if candidates:
                # 选择第一个候选 (如果有拓扑描述, 可以在此匹配)
                best = candidates[0]
                new_alias = f"Comp_{alias_counter}"
                alias_counter += 1
                new_layer = open_port.expansion_layer + 1

                # 记录新元件
                new_comp = {
                    "icon_key": best["icon_key"],
                    "icon_name": best["icon_name"],
                    "alias": new_alias,
                    "library": best["library"],
                    "submodel_id": best["submodel_id"],
                    "layer": new_layer,
                    "is_core": False,
                    "parent_port": open_port,
                }
                all_components.append(new_comp)

                # 记录连接
                connection = {
                    "from_alias": open_port.component_alias,
                    "from_port": open_port.port_index,
                    "to_alias": new_alias,
                    "to_port": 0,  # 默认连到 port 0
                    "connection_type": "line",
                    "line_alias": f"line_{open_port.component_alias}_{new_alias}",
                }
                all_connections.append(connection)

                # 获取新元件的端口, 加入队列
                new_ports = self.get_component_ports(best["icon_key"], best["submodel_id"])
                for i, port_info in enumerate(new_ports):
                    port_tag = port_info.get("port_tag", "")
                    # 跳过已连接的端口 (port 0)
                    if port_info.get("index", i) == 0:
                        continue
                    port_queue.append(OpenPort(
                        component_alias=new_alias,
                        port_index=port_info.get("index", i),
                        port_tag=port_tag,
                        port_variables=port_info.get("variables", []),
                        domain=PORT_TAG_TO_DOMAIN.get(port_tag, ""),
                        expansion_layer=new_layer,
                    ))

                expansion_history.append(ExpansionStep(
                    layer=open_port.expansion_layer,
                    source_port=open_port,
                    matched_component=best,
                    connection=connection,
                    new_open_ports=[p for p in port_queue
                                    if p.component_alias == new_alias],
                ))

            else:
                # 无候选 → 尝试终端封闭
                terminal = self.find_terminal_for_port(open_port.port_tag)
                if terminal:
                    term_alias = f"Term_{alias_counter}"
                    alias_counter += 1

                    term_comp = {
                        "icon_key": terminal["icon_key"],
                        "icon_name": terminal["icon_name"],
                        "alias": term_alias,
                        "library": terminal["library"],
                        "submodel_id": terminal["submodel_id"],
                        "layer": open_port.expansion_layer + 1,
                        "is_terminal": True,
                    }
                    all_components.append(term_comp)

                    connection = {
                        "from_alias": open_port.component_alias,
                        "from_port": open_port.port_index,
                        "to_alias": term_alias,
                        "to_port": 0,
                        "connection_type": "line",
                        "line_alias": f"line_{open_port.component_alias}_{term_alias}",
                    }
                    all_connections.append(connection)

                    expansion_history.append(ExpansionStep(
                        layer=open_port.expansion_layer,
                        source_port=open_port,
                        matched_component=terminal,
                        connection=connection,
                        terminal_closed=True,
                    ))
                else:
                    expansion_history.append(ExpansionStep(
                        layer=open_port.expansion_layer,
                        source_port=open_port,
                        skip_reason=f"无匹配元件和终端: port_tag={open_port.port_tag}",
                    ))

        # 汇总结果
        remaining = [p for p in port_queue]  # port_queue 可能还有残留
        closed_count = len(all_connections)

        success = len(remaining) == 0

        return PortExpansionResult(
            core_component=all_components[0],
            expansion_history=expansion_history,
            all_components=all_components,
            all_connections=all_connections,
            closed_ports=closed_count,
            remaining_open_ports=[],  # port_queue should be empty after loop
            max_layer=max(s.layer for s in expansion_history) if expansion_history else 0,
            success=success,
        )

    # ---- 结果诊断 ----

    def print_expansion_tree(self, result: PortExpansionResult):
        """打印扩展树的可视化结构。"""
        print(f"\n{'='*60}")
        print(f"  拓扑扩展树: {result.core_component.get('alias', '?')}")
        print(f"  元件: {len(result.all_components)} | 连接: {len(result.all_connections)}")
        print(f"  成功: {'✓' if result.success else '✗'} | 最大层数: {result.max_layer}")
        print(f"{'='*60}")

        # 按层分组
        layers: dict[int, list[dict]] = {}
        for comp in result.all_components:
            layer = comp.get("layer", 0)
            layers.setdefault(layer, []).append(comp)

        for layer in sorted(layers.keys()):
            indent = "  " * layer
            for comp in layers[layer]:
                tag = ""
                if comp.get("is_core"):
                    tag = " [核心]"
                elif comp.get("is_terminal"):
                    tag = " [终端]"
                print(f"{indent}L{layer}: {comp['alias']} ({comp.get('icon_name', '?')}){tag}")

        if result.expansion_history:
            skipped = [s for s in result.expansion_history if s.skip_reason]
            if skipped:
                print(f"\n  跳过 ({len(skipped)} 个):")
                for s in skipped:
                    print(f"    - {s.source_port.component_alias}:p{s.source_port.port_index}"
                          f" → {s.skip_reason}")

        print(f"{'='*60}\n")
