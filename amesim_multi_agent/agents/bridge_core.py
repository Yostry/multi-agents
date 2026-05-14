"""
Bridge Core — 纯代码连接规划引擎

零 LLM 参与。根据拓扑描述、元件列表和端口数据，自动规划所有连接。

流程:
  1. parse_topology_pairs() — 解析拓扑文本为连接对
  2. verify_components_exist() — 验证元件在数据库中
  3. plan_connections() — 逐对匹配兼容端口
  4. detect_and_resolve_dangling() — 悬空端口终端补全
  5. run_bridge_pipeline() — 完整流程编排

输出: (connections_list, bridge_components_list, terminal_components_list)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# 导入现有工具（纯代码版本）
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from amesim_builder.unified_search import get_searcher


# ============================================================
# 结果数据类
# ============================================================

@dataclass
class BridgePlanResult:
    """Bridge 阶段的完整产出"""
    connections: list[dict] = field(default_factory=list)
    bridge_components: list[dict] = field(default_factory=list)
    terminal_components: list[dict] = field(default_factory=list)
    validation_issues: list[dict] = field(default_factory=list)
    is_valid: bool = True
    dangling_ports: list[dict] = field(default_factory=list)


# ============================================================
# Step 1: 拓扑解析
# ============================================================

def parse_topology_pairs(topology_description: str, component_aliases: list[str]) -> list[tuple[str, str]]:
    """解析拓扑描述文本，提取元件之间的连接对。

    支持格式:
      - "A → B → C"              → [("A","B"), ("B","C")]
      - "A → B, 同时 C → D"      → [("A","B"), ("C","D")]
      - "Pump出口 → ReliefValve入口 + DirectionalValve入口"  → 需要 LLM 辅助处理分支

    Args:
        topology_description: 拓扑关系文字描述
        component_aliases: 已知元件别名列表（用于过滤和匹配）

    Returns:
        连接对列表 [(from_alias, to_alias), ...]
    """
    pairs = []
    aliases_set = {a.lower() for a in component_aliases if a}

    # 策略1: 解析 "A → B" 或 "A -> B" 或 "A -- B" 格式 (split方案支持链式)
    arrow_pattern = re.compile(r'(?:→|->|--|—)')
    sep_pattern = re.compile(r'[;；]')
    segments = sep_pattern.split(topology_description)
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        names = [n.strip() for n in arrow_pattern.split(seg) if n.strip()]
        for i in range(len(names) - 1):
            from_alias = _match_alias(names[i], aliases_set, component_aliases)
            to_alias = _match_alias(names[i + 1], aliases_set, component_aliases)
            if from_alias and to_alias and from_alias != to_alias:
                pair = (from_alias, to_alias)
                if pair not in pairs:
                    pairs.append(pair)

    # 策略1.5: 解析中文连接词 "连接到"、"流入"、"输入" 等格式
    #   "A连接到B" → (A, B)
    if not pairs:
        cn_conn_pattern = re.compile(
            r'(\w[\w\s]*?)\s*(?:的两相流端口|的热端口|的信号端口|的出口|的入口)?'
            r'(?:连接到|流入|输出到|输送到|传递给|输入到|进入|排出到|排向)\s*'
            r'(\w[\w\s]*\w)'
        )
        for m in cn_conn_pattern.finditer(topology_description):
            from_name = m.group(1).strip()
            to_name = m.group(2).strip()
            from_alias = _match_alias(from_name, aliases_set, component_aliases)
            to_alias = _match_alias(to_name, aliases_set, component_aliases)
            if from_alias and to_alias and from_alias != to_alias:
                pair = (from_alias, to_alias)
                if pair not in pairs:
                    pairs.append(pair)

    # 策略2: 解析逗号分隔的序列 "A, B, C, D" → [(A,B), (B,C), (C,D)]
    if not pairs:
        # 尝试提取所有大写开头的词作为元件名
        name_pattern = re.compile(r'\b([A-Z][a-zA-Z_]*)\b')
        names = name_pattern.findall(topology_description)
        seen = []
        for n in names:
            if n not in seen and _match_alias(n, aliases_set, component_aliases):
                seen.append(n)
        for i in range(len(seen) - 1):
            pairs.append((seen[i], seen[i + 1]))

    return pairs


def _match_alias(name: str, aliases_lower: set[str], original_aliases: list[str]) -> Optional[str]:
    """将拓扑文本中的名称匹配到已知元件别名。"""
    name_lower = name.lower().strip()
    if name_lower in aliases_lower:
        for orig in original_aliases:
            if orig.lower() == name_lower:
                return orig
    # 模糊匹配: 名称包含别名或别名包含名称
    for orig in original_aliases:
        orig_lower = orig.lower()
        if orig_lower in name_lower or name_lower in orig_lower:
            return orig
    return None


# ============================================================
# Step 2: 元件存在性验证 (复用 bridge_tools 的验证逻辑)
# ============================================================

def verify_components_exist(components: list[dict]) -> dict:
    """验证所有元件是否在知识库中存在。

    Args:
        components: [{"alias":..., "icon_key":..., "icon_name":..., "library":...}, ...]

    Returns:
        {"all_exist": bool, "missing": [...], "verified": [...]}
    """
    from pathlib import Path
    comp_idx_path = (Path(__file__).resolve().parent.parent.parent.parent /
                     "knowledge_base" / "json" / "index" / "component_index.json")
    with comp_idx_path.open("r", encoding="utf-8") as f:
        comp_idx = json.load(f)

    verified = []
    missing = []

    for comp in components:
        icon_key = comp.get("icon_key", "")
        alias = comp.get("alias", "?")
        entry = comp_idx.get(icon_key)
        if entry:
            verified.append({
                "alias": alias,
                "icon_key": icon_key,
                "exists": True,
                "library": entry.get("library", ""),
            })
        else:
            missing.append({
                "alias": alias,
                "icon_key": icon_key,
                "exists": False,
                "reason": f"在 component_index 中未找到 {icon_key}",
            })

    return {
        "all_exist": len(missing) == 0,
        "verified": verified,
        "missing": missing,
    }


# ============================================================
# 端口 IO 方向判断 (模块级工具函数)
# ============================================================

def _get_port_io(port: dict) -> str:
    """获取端口的 io 方向。'1'=output, '2'=input, '3'=bidirectional, ''=unknown"""
    variables = port.get("variables", [])
    io_values = {v.get("io", "") for v in variables}
    if "3" in io_values or ("1" in io_values and "2" in io_values):
        return "3"  # bidirectional
    if "1" in io_values:
        return "1"  # output
    if "2" in io_values:
        return "2"  # input
    return ""


# ============================================================
# Step 3: 端口匹配与连接规划
# ============================================================

def plan_connections(
    topology_pairs: list[tuple[str, str]],
    component_ports: dict[str, list[dict]],
    component_libraries: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    """@deprecated: Only used as fallback in _run_bridge_stage().

    根据拓扑对和端口数据，自动规划连接。

    对每个拓扑对 (A, B):
    1. 获取 A 和 B 的所有可用端口
    2. 过滤已占用的端口
    3. 查找同域端口对 (port_tag 相同)
    4. 匹配 io 方向: output(1) → input(2)
    5. 生成连接

    Args:
        topology_pairs: 拓扑连接对列表
        component_ports: {alias: [port_dict, ...]}  端口数据
        component_libraries: {alias: library_name}

    Returns:
        (connections_list, issues_list)
    """
    connections = []
    issues = []
    used_ports: dict[str, set[int]] = defaultdict(set)

    def _is_port_used(alias: str, port_idx: int) -> bool:
        return port_idx in used_ports.get(alias, set())

    def _mark_port_used(alias: str, port_idx: int):
        used_ports[alias].add(port_idx)

    for from_alias, to_alias in topology_pairs:
        suggestion = plan_single_connection(
            from_alias, to_alias, component_ports, used_ports,
        )

        if not suggestion["has_compatible_pair"]:
            from_has = bool(component_ports.get(from_alias))
            to_has = bool(component_ports.get(to_alias))
            if not from_has:
                issues.append({
                    "level": "warning",
                    "code": "NO_PORTS_INFO",
                    "message": f"元件 '{from_alias}' 无端口信息",
                })
            elif not to_has:
                issues.append({
                    "level": "warning",
                    "code": "NO_PORTS_INFO",
                    "message": f"元件 '{to_alias}' 无端口信息",
                })
            else:
                issues.append({
                    "level": "warning",
                    "code": "NO_COMPATIBLE_PORT_PAIR",
                    "message": f"'{from_alias}' 与 '{to_alias}' 之间未找到兼容端口对",
                })
            continue

        fp_idx = suggestion["suggested_from_port"]
        tp_idx = suggestion["suggested_to_port"]

        connections.append({
            "from_alias": from_alias,
            "from_port": fp_idx,
            "to_alias": to_alias,
            "to_port": tp_idx,
            "type": "line",
            "line_alias": f"wire_{from_alias}_{to_alias}",
            "port_type_from": suggestion["from_port_details"].get("port_tag", ""),
            "port_type_to": suggestion["to_port_details"].get("port_tag", ""),
        })
        _mark_port_used(from_alias, fp_idx)
        _mark_port_used(to_alias, tp_idx)

    return connections, issues


# ============================================================
# Step 3.5: 逐对连接 — plan_single_connection()
# ============================================================

def plan_single_connection(
    from_alias: str,
    to_alias: str,
    component_ports: dict[str, list[dict]],
    used_ports: dict[str, set[int]] | None = None,
) -> dict:
    """@deprecated: Scoring replaced by batch LLM review in format_batch_review_prompt().

    对单个拓扑对进行代码端口匹配，生成建议供 LLM 审核。

    这是"代码做查算"的核心——纯代码判断哪些端口应该连接。
    LLM 随后审核此建议以决定: DIRECT / JUNCTION / BRIDGE。

    Args:
        from_alias: 源元件别名
        to_alias: 目标元件别名
        component_ports: {alias: [port_dict, ...]}
        used_ports: 已占用端口集合 {alias: {port_index, ...}}

    Returns:
        {
            "from_alias": str,
            "to_alias": str,
            "has_compatible_pair": bool,
            "suggested_from_port": int,
            "suggested_to_port": int,
            "same_domain": bool,
            "score": int,
            "from_port_details": dict,   # port_tag, io_direction, variables
            "to_port_details": dict,
            "from_ports_total": int,
            "to_ports_total": int,
        }
    """
    if used_ports is None:
        used_ports = {}

    from_ports = component_ports.get(from_alias, [])
    to_ports = component_ports.get(to_alias, [])

    result: dict = {
        "from_alias": from_alias,
        "to_alias": to_alias,
        "has_compatible_pair": False,
        "suggested_from_port": -1,
        "suggested_to_port": -1,
        "same_domain": False,
        "score": 0,
        "from_port_details": {},
        "to_port_details": {},
        "from_ports_total": len(from_ports),
        "to_ports_total": len(to_ports),
    }

    if not from_ports or not to_ports:
        return result

    def _is_used(alias: str, port_idx: int) -> bool:
        """检查端口是否已被占用。
        
        bidirectional (io=3) 端口允许多重连接（如 1-port mass 可并联多个力元件）。
        """
        return port_idx in used_ports.get(alias, set())

    def _is_port_reusable(port: dict) -> bool:
        """判断端口是否可被复用（允许多重连接）。"""
        io_val = _get_port_io(port)
        return io_val == "3"  # bidirectional 端口允许多重连接

    best_match = None
    best_score = -1

    for fp in from_ports:
        fp_idx = fp.get("port_index", -1)
        if _is_used(from_alias, fp_idx) and not _is_port_reusable(fp):
            continue
        fp_tag = fp.get("port_tag", "")
        fp_io = _get_port_io(fp)

        for tp in to_ports:
            tp_idx = tp.get("port_index", -1)
            if _is_used(to_alias, tp_idx) and not _is_port_reusable(tp):
                continue
            tp_tag = tp.get("port_tag", "")
            tp_io = _get_port_io(tp)

            score = 0
            if fp_tag and tp_tag and fp_tag == tp_tag:
                score += 10  # 同域优先
            if fp_io == "1" and tp_io == "2":
                score += 5
            elif fp_io == "2" and tp_io == "1":
                score += 5
            elif fp_io == "3" or tp_io == "3":
                score += 2
            elif not fp_io or not tp_io:
                score += 1

            if score > best_score:
                best_score = score
                best_match = (fp, tp)

    if best_match:
        fp, tp = best_match
        result["has_compatible_pair"] = True
        result["suggested_from_port"] = fp.get("port_index", -1)
        result["suggested_to_port"] = tp.get("port_index", -1)
        result["same_domain"] = (fp.get("port_tag", "") == tp.get("port_tag", ""))
        result["score"] = best_score
        result["from_port_details"] = {
            "port_index": fp.get("port_index"),
            "port_tag": fp.get("port_tag", ""),
            "io_direction": _get_port_io(fp),
            "variables": [
                {"norm": v.get("norm"), "io": v.get("io"),
                 "title": v.get("title"), "units": v.get("units")}
                for v in fp.get("variables", [])
            ],
        }
        result["to_port_details"] = {
            "port_index": tp.get("port_index"),
            "port_tag": tp.get("port_tag", ""),
            "io_direction": _get_port_io(tp),
            "variables": [
                {"norm": v.get("norm"), "io": v.get("io"),
                 "title": v.get("title"), "units": v.get("units")}
                for v in tp.get("variables", [])
            ],
        }

    return result


# ============================================================
# Step 3.6: LLM 审核提示词 — 逐对连接审核
# ============================================================

def format_connection_review_prompt(
    suggestion: dict,
    pair_index: int,
    total_pairs: int,
    topology_description: str,
    all_component_data: list[dict],
) -> str:
    """@deprecated: Replaced by format_batch_review_prompt(). Remove after Task 6.

    为单个连接对生成 LLM 审核提示词。

    LLM 需要决定: DIRECT（直连）/ JUNCTION（需三通/分配器）/ BRIDGE（需桥接元件）。

    Args:
        suggestion: plan_single_connection() 的返回结果
        pair_index: 当前连接对序号 (0-based)
        total_pairs: 总连接对数
        topology_description: 完整拓扑描述
        all_component_data: 所有元件数据列表

    Returns:
        LLM prompt 字符串
    """
    import json as _json

    from_alias = suggestion["from_alias"]
    to_alias = suggestion["to_alias"]

    prompt = f"""## 逐对连接审核 [{pair_index + 1}/{total_pairs}]

### 完整拓扑描述
{topology_description}

### 当前连接对
从元件 **{from_alias}** 连接到 **{to_alias}**

### 代码端口匹配建议
- 建议: {from_alias}.port{suggestion['suggested_from_port']} → {to_alias}.port{suggestion['suggested_to_port']}
- 同域: {"是" if suggestion['same_domain'] else "否"}
- 匹配分数: {suggestion['score']}

**{from_alias} 端口详情:**
{_json.dumps(suggestion['from_port_details'], ensure_ascii=False, indent=2)}

**{to_alias} 端口详情:**
{_json.dumps(suggestion['to_port_details'], ensure_ascii=False, indent=2)}

### 所有元件上下文
{_json.dumps([{"alias": c.get("alias"), "icon_name": c.get("icon_name"), "library": c.get("library")} for c in all_component_data], ensure_ascii=False, indent=2)}

### 审核要求
请判断这个连接对应该如何处理，输出 JSON:

1. **DIRECT**: 如果两个端口同域(same_domain=true)且 io 方向互补，可直连
2. **JUNCTION**: 如果拓扑描述中 {from_alias} 需要同时连接多个目标（分两路/分支/多路），则需要插入三通/分配器元件
3. **BRIDGE**: 如果两个端口不同域，需要插入桥接元件进行跨域连接

输出 JSON 格式:
```json
{{
  "decision": "DIRECT|JUNCTION|BRIDGE",
  "connections": [
    {{"from_alias": "...", "from_port": 0, "to_alias": "...", "to_port": 0, "type": "line", "line_alias": "wire_N"}}
  ],
  "bridge_components": [
    {{"icon_key": "...", "icon_name": "...", "library": "...", "alias": "...", "position": [x, y], "submodel": "...", "submodel_path": "...", "reason": "..."}}
  ],
  "reason": "决策理由"
}}
```

注意:
- 如果 decision=DIRECT，connections 中只放一条连接
- 如果 decision=JUNCTION，需要新增 junction 元件到 bridge_components，并为所有分支指定 connections
- 如果 decision=BRIDGE，需要新增桥接元件到 bridge_components，connections 包含进出桥接元件的两条连接
- line_alias 必须唯一，建议用 wire_{from_alias}_{to_alias} 格式"""

    return prompt


# ============================================================
# Step 3.7: LLM 批量审核提示词 — 一次性规划所有连接
# ============================================================

def format_batch_review_prompt(
    topology_description: str,
    topology_pairs: list[tuple[str, str]],
    component_data: list[dict],
) -> str:
    """为所有拓扑对生成一次性批量审核提示词。

    LLM 在单个 prompt 中完成所有连接规划，基于端口 norm 语义进行匹配，
    而非评分体系。替代逐对调用的 format_connection_review_prompt()。

    Args:
        topology_description: 完整拓扑描述文本
        topology_pairs: 解析后的连接对列表
        component_data: 所有元件数据列表

    Returns:
        批量 LLM prompt 字符串
    """
    import json as _json

    # ---- 1. 构建端口摘要表 ----
    port_lines = []
    for comp in component_data:
        alias = comp.get("alias", "?")
        icon_name = comp.get("icon_name", "?")
        library = comp.get("library", "?")
        ports = comp.get("ports", [])
        port_lines.append(f"- **{alias}** ({library}:{icon_name}):")
        for port in ports:
            pidx = port.get("port_index", "?")
            ptag = port.get("port_tag", "?")
            io_val = _get_port_io(port)
            io_map = {"1": "output", "2": "input", "3": "bidirectional"}
            io_label = io_map.get(io_val, "unknown")
            variables = port.get("variables", [])
            norms = [v.get("norm", "?") for v in variables[:3]]
            norms_str = ", ".join(norms) if norms else "(none)"
            port_lines.append(
                f"  - Port {pidx}: tag={ptag}, io={io_label}, norms=[{norms_str}]"
            )

    ports_summary = "\n".join(port_lines)

    # ---- 2. 构建拓扑对列表 ----
    pairs_lines = [f"  {i+1}. {fa} → {ta}" for i, (fa, ta) in enumerate(topology_pairs)]
    pairs_summary = "\n".join(pairs_lines) if pairs_lines else "  (从拓扑描述动态解析)"

    # ---- 3. 生成完整 prompt ----
    prompt = f"""## 批量端口匹配审核

### 完整拓扑描述
{topology_description}

### 解析出的连接对 ({len(topology_pairs)} 对)
{pairs_summary}

### 所有元件端口摘要
{ports_summary}

### 判定规则
1. **DIRECT**: 两个端口 port_tag 相同且 io 方向互补 (output→input) → 直连
2. **JUNCTION**: 拓扑中一个元件需要连接多个目标（分两路/分支） → 插入三通/分配器
3. **BRIDGE**: 两个端口 port_tag 不同 → 需要桥接元件进行跨域连接

### 端口匹配方法
- **不要使用评分**。直接根据端口的 norms（物理量）和 io 方向进行语义匹配。
- 同域端口: 匹配互补的 io 方向。例如 Pump 的 output(pressure) 应连接 Valve 的 input(pressure)。
- 跨域端口: port_tag 不同时需要桥接。例如 thermal output → two_phase_flow input 需要 THCD00 桥接元件。
- 端口号从 0 开始 (0-indexed)。

### 输出 JSON 格式
```json
{{
  "connections": [
    {{{{
      "from_alias": "Pump",
      "from_port": 1,
      "to_alias": "Valve",
      "to_port": 0,
      "type": "line",
      "line_alias": "wire_Pump_Valve",
      "port_type_from": "hydraulic",
      "port_type_to": "hydraulic"
    }}}}
  ],
  "bridge_components": [
    {{{{
      "icon_key": "libth:th_c",
      "icon_name": "th_c",
      "library": "libth",
      "alias": "bridge_1",
      "position": [350, 400],
      "submodel": "THC000",
      "submodel_path": "$AME/libth/submodels",
      "reason": "连接 thermal 域 (port X) 与 two_phase_flow 域 (port Y)"
    }}}}
  ],
  "reason": "整体规划说明"
}}
```

### 重要约束
- 端口号使用 0-indexed
- line_alias 必须唯一，建议用 wire_<from>_<to> 格式
- 每个 (alias, port_index) 组合只能出现在一条连接中
- 必须使用上述端口摘要中的元件别名，不得自创
- 所有拓扑对都应产生至少一条连接
- 如果 port_tag 相同但 io 方向都是 output，优先匹配有最多共同 norms 的端口对
"""
    return prompt


# ============================================================
# Step 4: 悬空端口检测与终端补全
# ============================================================

def detect_and_resolve_dangling(
    connections: list[dict],
    component_ports: dict[str, list[dict]],
    component_info: dict[str, dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """检测悬空端口并补全终端元件。

    Args:
        connections: 已规划的连接列表
        component_ports: {alias: [port_dict, ...]}
        component_info: {alias: {"icon_key":..., "library":..., "submodel_id":...}}

    Returns:
        (new_connections, terminal_components, dangling_ports_report)
    """
    # 统计已占用端口
    used_ports: dict[str, set[int]] = defaultdict(set)
    for conn in connections:
        fa = conn.get("from_alias", "")
        fp = conn.get("from_port", -1)
        ta = conn.get("to_alias", "")
        tp = conn.get("to_port", -1)
        if fa and fp >= 0:
            used_ports[fa].add(fp)
        if ta and tp >= 0:
            used_ports[ta].add(tp)

    # 查找悬空端口
    dangling_ports = []
    for alias, ports in component_ports.items():
        for p in ports:
            p_idx = p.get("port_index", -1)
            if p_idx not in used_ports.get(alias, set()):
                variables = p.get("variables", [])
                io_values = {v.get("io", "") for v in variables if v.get("io") in ("1", "2")}
                if all(v == "1" for v in io_values):
                    io_summary = "output"
                elif all(v == "2" for v in io_values):
                    io_summary = "input"
                elif io_values:
                    io_summary = "bidirectional"
                else:
                    io_summary = "unknown"

                dangling_ports.append({
                    "alias": alias,
                    "port_index": p_idx,
                    "port_tag": p.get("port_tag", ""),
                    "io_summary": io_summary,
                    "icon_key": component_info.get(alias, {}).get("icon_key", ""),
                    "library": component_info.get(alias, {}).get("library", ""),
                })

    # 终端元件补全（简化版：根据 domain 选择通用终端）
    terminal_components = []
    new_connections = []

    terminal_idx = 0
    domain_terminals = {
        "mec_trans": {"icon_name": "zero_force_source", "library": "libmec", "submodel": "F000", "label": "Zero Force Source"},
        "mec_rot": {"icon_name": "zero_torque_source", "library": "libmec", "submodel": "T000", "label": "Zero Torque Source"},
        "signal": {"icon_name": "constant", "library": "libsig", "submodel": "CONS00", "label": "Constant"},
        "hydraulic": {"icon_name": "tank", "library": "libhydr", "submodel": "TK000", "label": "Tank"},
        "thermal": {"icon_name": "constant_temperature", "library": "libth", "submodel": "THC000", "label": "Const Temp"},
        "two_phase_flow": {"icon_name": "tank", "library": "libtpf", "submodel": "TPTK00", "label": "Tank"},
        "pneumatic": {"icon_name": "constant_pressure_source", "library": "libpn", "submodel": "PNCS001", "label": "Const Press"},
        "electric": {"icon_name": "ground", "library": "libeb", "submodel": "EBG00", "label": "Ground"},
    }

    for dp in dangling_ports:
        port_tag = dp.get("port_tag", "")
        terminal_idx += 1

        # 匹配域
        terminal = None
        for domain_key, t in domain_terminals.items():
            if domain_key in port_tag.lower() or port_tag.lower() in domain_key:
                terminal = t
                break

        if not terminal:
            # 通用回退: signal → constant, 其他 → 空（标记为 issue）
            if "signal" in port_tag.lower():
                terminal = domain_terminals["signal"]
            else:
                continue  # 无法确定终端，跳过（由 LLM 后续处理）

        alias = f"terminal_{terminal_idx}"
        terminal_components.append({
            "icon_key": f"{terminal['library']}:{terminal['icon_name']}",
            "icon_name": terminal["icon_name"],
            "library": terminal["library"],
            "alias": alias,
            "position": [100, 100 + terminal_idx * 150],
            "submodel": terminal["submodel"],
            "submodel_path": f"$AME/{terminal['library']}/submodels",
            "reason": f"悬空端口 {dp['alias']}.port{dp['port_index']} ({dp['io_summary']}) 的边界条件",
        })

        # 决定连接方向
        if dp["io_summary"] == "input":
            new_connections.append({
                "from_alias": alias,
                "from_port": 0,
                "to_alias": dp["alias"],
                "to_port": dp["port_index"],
                "type": "line",
                "line_alias": f"wire_{alias}_{dp['alias']}",
            })
        else:
            new_connections.append({
                "from_alias": dp["alias"],
                "from_port": dp["port_index"],
                "to_alias": alias,
                "to_port": 0,
                "type": "line",
                "line_alias": f"wire_{dp['alias']}_{alias}",
            })

    return new_connections, terminal_components, dangling_ports


# ============================================================
# Step 4.5: 纯检测悬空端口 — detect_dangling_ports_detailed()
# ============================================================

def detect_dangling_ports_detailed(
    connections: list[dict],
    component_ports: dict[str, list[dict]],
    component_info: dict[str, dict],
) -> list[dict]:
    """纯代码检测所有未连接端口及其性质（不补全终端）。

    这是迭代悬空闭环中"代码做查算"的部分——只检测，不决策。
    LLM 随后审核悬空端口列表并选择合适的终端元件。

    Args:
        connections: 当前所有连接 [{from_alias, from_port, to_alias, to_port}, ...]
        component_ports: {alias: [port_dict, ...]}
        component_info: {alias: {"icon_key":..., "library":..., "submodel_id":...}}

    Returns:
        悬空端口列表，每项含:
        - alias, port_index, port_tag, io_summary
        - icon_key, library（所属元件信息）
        - variable_norms（端口变量 norm 列表）
    """
    # 统计已占用端口
    used_ports: dict[str, set[int]] = defaultdict(set)
    for conn in connections:
        fa = conn.get("from_alias", "")
        fp = conn.get("from_port", -1)
        ta = conn.get("to_alias", "")
        tp = conn.get("to_port", -1)
        if fa and fp >= 0:
            used_ports[fa].add(fp)
        if ta and tp >= 0:
            used_ports[ta].add(tp)

    dangling_ports: list[dict] = []

    for alias, ports in component_ports.items():
        for p in ports:
            p_idx = p.get("port_index", -1)
            if p_idx in used_ports.get(alias, set()):
                continue

            variables = p.get("variables", [])
            io_values = {v.get("io", "") for v in variables if v.get("io") in ("1", "2")}
            if all(v == "1" for v in io_values):
                io_summary = "output"
            elif all(v == "2" for v in io_values):
                io_summary = "input"
            elif io_values:
                io_summary = "bidirectional"
            else:
                io_summary = "unknown"

            info = component_info.get(alias, {})
            dangling_ports.append({
                "alias": alias,
                "port_index": p_idx,
                "port_tag": p.get("port_tag", ""),
                "io_summary": io_summary,
                "icon_key": info.get("icon_key", ""),
                "library": info.get("library", ""),
                "submodel_id": info.get("submodel_id", ""),
                "variable_norms": [v.get("norm") for v in variables if v.get("norm")],
            })

    return dangling_ports


# ============================================================
# Step 4.6: LLM 审核提示词 — 迭代悬空端口审核
# ============================================================

def format_dangling_review_prompt(
    dangling_ports: list[dict],
    connections: list[dict],
    terminal_components: list[dict],
    iteration: int,
    max_iterations: int,
    topology_description: str,
    all_component_data: list[dict],
) -> str:
    """为悬空端口检测结果生成 LLM 审核提示词。

    LLM 需要为每个悬空端口选择合适的终端/边界元件。

    Args:
        dangling_ports: detect_dangling_ports_detailed() 的返回结果
        connections: 当前所有连接
        terminal_components: 已有的终端元件（避免重复补充）
        iteration: 当前迭代轮次 (0-based)
        max_iterations: 最大迭代轮次
        topology_description: 完整拓扑描述
        all_component_data: 所有元件数据列表

    Returns:
        LLM prompt 字符串
    """
    import json as _json

    existing_terminal_aliases = {t.get("alias", "") for t in terminal_components}
    # 过滤掉已补全终端的悬空端口（同名alias的已处理过）
    pending_dangling = [
        dp for dp in dangling_ports
        if dp["alias"] not in existing_terminal_aliases
        or not any(
            t.get("alias") == dp["alias"]
            for t in terminal_components
        )
    ]

    if not pending_dangling:
        pending_dangling = dangling_ports

    prompt = f"""## 悬空端口闭环审核 [第 {iteration + 1}/{max_iterations} 轮]

### 完整拓扑描述
{topology_description}

### 当前悬空端口 ({len(pending_dangling)} 个)
{_json.dumps(pending_dangling, ensure_ascii=False, indent=2)}

### 已有连接 ({len(connections)} 条)
{_json.dumps(connections[-20:] if len(connections) > 20 else connections, ensure_ascii=False, indent=2)}

### 已有终端元件 ({len(terminal_components)} 个)
{_json.dumps(terminal_components, ensure_ascii=False, indent=2)}

### 所有元件上下文
{_json.dumps([{"alias": c.get("alias"), "icon_name": c.get("icon_name"), "library": c.get("library")} for c in all_component_data], ensure_ascii=False, indent=2)}

### 审核要求
为每个悬空端口选择最合适的终端/边界元件。根据 port_tag 域类型和 io 方向:
- hydraulic input → tank, constant_pressure_source
- hydraulic output → tank, orifice
- mechanical input → zero_force_source, ground, constant_speed
- mechanical output → zero_force_source, damper
- thermal input → constant_temperature, ambient
- thermal output → constant_temperature, convective
- signal input → constant, piecewise_linear_source, sine_source
- signal output → signal_sink, scope
- two_phase_flow input → tank, mass_flow_source
- two_phase_flow output → pressure_source, tank
- electric → ground

输出 JSON 格式:
```json
{{
  "terminal_components": [
    {{"icon_key": "...", "icon_name": "...", "library": "...", "alias": "terminal_N", "position": [x, y], "submodel": "...", "submodel_path": "...", "reason": "为XX的portY提供边界条件"}}
  ],
  "connections": [
    {{"from_alias": "...", "from_port": 0, "to_alias": "...", "to_port": 0, "type": "line", "line_alias": "wire_terminal_N"}}
  ],
  "reason": "本轮终端补全说明",
  "no_more_dangling": false
}}
```

注意:
- 如果某个端口确实需要保留悬空（如测试端口），设置 no_more_dangling=true
- terminal 的端口号 port 0 是默认出口端口
- input 端口 → terminal 作为源（from_alias=terminal），output 端口 → terminal 作为汇（to_alias=terminal）
- 不要为已经补全过的元件重复添加终端"""

    return prompt


# ============================================================
# 主流程: run_bridge_pipeline()
# ============================================================

def run_bridge_pipeline(
    component_data: list[dict],
    topology_description: str,
    component_aliases: list[str] | None = None,
) -> BridgePlanResult:
    """运行完整的 Bridge 代码流程。

    Args:
        component_data: 元件端口数据列表
            每项含: alias, icon_key, icon_name, library, submodel_id, ports
        topology_description: 拓扑描述文本
        component_aliases: 已知别名列表（可选，供拓扑解析使用）

    Returns:
        BridgePlanResult
    """
    # ── 构建索引 ──
    component_ports: dict[str, list[dict]] = {}
    component_info: dict[str, dict] = {}
    aliases = []

    for c in component_data:
        alias = c.get("alias", "")
        if not alias:
            continue
        aliases.append(alias)
        component_ports[alias] = c.get("ports", [])
        component_info[alias] = {
            "icon_key": c.get("icon_key", ""),
            "library": c.get("library", ""),
            "submodel_id": c.get("submodel_id", ""),
            "icon_name": c.get("icon_name", ""),
        }

    if component_aliases is None:
        component_aliases = aliases

    result = BridgePlanResult()

    # ── Step 1: 元件存在性验证 ──
    verify_components = []
    for alias, info in component_info.items():
        verify_components.append({
            "alias": alias,
            "icon_key": info["icon_key"],
            "icon_name": info["icon_name"],
            "library": info["library"],
        })
    verification = verify_components_exist(verify_components)
    if not verification["all_exist"]:
        result.is_valid = False
        for m in verification["missing"]:
            result.validation_issues.append({
                "level": "error",
                "code": "COMPONENT_NOT_IN_DB",
                "message": f"元件 '{m['alias']}' ({m['icon_key']}) 不在数据库中",
            })

    # ── Step 2: 解析拓扑对 ──
    topology_pairs = parse_topology_pairs(topology_description, component_aliases)
    print(f"  [Bridge-Core] 解析到 {len(topology_pairs)} 个拓扑对: {topology_pairs}")

    # ── Step 3: 端口匹配连接 ──
    connections, plan_issues = plan_connections(
        topology_pairs, component_ports,
        {a: component_info[a]["library"] for a in component_info},
    )
    result.connections = connections
    for issue in plan_issues:
        result.validation_issues.append(issue)

    print(f"  [Bridge-Core] 规划了 {len(connections)} 个连接")

    # ── Step 4: 悬空端口终端补全 ──
    new_conns, terminals, dangling = detect_and_resolve_dangling(
        connections, component_ports, component_info,
    )
    result.connections.extend(new_conns)
    result.terminal_components = terminals
    result.dangling_ports = dangling

    if dangling:
        print(f"  [Bridge-Core] {len(dangling)} 个悬空端口, 补全了 {len(terminals)} 个终端元件")
    else:
        print(f"  [Bridge-Core] 无悬空端口")

    # ── 最终验证 ──
    result.is_valid = result.is_valid and not any(
        i.get("level") == "error" for i in result.validation_issues
    )

    return result
