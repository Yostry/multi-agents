"""
Bridge Agent 工具 — 元件审查、端口查询、连接规划、桥接查找

基于统一文档 SQLite + 动态端口匹配 (替代预存端口图):
  - verify_all_components_exist: 批量验证元件是否在知识库中
  - query_component_ports: 查询元件的端口信息 (含变量级 io/norm/title/units)
  - find_port_connections: 动态匹配端口连接候选
  - find_bridge_paths: 查找跨域桥接路径
  - find_junction_component: ★ 查找分叉/汇合三通或信号分配器元件
  - validate_connection_plan: 验证完整连接方案的合法性
"""
from __future__ import annotations

import json
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from agents import function_tool


def _load_component_index():
    """加载 component_index.json"""
    from pathlib import Path
    p = (Path(__file__).resolve().parent.parent.parent.parent /
         "knowledge_base" / "json" / "index" / "component_index.json")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


@function_tool
def find_junction_component(domain: str, junction_type: str = "tee",
                            library: str = "", top_k: int = 5) -> str:
    """★ 查找分叉/汇合三通或信号分配器元件。

    当检测到以下情况时调用本工具:
    - 一个端口需要连到多个目标 → 分叉三通 (junction_type="tee")
    - 多个端口需要汇合到同一个目标 → 汇合三通 (junction_type="merge")
    - 一个信号源需要控制多个目标 → 信号分配器 (junction_type="splitter")

    Args:
        domain: 物理域 (hydraulic, thermal, two_phase_flow, signal, pneumatic, electric)
        junction_type: 分叉类型 ("tee"分叉三通 / "merge"汇合三通 / "splitter"信号分配器)
        library: 限定库名，如 "libhydr"，留空时按 domain 自动推断
        top_k: 返回候选数量

    Returns:
        JSON 格式的候选元件列表
    """
    from amesim_builder.unified_search import get_searcher

    # 按 domain 推断默认搜索库和关键词
    domain_map = {
        "hydraulic": ("libhydr", ["tee", "junction", "t_junction"]),
        "thermal": ("libth", ["tee", "junction", "thermal junction"]),
        "two_phase_flow": ("libtpf", ["tee", "junction", "t_junction"]),
        "signal": ("libsig", ["junction", "splitter", "multiplexer", "demux"]),
        "pneumatic": ("libpn", ["tee", "junction"]),
        "electric": ("libeb", ["junction", "node"]),
    }

    default_lib, keywords = domain_map.get(domain, ("", [junction_type]))
    search_lib = library if library else default_lib

    searcher = get_searcher()
    all_results = []

    for kw in keywords:
        hits = searcher.search(kw, top_k=top_k, library=search_lib or None)
        for h in hits:
            all_results.append({
                "icon_key": f"{h['library']}:{h['icon_name']}",
                "icon_name": h.get("icon_name", ""),
                "library": h.get("library", ""),
                "label": (h.get("label", "") or "")[:80],
                "description_preview": (h.get("text", "") or "")[:150],
                "port_count": len(h.get("ports", [])),
                "rrf_score": h.get("rrf_score", 0),
            })

    # 去重 + 按分数排序
    seen = set()
    unique = []
    for r in sorted(all_results, key=lambda x: -x["rrf_score"]):
        if r["icon_key"] not in seen:
            seen.add(r["icon_key"])
            unique.append(r)

    return json.dumps({
        "domain": domain,
        "junction_type": junction_type,
        "library_searched": search_lib or "all",
        "keywords_used": keywords,
        "hits": len(unique),
        "candidates": unique[:top_k],
        "note": "选择端口数≥3的元件作为三通/分配器",
    }, ensure_ascii=False, indent=2)


@function_tool
def verify_all_components_exist(components_json: str) -> str:
    """验证Selector选定的所有元件是否都在knowledge_base中。

    Args:
        components_json: JSON数组，每项含 icon_key (格式 "library:icon_name") 和 alias

    Returns:
        JSON格式的验证结果: {all_exist: bool, missing: [...], verified: [...]}
    """
    components = json.loads(components_json)
    comp_idx = _load_component_index()
    verified = []
    missing = []

    for comp in components:
        key = comp.get("icon_key", "")
        alias = comp.get("alias", "?")
        entry = comp_idx.get(key)
        if entry:
            verified.append({
                "alias": alias,
                "icon_key": key,
                "exists": True,
                "library": entry["library"],
                "domain": entry.get("domain_cn", ""),
            })
        else:
            missing.append({
                "alias": alias,
                "icon_key": key,
                "exists": False,
                "reason": f"在 component_index 中未找到 {key}",
            })

    return json.dumps({
        "all_exist": len(missing) == 0,
        "verified_count": len(verified),
        "missing_count": len(missing),
        "verified": verified[:50],
        "missing": missing,
    }, ensure_ascii=False, indent=2)


@function_tool
def query_component_ports(icon_key: str, submodel_id: str = "") -> str:
    """查询指定元件的所有端口信息，包括端口编号、物理域tag、变量级io方向。

    Args:
        icon_key: 格式为 "library:icon_name"，如 "libth:th_c"
        submodel_id: 可选，指定子模型ID以精准定位

    Returns:
        JSON格式的端口列表，每端口含 variables[norm/io/title/units]
    """
    from amesim_builder.unified_search import get_searcher

    searcher = get_searcher()

    if submodel_id:
        # 精准查 submodel
        lib, icon = icon_key.split(":", 1) if ":" in icon_key else ("", icon_key)
        doc_id = f"{lib}:{submodel_id}|{icon}"
        doc = searcher.get_document(doc_id)
    else:
        doc = searcher.get_document_by_iconkey(icon_key)

    if not doc:
        return json.dumps({
            "icon_key": icon_key,
            "found": False,
            "message": f"未找到元件端口信息: {icon_key}",
            "ports": [],
        }, ensure_ascii=False, indent=2)

    ports_out = []
    for p in doc.get("ports", []):
        ports_out.append({
            "port_index": p.get("index"),
            "port_tag": p.get("tag", ""),
            "variables": [
                {"norm": v.get("norm"), "io": v.get("io"),
                 "title": v.get("title"), "units": v.get("units")}
                for v in p.get("variables", [])
            ],
        })

    return json.dumps({
        "icon_key": icon_key,
        "found": True,
        "doc_id": doc.get("doc_id"),
        "library": doc.get("library"),
        "icon_name": doc.get("icon_name"),
        "submodel_id": doc.get("submodel_id"),
        "total_ports": len(ports_out),
        "ports": ports_out,
    }, ensure_ascii=False, indent=2)


@function_tool
def batch_query_ports(components_json: str) -> str:
    """★ 批量查询多个元件的端口信息。一次调用即可获取所有元件的端口详情。

    应优先使用此工具替代多次 query_component_ports 调用。

    Args:
        components_json: JSON数组，每项含 icon_key (格式 "library:icon_name") 和可选的 submodel_id、alias
            示例: [{"icon_key": "libmec:mass_friction_endstops", "alias": "Mass"}, ...]

    Returns:
        JSON格式，每项含 alias/icon_key/ports 列表
    """
    from amesim_builder.unified_search import get_searcher

    components = json.loads(components_json)
    searcher = get_searcher()
    results = []

    for comp in components:
        icon_key = comp.get("icon_key", "")
        submodel_id = comp.get("submodel_id", "")
        alias = comp.get("alias", "")

        if submodel_id:
            lib, icon = icon_key.split(":", 1) if ":" in icon_key else ("", icon_key)
            doc_id = f"{lib}:{submodel_id}|{icon}"
            doc = searcher.get_document(doc_id)
        else:
            doc = searcher.get_document_by_iconkey(icon_key)

        if not doc:
            results.append({
                "alias": alias,
                "icon_key": icon_key,
                "found": False,
                "message": f"未找到元件端口信息: {icon_key}",
                "ports": [],
            })
            continue

        ports_out = []
        for p in doc.get("ports", []):
            ports_out.append({
                "port_index": p.get("index"),
                "port_tag": p.get("tag", ""),
                "variables": [
                    {"norm": v.get("norm"), "io": v.get("io"),
                     "title": v.get("title"), "units": v.get("units")}
                    for v in p.get("variables", [])
                ],
            })

        results.append({
            "alias": alias,
            "icon_key": icon_key,
            "found": True,
            "library": doc.get("library"),
            "icon_name": doc.get("icon_name"),
            "submodel_id": doc.get("submodel_id"),
            "total_ports": len(ports_out),
            "ports": ports_out,
        })

    return json.dumps({
        "total": len(results),
        "found_count": sum(1 for r in results if r.get("found")),
        "results": results,
    }, ensure_ascii=False, indent=2)


@function_tool
def find_port_connections(comp_list_json: str, topology_description: str) -> str:
    """根据元件列表和拓扑描述，动态匹配可行的端口连接方案。

    使用统一文档存储 + 动态 can_connect() 判定，不依赖预存端口图边。

    Args:
        comp_list_json: JSON数组，每项含 alias, icon_name, library, submodel_id
        topology_description: 拓扑关系文字描述 (来自Orchestrator)

    Returns:
        JSON格式的连接规划建议
    """
    from amesim_builder.unified_search import get_searcher, find_compatible_ports

    comp_list = json.loads(comp_list_json)
    searcher = get_searcher()
    suggestions = []

    for comp in comp_list:
        icon = comp.get("icon_name", "")
        submodel = comp.get("submodel_id", "")
        alias = comp.get("alias", "")
        library = comp.get("library", "")

        doc_id = f"{library}:{submodel}|{icon}"

        # 查询每个端口的兼容连接
        for port_idx in range(1, 7):  # 最多查6个端口
            try:
                candidates = find_compatible_ports(searcher, doc_id, port_idx)
                if candidates:
                    top_candidates = [
                        {
                            "library": c["library"],
                            "icon_name": c["icon_name"],
                            "submodel_id": c["submodel_id"],
                            "port_index": c["port_index"],
                            "port_tag": c.get("port_tag", ""),
                            "label": c.get("label", ""),
                            "matched_norms": c.get("matched_norms", []),
                        }
                        for c in candidates[:5]
                    ]
                    suggestions.append({
                        "from_alias": alias,
                        "from_port": port_idx,
                        "from_icon": icon,
                        "candidate_count": len(candidates),
                        "top_candidates": top_candidates,
                    })
            except Exception:
                pass

    return json.dumps({
        "topology_description": topology_description,
        "match_type": "dynamic_port_matching",
        "suggestion_count": len(suggestions),
        "suggestions": suggestions[:30],
        "note": "候选列表基于动态端口变量匹配(共享变量IO互补)。同一物理域端口间可直连。",
    }, ensure_ascii=False, indent=2)


@function_tool
def find_bridge_paths(comp_a_json: str, comp_b_json: str) -> str:
    """查找两个可能跨物理域的元件之间的桥接路径。

    Args:
        comp_a_json: 元件A的JSON，含 icon_name, library, submodel_id, port_index
        comp_b_json: 元件B的JSON，含 icon_name, library, submodel_id, port_index

    Returns:
        JSON格式的桥接路径列表
    """
    from amesim_builder.unified_search import get_searcher, find_bridge

    comp_a = json.loads(comp_a_json)
    comp_b = json.loads(comp_b_json)

    doc_id_a = f"{comp_a['library']}:{comp_a['submodel_id']}|{comp_a['icon_name']}"
    doc_id_b = f"{comp_b['library']}:{comp_b['submodel_id']}|{comp_b['icon_name']}"
    port_a = comp_a.get('port_index', 1)
    port_b = comp_b.get('port_index', 1)

    try:
        searcher = get_searcher()
        paths = find_bridge(searcher, doc_id_a, port_a, doc_id_b, port_b, max_hops=3, top_k=3)
    except Exception as e:
        return json.dumps({
            "found": False,
            "message": f"桥接搜索失败: {e}",
            "common_bridges": {
                "thermal_to_tpf": "THCD00 (libthcd) 或 THC100 (libth)",
                "mechanical_to_signal": "力/位移传感器",
                "hydraulic_to_mechanical": "液压缸",
            },
        }, ensure_ascii=False, indent=2)

    if not paths:
        return json.dumps({
            "comp_a": comp_a.get("icon_name"),
            "comp_b": comp_b.get("icon_name"),
            "found": False,
            "message": "未找到桥接路径，两个元件可能无需桥接或尚不支持",
        }, ensure_ascii=False, indent=2)

    result_paths = []
    for i, path in enumerate(paths[:3]):
        nodes = [
            f"{n['library']}:{n['icon_name']}#{n['port_index']}({n.get('submodel_id','')})"
            for n in path
        ]
        result_paths.append({
            "path_index": i + 1,
            "hops": len(path),
            "route": " → ".join(nodes),
            "bridge_components": [
                n for n in path[1:-1]
            ],
        })

    return json.dumps({
        "comp_a": comp_a.get("icon_name"),
        "comp_b": comp_b.get("icon_name"),
        "found": True,
        "match_type": "dynamic_bridge_search",
        "paths": result_paths,
    }, ensure_ascii=False, indent=2)


@function_tool
def validate_connection_plan(connections_json: str, components_json: str) -> str:
    """验证完整的连接方案的合法性。

    检查:
    - 端口是否被多次连接 (PORT_MULTIPLE_CONN)
    - 别名引用是否存在 (UNKNOWN_ALIAS)
    - 端口号是否越界

    Args:
        connections_json: 连接定义JSON数组
        components_json: 元件定义JSON数组

    Returns:
        JSON格式的验证结果
    """
    connections = json.loads(connections_json)
    components = json.loads(components_json)

    # 构建 alias -> 端口数 映射
    alias_set = {c["alias"] for c in components}

    issues = []
    port_usage: dict[tuple[str, int], int] = defaultdict(int)

    for i, conn in enumerate(connections):
        fa, fp = conn.get("from_alias", ""), conn.get("from_port", -1)
        ta, tp = conn.get("to_alias", ""), conn.get("to_port", -1)

        # 别名检查
        if fa and fa not in alias_set:
            issues.append({
                "level": "error",
                "code": "UNKNOWN_ALIAS",
                "message": f"连接{i}: 元件 '{fa}' 不在元件列表中",
            })
        if ta and ta not in alias_set:
            issues.append({
                "level": "error",
                "code": "UNKNOWN_ALIAS",
                "message": f"连接{i}: 元件 '{ta}' 不在元件列表中",
            })

        # 端口占用统计
        port_usage[(fa, fp)] += 1
        port_usage[(ta, tp)] += 1

    # 检查重复连接
    for (alias, port), cnt in port_usage.items():
        if cnt > 1:
            issues.append({
                "level": "error",
                "code": "PORT_MULTIPLE_CONN",
                "message": f"{alias}.port{port} 被连接了 {cnt} 次",
            })

    is_valid = not any(i["level"] == "error" for i in issues)

    # 悬空端口检测: 查询各元件端口，找出未出现在 connections 中的端口
    dangling_ports: list[dict] = []
    for comp in components:
        alias = comp.get("alias", "")
        icon_key = comp.get("icon_key", "")
        # 统计该元件在 connections 中占用的端口号
        used_ports: set[int] = set()
        for conn in connections:
            if conn.get("from_alias") == alias:
                used_ports.add(conn.get("from_port", -1))
            if conn.get("to_alias") == alias:
                used_ports.add(conn.get("to_port", -1))
        # 尝试查询该元件的端口信息
        try:
            ports_info_str = query_component_ports(icon_key)
            ports_info = json.loads(ports_info_str)
            total_ports = ports_info.get("total_ports", 0)
            for p in ports_info.get("ports", []):
                pi = p.get("port_index", -1)
                if pi not in used_ports:
                    variables = p.get("variables", [])
                    io_values = [v.get("io", "") for v in variables if v.get("io") in ("1", "2")]
                    if all(v == "1" for v in io_values):
                        io_summary = "output"
                    elif all(v == "2" for v in io_values):
                        io_summary = "input"
                    elif io_values:
                        io_summary = "bidirectional"
                    else:
                        io_summary = "unknown"
                    issues.append({
                        "level": "warning",
                        "code": "PORT_DANGLING",
                        "message": f"{alias}.port{pi} ({port_tag}/{io_summary}) 未连接，可能需终端元件",
                    })
                    dangling_ports.append({
                        "alias": alias,
                        "port_index": pi,
                        "port_tag": p.get("port_tag", ""),
                        "io_summary": io_summary,
                        "variable_norms": [v.get("norm") for v in variables if v.get("norm")],
                    })
        except Exception:
            pass

    return json.dumps({
        "is_valid": is_valid,
        "connection_count": len(connections),
        "component_count": len(components),
        "issues": issues,
        "port_usage_summary": {f"{a}.p{p}": c for (a, p), c in port_usage.items()},
        "dangling_ports": dangling_ports,
        "dangling_count": len(dangling_ports),
    }, ensure_ascii=False, indent=2)


@function_tool
def detect_dangling_ports(components_json: str, connections_json: str) -> str:
    """检测所有元件中未被连接的悬空端口。

    遍历所有元件的全部端口，对比已规划的连接列表，
    找出零次出现的端口，返回每个悬空端口的完整上下文信息。

    Args:
        components_json: JSON数组，每项含 alias, icon_key, library, submodel_id
        connections_json: JSON数组，每项含 from_alias, from_port, to_alias, to_port

    Returns:
        JSON格式的悬空端口列表，每项含 alias, port_index, port_tag, io_summary, doc_id, variable_norms, reason
    """
    from amesim_builder.unified_search import get_searcher

    components = json.loads(components_json)
    connections = json.loads(connections_json)

    # 构建已占用端口集合: (alias, port_index)
    used_ports: set[tuple[str, int]] = set()
    for conn in connections:
        fa = conn.get("from_alias", "")
        fp = conn.get("from_port", -1)
        ta = conn.get("to_alias", "")
        tp = conn.get("to_port", -1)
        if fa and fp >= 0:
            used_ports.add((fa, fp))
        if ta and tp >= 0:
            used_ports.add((ta, tp))

    # 构建 alias → {icon_key, library, submodel_id} 映射
    alias_map: dict[str, dict] = {}
    for comp in components:
        alias_map[comp.get("alias", "")] = {
            "icon_key": comp.get("icon_key", ""),
            "library": comp.get("library", ""),
            "submodel_id": comp.get("submodel_id", ""),
        }

    searcher = get_searcher()
    dangling_ports: list[dict] = []
    total_connected = 0
    all_checked = 0

    for comp in components:
        alias = comp.get("alias", "")
        icon_key = comp.get("icon_key", "")
        library = comp.get("library", "")
        submodel = comp.get("submodel_id", "")
        if not alias or not icon_key:
            continue
        all_checked += 1

        # 查询端口信息
        doc_id = f"{library}:{submodel}|{comp.get('icon_name', icon_key.split(':')[-1])}" if library and submodel else ""
        ports_out = []
        try:
            ports_info_str = query_component_ports(icon_key, submodel_id=submodel)
            ports_info = json.loads(ports_info_str)
            doc_id = ports_info.get("doc_id", doc_id)
            ports_out = ports_info.get("ports", [])
        except Exception:
            # 回退: 尝试通过 searcher 直接查
            if not doc_id:
                try:
                    doc = searcher.get_document_by_iconkey(icon_key)
                    if doc:
                        doc_id = doc.get("doc_id", "")
                        ports_out = doc.get("ports", [])
                except Exception:
                    pass

        for p in ports_out:
            pi = p.get("port_index", -1)
            if (alias, pi) in used_ports:
                total_connected += 1
                continue

            variables = p.get("variables", [])
            io_values = [v.get("io", "") for v in variables if v.get("io") in ("1", "2")]
            if all(v == "1" for v in io_values):
                io_summary = "output"
            elif all(v == "2" for v in io_values):
                io_summary = "input"
            elif io_values:
                io_summary = "bidirectional"
            else:
                io_summary = "unknown"

            port_tag = p.get("port_tag", "") or p.get("tag", "")
            dangling_ports.append({
                "alias": alias,
                "port_index": pi,
                "port_tag": port_tag,
                "io_summary": io_summary,
                "doc_id": doc_id,
                "icon_key": icon_key,
                "library": library,
                "submodel_id": submodel,
                "variable_norms": [v.get("norm") for v in variables if v.get("norm")],
                "reason": f"{alias}.port{pi} ({port_tag}/{io_summary}) 未连接任何元件",
            })

    return json.dumps({
        "dangling_ports": dangling_ports,
        "total_dangling": len(dangling_ports),
        "total_connected": total_connected,
        "all_components_checked": all_checked,
        "note": "对每个悬空端口，后续应调用 find_dangling_terminal_candidates 查找兼容的终端元件",
    }, ensure_ascii=False, indent=2)


@function_tool
def find_dangling_terminal_candidates(dangling_port_json: str, top_k: int = 8) -> str:
    """为单个悬空端口查找端口兼容的候选终端元件。

    基于统一搜索引擎的动态端口匹配 (find_compatible_ports)，
    找出所有端口变量 IO 互补的候选元件，并附加启发式标注
    (likely_terminal) 帮助 LLM 判断是否为合适的边界/终端元件。

    Args:
        dangling_port_json: 单个悬空端口的JSON，含 doc_id, port_index (从 detect_dangling_ports 获取)
        top_k: 返回候选数量上限

    Returns:
        JSON格式的候选列表，每项含 icon_key, icon_name, library, submodel_id,
        port_index, port_tag, label, description, matched_norms, likely_terminal
    """
    from amesim_builder.unified_search import get_searcher, find_compatible_ports

    dp = json.loads(dangling_port_json)
    doc_id = dp.get("doc_id", "")
    port_index = dp.get("port_index", -1)
    alias = dp.get("alias", "?")
    port_tag = dp.get("port_tag", "")
    io_summary = dp.get("io_summary", "")

    if not doc_id or port_index < 0:
        return json.dumps({
            "dangling_port": dp,
            "error": "缺少 doc_id 或 port_index，无法查询",
            "candidates": [],
            "candidate_count": 0,
        }, ensure_ascii=False, indent=2)

    searcher = get_searcher()
    candidates_raw = find_compatible_ports(searcher, doc_id, port_index)

    # 终端元件关键词 (label 中含以下词 → 可能是边界条件元件)
    terminal_keywords = [
        "tank", "reservoir", "source", "ground", "constant",
        "ambient", "sink", "scope", "zero", "atmosphere",
        "pressure_source", "flow_source", "temperature_source",
        "signal_sink", "force_source", "speed_source",
    ]

    candidates = []
    for c in candidates_raw[:top_k * 2]:  # 多取一些做筛选
        label = (c.get("label", "") or "").lower()
        desc = label  # 简化: 用 label 作为 description
        # 启发式判断
        likely = any(kw in label for kw in terminal_keywords)
        candidates.append({
            "icon_key": f"{c['library']}:{c['icon_name']}",
            "icon_name": c.get("icon_name", ""),
            "library": c.get("library", ""),
            "submodel_id": c.get("submodel_id", ""),
            "port_index": c.get("port_index", 0),
            "port_tag": c.get("port_tag", ""),
            "label": c.get("label", "")[:100] if c.get("label") else "",
            "description": desc[:150],
            "matched_norms": c.get("matched_norms", []),
            "likely_terminal": likely,
        })

    # 排序: likely_terminal=true 的优先，再按同库优先
    candidates.sort(key=lambda x: (
        not x["likely_terminal"],
        x["library"] != dp.get("library", ""),
    ))
    candidates = candidates[:top_k]

    return json.dumps({
        "dangling_port": {
            "alias": alias,
            "port_index": port_index,
            "port_tag": port_tag,
            "io_summary": io_summary,
            "doc_id": doc_id,
        },
        "candidates": candidates,
        "candidate_count": len(candidates),
        "guidance": (
            f"该悬空端口属于 {port_tag or '?'} 域, io方向={io_summary}。"
            "likely_terminal=true 的候选是常见的边界/终端元件，优先考虑。"
        ),
    }, ensure_ascii=False, indent=2)
