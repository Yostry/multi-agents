"""
Parameter Core — 纯代码参数默认分配引擎

零 LLM 参与。根据元件端口信息、连接关系和拓扑需求，自动分类参数：
  - 内部参数（无信号链 + 无特殊需求） → 代码赋默认值
  - 信号链参数（涉及 signal 域数据传递） → 标记需 LLM 根据物理意义推断
  - 拓扑/用户需求参数 → 标记需 LLM 处理

流程:
  1. load_component_params() — 批量查询所有元件参数定义
  2. classify_components() — 分类: internal vs signal_chain vs user_specified
  3. assign_internal_defaults() — 为内部元件赋默认值
  4. extract_signal_chain_items() — 提取需 LLM 处理的信号链项目
  5. run_parameter_pipeline() — 完整流程编排
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# 导入参数查询工具（纯代码版本）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from amesim_multi_agent.tools.parameter_tools import batch_query_params as _batch_query_params_fn


# ============================================================
# 结果数据类
# ============================================================

@dataclass
class ParamItem:
    """单个参数的详情"""
    varname: str = ""
    title: str = ""
    default: str = ""
    min_val: str = ""
    max_val: str = ""
    units: str = ""
    assigned_value: str = ""          # 代码分配的默认值
    needs_llm: bool = False           # 是否需要 LLM 处理
    llm_reason: str = ""              # 为什么需要 LLM


@dataclass
class ComponentParamResult:
    """单个元件的参数处理结果"""
    alias: str = ""
    icon_name: str = ""
    submodel_id: str = ""
    library: str = ""
    category: str = "internal"        # internal | signal_chain | user_specified
    params: list[ParamItem] = field(default_factory=list)
    all_found: bool = True


@dataclass
class ParameterPipelineResult:
    """参数阶段完整产出"""
    internal_assignments: list[ComponentParamResult] = field(default_factory=list)
    signal_chain_items: list[ComponentParamResult] = field(default_factory=list)
    user_specified_items: list[ComponentParamResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ============================================================
# Step 1: 批量查询参数
# ============================================================

def load_component_params(
    component_data: list[dict],
) -> list[dict]:
    """批量查询所有元件的子模型参数定义。

    Args:
        component_data: 来自 _component_data.json 的元件列表
            每项含: alias, library, submodel_id, icon_name

    Returns:
        批量查询结果列表 [{alias, library, submodel_id, real_params, int_params, ...}, ...]
    """
    query_list = []
    for c in component_data:
        lib = c.get("library", "")
        sm = c.get("submodel_id", "")
        alias = c.get("alias", "")
        if lib and sm and alias:
            query_list.append({
                "library": lib,
                "submodel_id": sm,
                "alias": alias,
            })

    if not query_list:
        return []

    try:
        result_str = _batch_query_params_fn(json.dumps(query_list, ensure_ascii=False))
        result = json.loads(result_str)
        return result.get("results", [])
    except Exception:
        return []


# ============================================================
# Step 2: 分类 — internal vs signal_chain vs user_specified
# ============================================================

def _get_port_types_for_alias(alias: str, component_data: list[dict]) -> list[str]:
    """获取指定元件的所有端口类型标签。"""
    for c in component_data:
        if c.get("alias") == alias:
            ports = c.get("ports", [])
            return [p.get("port_tag", "") for p in ports]
    return []


def _get_connected_port_types_for_alias(
    alias: str, connections: list[dict], component_data: list[dict]
) -> list[str]:
    """获取指定元件已连接端口的端口类型标签。"""
    all_port_types = _get_port_types_for_alias(alias, component_data)
    connected_ports = set()

    for conn in connections:
        if conn.get("from_alias") == alias:
            connected_ports.add(conn.get("from_port", -1))
        if conn.get("to_alias") == alias:
            connected_ports.add(conn.get("to_port", -1))

    result = []
    for port_idx in sorted(connected_ports):
        if port_idx < len(all_port_types):
            result.append(all_port_types[port_idx])
    return result


def is_signal_chain_component(
    alias: str,
    connections: list[dict],
    component_data: list[dict],
) -> bool:
    """判断元件是否涉及信号域数据传递。

    信号链判定标准:
    - 元件的已连接端口中，任一端口类型属于 signal 域
    - 元件通过 signal 端口与其他元件交换数据

    Args:
        alias: 元件别名
        connections: 连接列表
        component_data: 元件端口数据

    Returns:
        True 表示需要 LLM 根据物理意义推断参数
    """
    connected_port_types = _get_connected_port_types_for_alias(
        alias, connections, component_data,
    )

    # 检查是否涉及 signal 域
    signal_keywords = {"signal", "sig_", "control", "sensor", "lin_sig"}
    for pt in connected_port_types:
        pt_lower = pt.lower()
        for kw in signal_keywords:
            if kw in pt_lower:
                return True

    # 检查元件自身是否为信号库元件
    for c in component_data:
        if c.get("alias") == alias:
            library = c.get("library", "").lower()
            if "sig" in library:
                return True

    return False


def classify_components(
    component_data: list[dict],
    connections: list[dict],
    user_request: str = "",
    topology_description: str = "",
) -> tuple[list[dict], list[dict], list[dict]]:
    """分类元件: internal / signal_chain / user_specified。

    Returns:
        (internal_list, signal_chain_list, user_specified_list)
        每项为 component_data 子集
    """
    internal = []
    signal_chain = []
    user_specified = []

    # 从用户需求中提取可能的用户指定关键词（元件名称等）
    user_keywords = set()
    if user_request:
        # 简单提取: 包含数值的参数描述
        for word in user_request.replace("，", ",").replace("、", ",").split(","):
            word = word.strip()
            if any(ch.isdigit() for ch in word):
                user_keywords.add(word)

    for c in component_data:
        alias = c.get("alias", "")

        # 检查是否信号链
        if is_signal_chain_component(alias, connections, component_data):
            signal_chain.append(c)
            continue

        # 检查是否用户指定需求（有数值要求）
        if user_keywords and any(
            kw.lower() in alias.lower() or alias.lower() in kw.lower()
            for kw in user_keywords
        ):
            user_specified.append(c)
            continue

        # 是否需要看拓扑描述中的特殊要求
        if topology_description and alias.lower() in topology_description.lower():
            # 检查是否有数值修饰
            desc_lower = topology_description.lower()
            alias_pos = desc_lower.find(alias.lower())
            if alias_pos >= 0:
                context = desc_lower[max(0, alias_pos - 50):alias_pos + len(alias) + 50]
                if any(ch.isdigit() for ch in context):
                    user_specified.append(c)
                    continue

        internal.append(c)

    return internal, signal_chain, user_specified


# ============================================================
# Step 3: 代码赋默认值
# ============================================================

def assign_internal_defaults(
    internal_components: list[dict],
    params_data: list[dict],
) -> list[ComponentParamResult]:
    """为内部元件自动赋默认值。

    Args:
        internal_components: 分类为 internal 的元件列表
        params_data: 批量查询的参数结果

    Returns:
        含默认值的 ComponentParamResult 列表
    """
    results = []

    # 建立 alias → params_data 索引
    params_by_alias = {}
    for p in params_data:
        alias = p.get("alias", "")
        if alias:
            params_by_alias[alias] = p

    for comp in internal_components:
        alias = comp.get("alias", "")
        icon_name = comp.get("icon_name", "")
        submodel_id = comp.get("submodel_id", "")
        library = comp.get("library", "")

        comp_result = ComponentParamResult(
            alias=alias,
            icon_name=icon_name,
            submodel_id=submodel_id,
            library=library,
            category="internal",
        )

        pd = params_by_alias.get(alias, {})
        if not pd.get("found", False):
            comp_result.all_found = False
            results.append(comp_result)
            continue

        all_params = []
        # 合并 real_params 和 int_params
        for rp in pd.get("real_params", []):
            all_params.append(ParamItem(
                varname=rp.get("varname", ""),
                title=rp.get("title", ""),
                default=str(rp.get("default", "")),
                min_val=str(rp.get("min", "")),
                max_val=str(rp.get("max", "")),
                units=rp.get("units", ""),
                assigned_value=str(rp.get("default", "")),  # 直接赋默认值
                needs_llm=False,
            ))
        for ip in pd.get("int_params", []):
            all_params.append(ParamItem(
                varname=ip.get("varname", ""),
                title=ip.get("title", ""),
                default=str(ip.get("default", "")),
                min_val=str(ip.get("min", "")),
                max_val=str(ip.get("max", "")),
                units=ip.get("units", ""),
                assigned_value=str(ip.get("default", "")),  # 直接赋默认值
                needs_llm=False,
            ))

        comp_result.params = all_params
        results.append(comp_result)

    return results


# ============================================================
# Step 4: 提取需 LLM 处理的信号链项目
# ============================================================

def extract_signal_chain_items(
    signal_chain_components: list[dict],
    params_data: list[dict],
    connections: list[dict],
    component_data: list[dict],
) -> list[ComponentParamResult]:
    """提取需 LLM 处理的信号链项目，附带连接上下文。

    为 LLM 提供必要的上下文信息:
    - 元件涉及的连接端口类型
    - 参数定义
    - 连接的对方元件信息

    Args:
        signal_chain_components: 分类为 signal_chain 的元件列表
        params_data: 批量查询的参数结果
        connections: 连接列表
        component_data: 元件端口数据

    Returns:
        ComponentParamResult 列表，needs_llm=True
    """
    results = []

    params_by_alias = {}
    for p in params_data:
        alias = p.get("alias", "")
        if alias:
            params_by_alias[alias] = p

    for comp in signal_chain_components:
        alias = comp.get("alias", "")
        icon_name = comp.get("icon_name", "")
        submodel_id = comp.get("submodel_id", "")
        library = comp.get("library", "")

        comp_result = ComponentParamResult(
            alias=alias,
            icon_name=icon_name,
            submodel_id=submodel_id,
            library=library,
            category="signal_chain",
        )

        pd = params_by_alias.get(alias, {})
        if not pd.get("found", False):
            comp_result.all_found = False
            results.append(comp_result)
            continue

        all_params = []
        # 标记所有参数需要 LLM 推断（因为涉及信号链）
        for rp in pd.get("real_params", []):
            all_params.append(ParamItem(
                varname=rp.get("varname", ""),
                title=rp.get("title", ""),
                default=str(rp.get("default", "")),
                min_val=str(rp.get("min", "")),
                max_val=str(rp.get("max", "")),
                units=rp.get("units", ""),
                assigned_value="",          # LLM 待填充
                needs_llm=True,
                llm_reason="signal_chain",
            ))
        for ip in pd.get("int_params", []):
            all_params.append(ParamItem(
                varname=ip.get("varname", ""),
                title=ip.get("title", ""),
                default=str(ip.get("default", "")),
                min_val=str(ip.get("min", "")),
                max_val=str(ip.get("max", "")),
                units=ip.get("units", ""),
                assigned_value="",          # LLM 待填充
                needs_llm=True,
                llm_reason="signal_chain",
            ))

        comp_result.params = all_params
        results.append(comp_result)

    return results


# ============================================================
# 主流程: run_parameter_pipeline()
# ============================================================

def run_parameter_pipeline(
    component_data: list[dict],
    connections: list[dict],
    user_request: str = "",
    topology_description: str = "",
) -> ParameterPipelineResult:
    """运行完整的参数代码流程。

    Args:
        component_data: 元件端口数据列表
            每项含: alias, library, submodel_id, icon_name, ports
        connections: 连接列表
            每项含: from_alias, from_port, to_alias, to_port, type, port_type_from, port_type_to
        user_request: 用户原始需求（用于识别用户指定参数）
        topology_description: 拓扑描述（用于识别特殊需求）

    Returns:
        ParameterPipelineResult
    """
    result = ParameterPipelineResult()

    # ── Step 1: 批量查询所有元件参数 ──
    params_data = load_component_params(component_data)
    if not params_data:
        result.warnings.append("参数查询失败或无元件数据")
        return result

    found_count = sum(1 for p in params_data if p.get("found", False))
    print(f"  [Param-Core] 查询了 {len(params_data)} 个元件的参数, {found_count} 个成功")

    # ── Step 2: 分类 ──
    internal, signal_chain, user_specified = classify_components(
        component_data, connections, user_request, topology_description,
    )
    print(f"  [Param-Core] 分类: {len(internal)} internal, {len(signal_chain)} signal_chain, {len(user_specified)} user_specified")

    # ── Step 3: 内部元件赋默认值 ──
    if internal:
        result.internal_assignments = assign_internal_defaults(internal, params_data)
        defaults_count = sum(
            len(a.params) for a in result.internal_assignments
        )
        print(f"  [Param-Core] 为 {len(result.internal_assignments)} 个内部元件赋了 {defaults_count} 个默认参数")

    # ── Step 4: 信号链提取 ──
    if signal_chain:
        result.signal_chain_items = extract_signal_chain_items(
            signal_chain, params_data, connections, component_data,
        )
        print(f"  [Param-Core] 提取了 {len(result.signal_chain_items)} 个信号链元件待 LLM 处理")

    # ── Step 5: 用户指定元件（也需 LLM 处理） ──
    if user_specified:
        # 用户指定的元件也按信号链方式处理（需 LLM 推断）
        user_items = extract_signal_chain_items(
            user_specified, params_data, connections, component_data,
        )
        for item in user_items:
            item.category = "user_specified"
            for p in item.params:
                p.llm_reason = "user_specified"
        result.user_specified_items = user_items
        print(f"  [Param-Core] 提取了 {len(result.user_specified_items)} 个用户指定元件待 LLM 处理")

    return result


# ============================================================
# 工具: 构建 LLM prompt 用信号链上下文
# ============================================================

def build_signal_chain_prompt(
    signal_chain_items: list[ComponentParamResult],
    connections: list[dict],
    component_data: list[dict],
    user_request: str = "",
    topology_description: str = "",
) -> str:
    """为 LLM 构建信号链参数处理 prompt。

    Args:
        signal_chain_items: 需 LLM 处理的元件（含参数定义）
        connections: 连接列表
        component_data: 元件端口数据
        user_request: 用户原始需求
        topology_description: 拓扑描述

    Returns:
        LLM prompt 文本
    """
    # 构建信号链元件的简化视图
    items_view = []
    for item in signal_chain_items:
        # 找到该元件的连接信息
        connected_ports = _get_connected_port_types_for_alias(
            item.alias, connections, component_data,
        )

        # 找到连接的对方元件
        peers = []
        for conn in connections:
            if conn.get("from_alias") == item.alias:
                peers.append(f"→ {conn.get('to_alias')}:{conn.get('to_port')} ({conn.get('port_type_to', '')})")
            if conn.get("to_alias") == item.alias:
                peers.append(f"← {conn.get('from_alias')}:{conn.get('from_port')} ({conn.get('port_type_from', '')})")

        params_view = []
        for p in item.params:
            params_view.append({
                "varname": p.varname,
                "title": p.title,
                "default": p.default,
                "min": p.min_val,
                "max": p.max_val,
                "units": p.units,
            })

        items_view.append({
            "alias": item.alias,
            "icon_name": item.icon_name,
            "submodel_id": item.submodel_id,
            "library": item.library,
            "category": item.category,
            "connected_ports": connected_ports,
            "connected_to": peers,
            "params": params_view,
        })

    prompt = f"""请为以下涉及信号链/数据传递的元件设置参数值:

## 用户需求
{user_request or '(无特殊需求)'}

## 拓扑描述
{topology_description or '(无)'}

## 需设置参数的元件 ({len(items_view)} 个)
{json.dumps(items_view, ensure_ascii=False, indent=2)}

## 参数设置规则
1. **信号链元件**: 根据端口连接的物理意义推断参数值
   - 传感器元件: 根据被测物理量设置量程/增益
   - 信号源元件: 根据下游需求设置输出值
   - 控制器元件: 根据控制逻辑设置PID等参数
2. **用户指定元件**: 根据用户需求中的数值描述设置参数
3. 参数值必须在 min/max 范围内
4. 不确定的参数使用 default 值
5. params 中的 value 必须是字符串

## 输出格式
只输出参数设置 JSON:
```json
{{
  "assignments": [
    {{
      "alias": "Sensor_X",
      "icon_name": "sensor_xxx",
      "submodel_id": "XXX000",
      "params": {{"gain": "100", "offset": "0"}},
      "params_detail": [
        {{"varname": "gain", "title": "增益", "value": "100", "default": "1", "units": "null", "reasoning": "根据输入范围推算"}}
      ]
    }}
  ],
  "warnings": []
}}
```
"""
    return prompt


# ============================================================
# 工具: 合并代码默认值 + LLM信号链结果 → 完整参数
# ============================================================

def merge_parameter_results(
    internal_results: list[ComponentParamResult],
    signal_chain_results: list[ComponentParamResult],
    llm_assignments: list[dict],
) -> tuple[dict[str, str], list[dict]]:
    """合并代码默认值 + LLM 信号链结果 → 完整参数。

    Args:
        internal_results: 代码赋默认值的结果
        signal_chain_results: 信号链元件（参数定义）
        llm_assignments: LLM 返回的信号链参数设置 [{alias, icon_name, submodel_id, params, params_detail}, ...]

    Returns:
        (params_dict, params_detail_list)
        params_dict: {"varname@alias": "value", ...}
        params_detail_list: [{varname, title, value, ...}, ...]
    """
    params_dict: dict[str, str] = {}
    params_detail: list[dict] = []

    # 合并代码默认值
    for comp in internal_results:
        for p in comp.params:
            if p.assigned_value:
                params_dict[f"{p.varname}@{comp.alias}"] = p.assigned_value
                params_detail.append({
                    "alias": comp.alias,
                    "varname": p.varname,
                    "title": p.title,
                    "value": p.assigned_value,
                    "default": p.default,
                    "min": p.min_val,
                    "max": p.max_val,
                    "units": p.units,
                    "reasoning": "代码默认值 (内部元件)",
                })

    # 合并 LLM 信号链结果
    llm_by_alias = {}
    for a in llm_assignments:
        llm_by_alias[a.get("alias", "")] = a

    for comp in signal_chain_results:
        llm_data = llm_by_alias.get(comp.alias, {})
        llm_params = llm_data.get("params", {})
        llm_details = llm_data.get("params_detail", [])

        for p in comp.params:
            value = llm_params.get(p.varname, p.default)
            params_dict[f"{p.varname}@{comp.alias}"] = value

            detail = {
                "alias": comp.alias,
                "varname": p.varname,
                "title": p.title,
                "value": value,
                "default": p.default,
                "min": p.min_val,
                "max": p.max_val,
                "units": p.units,
                "reasoning": "LLM推断 (信号链/用户指定)",
            }
            # 尝试从 LLM detail 中获取 reasoning
            for ld in llm_details:
                if ld.get("varname") == p.varname:
                    detail["reasoning"] = ld.get("reasoning", detail["reasoning"])
                    break
            params_detail.append(detail)

    return params_dict, params_detail
