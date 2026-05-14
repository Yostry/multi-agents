"""
Parameter Agent 工具 — 查询子模型参数信息

提供:
  - query_submodel_params: 查询单个子模型的所有可设参数
  - batch_query_params: 批量查询多个子模型的参数
"""
from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from agents import function_tool


def _get_component_file(library: str) -> dict | None:
    """加载指定库的元件参数文件"""
    from pathlib import Path
    p = (Path(__file__).resolve().parent.parent.parent.parent /
         "knowledge_base" / "json" / "components" / f"{library}.json")
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_param(param: dict) -> dict:
    """格式化参数为Agent友好的结构"""
    return {
        "varname": param.get("varname", ""),
        "title": param.get("title", ""),
        "default": param.get("default", ""),
        "min": param.get("min", ""),
        "max": param.get("max", ""),
        "units": param.get("units", ""),
        "visibility": param.get("visibility", "True"),
    }


@function_tool
def query_submodel_params(library: str, submodel_id: str) -> str:
    """查询指定子模型的所有可设置参数。

    从 knowledge_base 中获取参数详情包括:
    varname(变量名), title(说明), default(默认值), min/max(范围), units(单位)

    Args:
        library: 库名，如 "libmec", "libsig", "libtpf", "libth"
        submodel_id: 子模型ID，如 "MAS000", "LSTP00A", "UD00"

    Returns:
        JSON格式的参数列表
    """
    comp_data = _get_component_file(library)
    if not comp_data:
        return json.dumps({
            "submodel_id": submodel_id,
            "library": library,
            "found": False,
            "message": f"未找到库文件: knowledge_base/json/components/{library}.json",
            "real_params": [],
            "int_params": [],
        }, ensure_ascii=False, indent=2)

    submodels = comp_data.get("submodels", {})
    sm_entry = submodels.get(submodel_id)
    if not sm_entry:
        return json.dumps({
            "submodel_id": submodel_id,
            "library": library,
            "found": False,
            "message": f"在{library}中未找到子模型: {submodel_id}",
            "available_submodels": list(submodels.keys())[:20],
            "real_params": [],
            "int_params": [],
        }, ensure_ascii=False, indent=2)

    real_params = [_format_param(p) for p in sm_entry.get("real_params", [])]
    int_params = [_format_param(p) for p in sm_entry.get("int_params", [])]

    return json.dumps({
        "submodel_id": submodel_id,
        "library": library,
        "icon_name": sm_entry.get("icon_name", ""),
        "label": sm_entry.get("label", ""),
        "found": True,
        "real_params_count": len(real_params),
        "int_params_count": len(int_params),
        "real_params": real_params,
        "int_params": int_params,
    }, ensure_ascii=False, indent=2)


@function_tool
def batch_query_params(components_json: str) -> str:
    """批量查询多个元件的子模型参数。

    Args:
        components_json: JSON数组，每项含 library, submodel_id, alias
        例如: [{"library": "libmec", "submodel_id": "MAS000", "alias": "Mass"}]

    Returns:
        JSON格式的批量查询结果
    """
    components = json.loads(components_json)
    results = []

    for comp in components:
        lib = comp.get("library", "")
        sm = comp.get("submodel_id", "")
        alias = comp.get("alias", "?")
        result_str = query_submodel_params(lib, sm)
        result = json.loads(result_str)
        result["alias"] = alias
        results.append(result)

    total_params = sum(r.get("real_params_count", 0) for r in results)
    all_found = all(r.get("found", False) for r in results)

    return json.dumps({
        "total_components": len(results),
        "all_found": all_found,
        "total_params": total_params,
        "results": results,
    }, ensure_ascii=False, indent=2)
