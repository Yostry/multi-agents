"""
预定义角色词库生成器

从 45 个库、8660 个子模型的 description 和用途字段中
离线生成英文 role 标签集合, 供主管 Agent 在规划时选择。

输出: role_vocabulary.json — 按 topological_role 分组的 role → library mapping

用法:
    python -m amesim_multi_agent.tri_agent.role_vocabulary.build_vocab
    python -m amesim_multi_agent.tri_agent.role_vocabulary.build_vocab --output role_vocab.json
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

# 知识库路径
_KB_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "knowledge_base"
)
_REGISTRY_PATH = os.path.join(_KB_ROOT, "json", "components", "_registry.json")

# 域 → 库映射
DOMAIN_LIB_MAP = {
    "mechanical_1d": "libmec",
    "mechanical_rotary": "libmec",
    "two_phase_flow": "libtpf",
    "thermal": "libth",
    "hydraulic": "libhydr",
    "pneumatic": "libpn",
    "signal": "libsig",
    "electric": "libeb",
    "thermal_hydraulic": "libthh",
    "cooling": "libcs",
    "planar_mechanical": "libplm",
    "air_conditioning": "libac",
    "engine": "libeng",
    "fuel_cell": "libfc",
    "electrochemistry": "libec",
    "gas_mixture": "libgm",
    "filling": "libfi",
    "aircraft_fuel": "libacf",
}

# topological_role → 典型 role 关键词映射
ROLE_KEYWORD_MAP = {
    "source": [
        "source", "supply", "generator", "driver", "input",
        "constant", "prescribed", "imposed",
    ],
    "sink": [
        "sink", "atmosphere", "ambient", "ground", "drain", "exhaust",
        "tank", "reservoir", "receiver", "outlet",
    ],
    "storage": [
        "chamber", "volume", "tank", "accumulator", "capacitance",
        "inertia", "mass", "body", "storage", "capacity",
    ],
    "transfer": [
        "pipe", "orifice", "valve", "restrictor", "conduit", "duct",
        "heat_exchanger", "conduction", "convection", "radiator",
        "spring", "damper", "link", "gear", "belt", "wire",
        "pump", "compressor", "turbine", "motor",
    ],
    "sensor": [
        "sensor", "transducer", "probe", "gauge",
        "measure", "detect", "monitor",
    ],
    "actuator": [
        "actuator", "cylinder", "piston", "motor", "heater", "cooler",
        "converter", "transformer", "pump", "fan", "compressor",
    ],
    "junction": [
        "junction", "tee", "node", "manifold", "splitter", "mixer",
        "collector", "divider", "branch", "connector",
    ],
    "control": [
        "controller", "pid", "gain", "integrator", "differentiator",
        "comparator", "switch", "limiter", "saturation",
        "function", "table", "lookup", "map", "profile",
        "constant", "sine", "step", "ramp", "pulse", "signal",
        "filter", "delay", "sample", "hold",
    ],
}


def build_role_vocabulary(
    registry_path: str = "",
    output_path: str = "",
    verbose: bool = False,
) -> dict:
    """从知识库生成预定义角色词库。

    Returns:
        {
            "source": {"hydraulic": {"libraries": [...], "common_icons": [...]}, ...},
            "sink": {...},
            ...
        }
    """
    registry_path = registry_path or _REGISTRY_PATH

    if not os.path.exists(registry_path):
        print(f"[ERROR] Registry not found: {registry_path}")
        print("Run knowledge base build scripts first.")
        return {}

    with open(registry_path, encoding="utf-8") as f:
        reg = json.load(f)

    libraries = reg.get("libraries", [])
    icon_index = reg.get("icon_index", {})

    # 按 topological_role 和 domain 分组
    vocab: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"libraries": set(), "common_icons": set(), "total": 0})
    )

    for icon_name, entries in icon_index.items():
        for entry in entries:
            lib_name = entry.get("lib", "")
            label = entry.get("label", "").lower()
            submodel_id = entry.get("id", "")

            # 推断 topological_role
            role = _classify_role(label, submodel_id, icon_name)

            # 推断 physical domain
            domain = _classify_domain(lib_name)

            if role and domain:
                vocab[role][domain]["libraries"].add(lib_name)
                vocab[role][domain]["common_icons"].add(icon_name)
                vocab[role][domain]["total"] += 1

    # 转换 set → list (JSON 序列化)
    result = {}
    for role, domains in vocab.items():
        result[role] = {}
        for domain, data in domains.items():
            result[role][domain] = {
                "libraries": sorted(list(data["libraries"])),
                "common_icons": sorted(list(data["common_icons"]))[:30],
                "total_submodels": data["total"],
            }

    # 保存
    if output_path:
        output = output_path
    else:
        output = os.path.join(os.path.dirname(__file__), "role_vocabulary.json")

    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[Role Vocab] Generated {len(result)} roles across {sum(len(d) for d in result.values())} domains")
    print(f"[Role Vocab] Saved to: {output}")

    # 打印摘要
    if verbose:
        for role, domains in sorted(result.items()):
            print(f"\n  [{role}]")
            for domain, data in sorted(domains.items()):
                print(f"    {domain}: {data['total_submodels']} submodels, "
                      f"{', '.join(data['libraries'])}")

    return result


def _classify_role(label: str, submodel_id: str, icon_name: str) -> str:
    """根据 label/submodel_id/icon_name 推断 topological_role。"""
    text = f"{label} {icon_name}".lower()

    scores: dict[str, int] = {}
    for role, keywords in ROLE_KEYWORD_MAP.items():
        score = 0
        for kw in keywords:
            if kw in text:
                score += 1
        if score > 0:
            scores[role] = score

    if not scores:
        return "transfer"  # 默认归为传输类

    return max(scores, key=scores.get)


def _classify_domain(lib_name: str) -> str:
    """根据库名推断物理域。"""
    for domain, lib in DOMAIN_LIB_MAP.items():
        if lib == lib_name:
            return domain
    # 特殊映射
    special = {
        "libmec": "mechanical_1d",
        "libsig": "signal",
        "libhydr": "hydraulic",
        "libtpf": "two_phase_flow",
        "libth": "thermal",
        "libpn": "pneumatic",
        "libeb": "electric",
        "libthh": "thermal_hydraulic",
        "libcs": "cooling",
    }
    return special.get(lib_name, "")


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build role vocabulary from knowledge base"
    )
    parser.add_argument("--output", "-o", default="",
                        help="Output JSON file path")
    parser.add_argument("--registry", default="",
                        help="Registry JSON path (default: KB default)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    build_role_vocabulary(
        registry_path=args.registry,
        output_path=args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
