"""
连接 Agent 独立测试入口

用法:
    # Phase 1 only (核心元件选择)
    python -m amesim_multi_agent.tri_agent.test_connector topo.json --phase1

    # Phase 2 only (螺旋展开, 需先有 core_components)
    python -m amesim_multi_agent.tri_agent.test_connector topo.json --phase2 --core core_result.json

    # Full two-phase run
    python -m amesim_multi_agent.tri_agent.test_connector topo.json --full

    # Inline JSON
    python -m amesim_multi_agent.tri_agent.test_connector --inline '{...}' --phase1
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from agents import Runner

from ..model_providers import create_run_config
from .agents.connector import (
    create_connector_phase1_agent,
    create_connector_phase2_agent,
)
from .topology_graph import TopologyGraph, TorsionBarModel


async def run_connector_phase1_test(
    topology_graph: TopologyGraph,
    output_file: str | None = None,
) -> dict | None:
    """测试连接 Agent Phase 1: 核心元件选择。"""
    agent = create_connector_phase1_agent()
    run_config = create_run_config()

    # 将 TopologyGraph 的核心节点提取为 ComponentRequirement 格式
    core_nodes = topology_graph.core_nodes or topology_graph.nodes[:3]
    requirements = []
    for i, node in enumerate(core_nodes):
        requirements.append({
            "index": i + 1,
            "functional_description": node.functional_description,
            "physical_domain": node.domain,
            "suggested_library": node.suggested_library,
            "quantity": node.quantity,
            "keywords": [node.role, node.label.replace("_", " "), *node.extracted_attrs.keys()],
            "topological_role": node.topological_role,
        })

    prompt = f"""Phase 1: Select core Amesim components for the following topology.

TOPOLOGY GRAPH:
{topology_graph.to_json()}

CORE NODE REQUIREMENTS:
{json.dumps(requirements, ensure_ascii=False, indent=2)}

For each core node, search and select the best Amesim component through 4-dimension verification.
Output ONLY the Phase 1 result JSON."""

    print(f"\n{'='*70}")
    print(f"  Connector Agent — Phase 1 Test")
    print(f"  Model: {topology_graph.model_name}")
    print(f"  Core nodes: {len(core_nodes)}")
    print(f"{'='*70}\n")

    try:
        result = await Runner.run(agent, prompt, run_config=run_config)
    except Exception as e:
        print(f"\n[ERROR] Connector Phase 1 failed: {e}")
        return None

    raw = result.final_output or ""
    print(f"[RAW OUTPUT]\n{raw[:500]}...\n")

    try:
        parsed = _extract_json(raw)
    except ValueError as e:
        print(f"[PARSE ERROR] {e}")
        return None

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        print(f"Saved to: {output_file}")

    return parsed


async def run_connector_phase2_test(
    topology_graph: TopologyGraph,
    phase1_result: dict,
    output_file: str | None = None,
) -> dict | None:
    """测试连接 Agent Phase 2: 螺旋展开。"""
    agent = create_connector_phase2_agent()
    run_config = create_run_config()

    prompt = f"""Phase 2: Spiral expansion from core components.

TOPOLOGY GRAPH:
{topology_graph.to_json()}

PHASE 1 RESULT (core components):
{json.dumps(phase1_result, ensure_ascii=False, indent=2)}

Detect dangling ports and expand outward layer by layer until all ports are closed.
Output ONLY the Phase 2 result JSON."""

    print(f"\n{'='*70}")
    print(f"  Connector Agent — Phase 2 Test")
    print(f"  Model: {topology_graph.model_name}")
    print(f"{'='*70}\n")

    try:
        result = await Runner.run(agent, prompt, run_config=run_config)
    except Exception as e:
        print(f"\n[ERROR] Connector Phase 2 failed: {e}")
        return None

    raw = result.final_output or ""
    print(f"[RAW OUTPUT]\n{raw[:500]}...\n")

    try:
        parsed = _extract_json(raw)
    except ValueError as e:
        print(f"[PARSE ERROR] {e}")
        return None

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        print(f"Saved to: {output_file}")

    return parsed


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    import re
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        return json.loads(match.group(1))
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError("Cannot extract JSON from output")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Connector Agent — Test Mode")
    parser.add_argument("topo_file", nargs="?", help="TopologyGraph JSON file")
    parser.add_argument("--inline", help="Inline TopologyGraph JSON string")
    parser.add_argument("--phase1", action="store_true", help="Run Phase 1 only")
    parser.add_argument("--phase2", action="store_true", help="Run Phase 2 only")
    parser.add_argument("--full", action="store_true", help="Run both phases")
    parser.add_argument("--core", help="Phase 1 result JSON file (for Phase 2 input)")
    parser.add_argument("--output", "-o", default="_connector_result.json")
    args = parser.parse_args()

    # Load TopologyGraph
    if args.inline:
        graph = TopologyGraph.from_json(args.inline)
    elif args.topo_file:
        with open(args.topo_file, encoding="utf-8") as f:
            graph = TopologyGraph.from_json(f.read())
    else:
        parser.print_help()
        sys.exit(1)

    if args.phase1 or args.full:
        result = asyncio.run(run_connector_phase1_test(graph, args.output))
        if result:
            print(f"\n✓ Connector Phase 1 PASSED")
            # Save for Phase 2
            with open("_phase1_result.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    if args.phase2 or args.full:
        if args.core:
            with open(args.core, encoding="utf-8") as f:
                phase1 = json.load(f)
        elif args.full:
            # Use Phase 1 result from above or saved file
            try:
                with open("_phase1_result.json", encoding="utf-8") as f:
                    phase1 = json.load(f)
            except FileNotFoundError:
                print("No Phase 1 result found. Run --phase1 first.")
                sys.exit(1)
        else:
            print("Need --core <phase1_result.json> for Phase 2.")
            sys.exit(1)

        result = asyncio.run(run_connector_phase2_test(graph, phase1, args.output))
        if result:
            print(f"\n✓ Connector Phase 2 PASSED")

    if not (args.phase1 or args.phase2 or args.full):
        parser.print_help()


if __name__ == "__main__":
    main()
