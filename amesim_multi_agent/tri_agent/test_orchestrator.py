"""
主管 Agent 独立测试入口

用法:
    python -m amesim_multi_agent.tri_agent.test_orchestrator "构建一个液氮储存系统"

    # 跳过人工审核
    python -m amesim_multi_agent.tri_agent.test_orchestrator "需求" --skip-review

    # 指定输出文件
    python -m amesim_multi_agent.tri_agent.test_orchestrator "需求" --output topo.json

    # 从文件读取需求
    python -m amesim_multi_agent.tri_agent.test_orchestrator --file requirement.txt

    # 交互模式 (多轮对话)
    python -m amesim_multi_agent.tri_agent.test_orchestrator --interactive
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from agents import Runner
from agents.run import RunConfig

from ..model_providers import create_run_config
from .agents.orchestrator import create_tri_orchestrator_agent
from .topology_graph import TopologyGraph
from .experience_memory import ExperienceMemory


def _build_test_prompt(user_request: str, skip_review: bool = False) -> str:
    """构建测试用的输入 prompt。"""
    return f"""Please analyze the following modeling request and generate a TopologyGraph.

USER REQUEST:
{user_request}

{"SKIP_REVIEW: Auto-approve is ON. Do not ask clarifying questions — make reasonable assumptions." if skip_review else "If critical information is missing, ask up to 3 clarifying questions."}

Output ONLY the TopologyGraph JSON. No markdown, no explanation outside the JSON."""


async def run_orchestrator_test(
    user_request: str,
    skip_review: bool = False,
    output_file: str | None = None,
    run_config: RunConfig | None = None,
    interactive: bool = False,
) -> TopologyGraph | None:
    """运行主管 Agent 独立测试。

    Returns:
        TopologyGraph if successful, None if failed.
    """
    agent = create_tri_orchestrator_agent()
    run_config = run_config or create_run_config()

    prompt = _build_test_prompt(user_request, skip_review)

    print(f"\n{'='*70}")
    print(f"  Orchestrator Agent — Test Mode")
    print(f"  Request: {user_request[:100]}...")
    print(f"  Skip Review: {skip_review}")
    print(f"{'='*70}\n")

    if interactive:
        # 交互模式: 多轮对话, 支持用户回答追问
        return await _interactive_loop(agent, user_request, run_config)

    try:
        result = await Runner.run(
            agent,
            prompt,
            run_config=run_config,
        )
    except Exception as e:
        print(f"\n[ERROR] Orchestrator call failed: {e}")
        return None

    raw_output = result.final_output or ""
    print(f"\n[RAW OUTPUT]\n{raw_output[:500]}...\n")

    # 解析 JSON
    try:
        graph_dict = _extract_json(raw_output)
        graph = TopologyGraph.from_dict(graph_dict)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"[PARSE ERROR] Failed to parse TopologyGraph: {e}")
        print(f"Raw output saved to _orchestrator_raw_output.txt")
        with open("_orchestrator_raw_output.txt", "w", encoding="utf-8") as f:
            f.write(raw_output)
        return None

    # 打印结果
    print(graph.summary())
    print(f"\n[TOPOLOGY GRAPH]")
    print(f"  Nodes ({len(graph.nodes)}):")
    for n in graph.nodes:
        marker = "★" if n.is_core else " "
        print(f"    {marker} {n.node_id}: {n.label} [{n.role}] domain={n.domain}")
    print(f"  Edges ({len(graph.edges)}):")
    for e in graph.edges:
        cross = " (CROSS-DOMAIN)" if e.is_cross_domain else ""
        print(f"    {e.edge_id}: {e.source_node_id} → {e.target_node_id} [{e.flow_type}]{cross}")
    if graph.unresolved:
        print(f"  Unresolved ({len(graph.unresolved)}):")
        for u in graph.unresolved:
            print(f"    Q: {u['question']}")

    # 保存
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(graph.to_json())
        print(f"\nSaved to: {output_file}")

    # 保存到 PipelineContext 兼容路径
    ctx_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "_topology_graph.json"
    )
    with open(ctx_path, "w", encoding="utf-8") as f:
        f.write(graph.to_json())

    return graph


async def _interactive_loop(agent, user_request, run_config):
    """交互式多轮对话。"""
    print("[Interactive Mode] Type your answers to agent questions.")
    print("Type 'done' to finish, 'skip' to skip a question.\n")

    prompt = user_request
    graph = None

    for turn in range(5):  # Max 5 turns
        result = await Runner.run(agent, prompt, run_config=run_config)
        raw = result.final_output or ""

        # Try to parse as final graph
        try:
            graph_dict = _extract_json(raw)
            graph = TopologyGraph.from_dict(graph_dict)
            if graph.nodes:
                print(f"\n[FINAL GRAPH RECEIVED] {len(graph.nodes)} nodes")
                return graph
        except Exception:
            pass

        # If not a graph, it's a question for the user
        print(f"\n[AGENT]: {raw}")
        user_input = input("\n[YOU]: ").strip()
        if user_input.lower() in ("done", "exit", "quit"):
            break
        elif user_input.lower() == "skip":
            prompt = f"The user skipped this question. Proceed with reasonable assumptions.\n{raw}"
        else:
            prompt = f"User's answer: {user_input}\n\nContinue planning."

    return graph


def _extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON 对象。"""
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 查找 ```json ... ``` 代码块
    import re
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        return json.loads(match.group(1))

    # 查找 { ... } 最外层
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError("Cannot extract JSON from output")


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Orchestrator Agent — Test Mode",
    )
    parser.add_argument(
        "request",
        nargs="?",
        help="Natural language modeling request",
    )
    parser.add_argument(
        "--file", "-f",
        help="Read request from file",
    )
    parser.add_argument(
        "--output", "-o",
        default="_topology_graph_test.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Skip human review gate",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive multi-turn mode",
    )

    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            request = f.read().strip()
    elif args.request:
        request = args.request
    else:
        parser.print_help()
        sys.exit(1)

    graph = asyncio.run(
        run_orchestrator_test(
            user_request=request,
            skip_review=args.skip_review,
            output_file=args.output,
            interactive=args.interactive,
        )
    )

    if graph:
        print(f"\n[OK] Orchestrator test PASSED: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        sys.exit(0)
    else:
        print(f"\n[FAIL] Orchestrator test FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
