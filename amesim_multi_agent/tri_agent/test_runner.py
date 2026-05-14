"""
参数运行 Agent 独立测试入口

用法:
    python -m amesim_multi_agent.tri_agent.test_runner model.json
    python -m amesim_multi_agent.tri_agent.test_runner model.json --build-only
    python -m amesim_multi_agent.tri_agent.test_runner model.json --stop-time 20 --interval 0.001
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from agents import Runner

from ..model_providers import create_run_config
from .agents.parameter_runner import create_parameter_runner_agent
from .topology_graph import TorsionBarModel


async def run_parameter_runner_test(
    model: TorsionBarModel,
    build_only: bool = False,
    stop_time: str = "10",
    interval: str = "0.01",
    output_file: str | None = None,
) -> dict | None:
    """测试参数运行 Agent。"""
    agent = create_parameter_runner_agent()
    run_config = create_run_config()

    action = "build only" if build_only else "full build + simulate"

    prompt = f"""Process the following model definition ({action}).

MODEL DEFINITION (TorsionBar format):
{model.to_json()}

STOP_TIME: {stop_time}
INTERVAL: {interval}
BUILD_ONLY: {str(build_only).lower()}

Steps:
1. Assign submodels where missing
2. Set parameters (use defaults for internal components, LLM inference for signal chains)
3. Build via AmeBuilder
{"4. Run simulation" if not build_only else ""}
{"5. Extract and analyze results" if not build_only else ""}

Output ONLY the Parameter-Runner result JSON."""

    print(f"\n{'='*70}")
    print(f"  Parameter-Runner Agent — Test Mode")
    print(f"  Model: {model.model_name}")
    print(f"  Components: {len(model.components)} | Connections: {len(model.connections)}")
    print(f"  Mode: {'build-only' if build_only else 'full'}")
    print(f"{'='*70}\n")

    if not model.is_complete():
        print("[ERROR] Model is incomplete — missing components or connections.")
        return None

    try:
        result = await Runner.run(agent, prompt, run_config=run_config)
    except Exception as e:
        print(f"\n[ERROR] Parameter-Runner failed: {e}")
        return None

    raw = result.final_output or ""
    print(f"[RAW OUTPUT]\n{raw[:500]}...\n")

    try:
        parsed = _extract_json(raw)
    except ValueError as e:
        print(f"[PARSE ERROR] {e}")
        return None

    # Print summary
    br = parsed.get("build_result", {})
    ps = parsed.get("parameter_summary", {})
    sr = parsed.get("simulation_results", {})

    print(f"  Build: {br.get('status', '?')} ({br.get('build_attempts', '?')} attempts)")
    print(f"  Params: {ps.get('params_set', '?')} set ({ps.get('code_defaults', '?')} defaults, {ps.get('llm_inferred', '?')} inferred)")
    if sr:
        print(f"  Simulation: {sr.get('status', '?')}")
        kv = sr.get("key_variables", [])
        for v in kv[:5]:
            print(f"    {v.get('alias')}.{v.get('variable')} = {v.get('final_value')} {v.get('units')}")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to: {output_file}")

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
    parser = argparse.ArgumentParser(description="Parameter-Runner Agent — Test Mode")
    parser.add_argument("model_file", help="TorsionBar JSON model file")
    parser.add_argument("--build-only", action="store_true", help="Build only, skip simulation")
    parser.add_argument("--stop-time", default="10", help="Simulation stop time")
    parser.add_argument("--interval", default="0.01", help="Sampling interval")
    parser.add_argument("--output", "-o", default="_runner_result.json")
    args = parser.parse_args()

    # 检查 Amesim 环境
    ame_python = os.environ.get("AME", "D:/AMESIM24/Amesim") + "/win64/AMEPython.exe"
    ame_python = ame_python.replace("\\", "/")
    if not os.path.exists(ame_python) and not args.build_only:
        print(f"[WARNING] AMEPython.exe not found at {ame_python}")
        print("  Build will be simulated. Use --build-only for safe testing.")

    with open(args.model_file, encoding="utf-8") as f:
        model = TorsionBarModel.from_json(f.read())

    result = asyncio.run(
        run_parameter_runner_test(
            model=model,
            build_only=args.build_only,
            stop_time=args.stop_time,
            interval=args.interval,
            output_file=args.output,
        )
    )

    if result:
        build_ok = result.get("build_result", {}).get("status") == "success"
        sim_ok = result.get("simulation_results", {}).get("status") != "failed"
        if build_ok and (args.build_only or sim_ok):
            print(f"\n✓ Parameter-Runner test PASSED")
            sys.exit(0)

    print(f"\n✗ Parameter-Runner test FAILED")
    sys.exit(1)


if __name__ == "__main__":
    main()
