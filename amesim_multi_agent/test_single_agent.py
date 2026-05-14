"""
独立测试: 单个 Agent 运行

不跑完整流水线，单独测试某个 Agent 的输出。
用于快速验证 Agent prompt + 工具调用是否正常。

运行方式 (在 multi_agent/ 目录下):

    # 测试 Selector（元件选型）
    uv run python -m amesim_multi_agent.test_single_agent selector

    # 测试 Bridge（连接规划）
    uv run python -m amesim_multi_agent.test_single_agent bridge

    # 测试 Orchestrator（拓扑规划）
    uv run python -m amesim_multi_agent.test_single_agent orchestrator "构建一个简单液压系统"

    # 交互模式 — 输入自定义 prompt
    uv run python -m amesim_multi_agent.test_single_agent selector --interactive

模式说明:
    - 默认模式: 使用内置的测试 prompt 运行 Agent，输出 JSON 和分析
    - 交互模式: 输入自定义 prompt，适合探索性测试
"""
from __future__ import annotations

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents import Runner
from .model_providers import create_kimi_run_config

# ============================================================
# 各 Agent 内置测试 Prompt
# ============================================================

SELECTOR_TEST_PROMPT = """请根据以下拓扑规划结果，从知识库中选择合适的元件。

## 拓扑规划
{
  "system_description": "简单液压系统: 液压泵驱动液压缸，溢流阀保护系统，换向阀控制方向，油箱回油",
  "topology_description": "Pump出口 → ReliefValve入口 + DirectionalValve P口; DirectionalValve A口 → Cylinder A口; DirectionalValve B口 → Cylinder B口; DirectionalValve T口 → Tank入口; ReliefValve出口 → Tank入口; Pump入口 ← Tank出口",
  "component_requirements": [
    {"index": 1, "functional_description": "液压泵", "topological_role": "动力源", "physical_domain": "hydraulic", "suggested_library": "libhydr", "keywords": ["hydraulic pump", "pump", "constant displacement pump"]},
    {"index": 2, "functional_description": "溢流阀", "topological_role": "安全保护", "physical_domain": "hydraulic", "suggested_library": "libhydr", "keywords": ["relief valve", "pressure relief valve", "safety valve"]},
    {"index": 3, "functional_description": "三位四通换向阀", "topological_role": "方向控制", "physical_domain": "hydraulic", "suggested_library": "libhydr", "keywords": ["directional valve", "4 way 3 position valve", "directional control valve"]},
    {"index": 4, "functional_description": "双作用液压缸", "topological_role": "执行元件", "physical_domain": "hydraulic", "suggested_library": "libhydr", "keywords": ["hydraulic cylinder", "double acting cylinder", "hydraulic actuator"]},
    {"index": 5, "functional_description": "液压油箱", "topological_role": "储油容器", "physical_domain": "hydraulic", "suggested_library": "libhydr", "keywords": ["hydraulic tank", "oil tank", "reservoir"]}
  ]
}

请第一步使用 batch_search_and_select 批量搜索（传入全部5个需求），
然后检查端口数（换向阀需≥4端口，双作用缸需≥2端口），最后输出 ComponentSelection JSON。"""

BRIDGE_TEST_PROMPT = """请对以下Selector选定的元件进行三步审查和连接规划:

## 已选定的元件
[
  {"alias": "Pump", "icon_key": "libhydr:pump01", "icon_name": "pump01", "library": "libhydr", "submodel_id": "PU001"},
  {"alias": "ReliefValve", "icon_key": "libhydr:rv00", "icon_name": "rv00", "library": "libhydr", "submodel_id": "RV000"},
  {"alias": "DirectionalValve", "icon_key": "libhydr:hsv_4port3pos", "icon_name": "hsv_4port3pos", "library": "libhydr", "submodel_id": "HSV34"},
  {"alias": "Cylinder", "icon_key": "libhydr:actuator001", "icon_name": "actuator001", "library": "libhydr", "submodel_id": "HJ010"},
  {"alias": "Tank", "icon_key": "libhydr:tank01", "icon_name": "tank01", "library": "libhydr", "submodel_id": "TK000"}
]

## 拓扑关系描述
Pump出口 → ReliefValve入口 + DirectionalValve P口;
DirectionalValve A口 → Cylinder A口;
DirectionalValve B口 → Cylinder B口;
DirectionalValve T口 → Tank入口;
ReliefValve出口 → Tank入口;
Pump入口 ← Tank出口

## 系统描述
简单液压系统: 液压泵驱动液压缸，溢流阀保护系统，换向阀控制方向，油箱回油

请按 Step 1(存在性审查) → Step 2(分支检测与三通插入) → Step 3(端口规划) → Step 4(跨域桥接) 的流程，输出 ConnectionPlan JSON。
注意: Pump出口需要分叉到两个目标，必须调用 find_junction_component 插入液压三通。"""

ORCHESTRATOR_TEST_PROMPT = """请对以下用户需求进行分解和拓扑规划:

用户需求: {user_request}

请严格按照系统指令中定义的JSON格式输出规划结果，必须包含 physical_flow_diagram 字段。"""

DIAGNOSIS_TEST_PROMPT = """请诊断以下子Agent错误:

阶段: bridge
错误类型: validation_failed
错误描述: 连接验证不通过: [PORT_MULTIPLE_CONN] Pump.port1 同时连接了 ReliefValve.port0 和 DirectionalValve.port0
受影响元件: Pump, ReliefValve, DirectionalValve
连续失败次数: 1

请调用 diagnose_error 工具进行分析，然后输出 DiagnosisReport JSON。"""

PARAMETER_TEST_PROMPT = """请为以下元件查询并设置参数:

## 元件列表
[
  {"alias": "Mass", "library": "libmec", "submodel_id": "MAS000"},
  {"alias": "Spring", "library": "libmec", "submodel_id": "SPR000"},
  {"alias": "Damper", "library": "libmec", "submodel_id": "DAM000"}
]

## 用户需求
质量-弹簧-阻尼系统: 质量10kg, 弹簧刚度1000N/m, 阻尼100Ns/m

请使用 batch_query_params 工具查询所有元件的参数，然后根据物理常识推断合理的参数值。
输出 ParameterAssignment JSON。"""

# ============================================================
# 测试函数
# ============================================================

async def test_agent(agent_name: str, user_request: str = "", interactive: bool = False):
    """测试单个 Agent"""

    run_cfg = create_kimi_run_config()

    # 创建 Agent
    if agent_name == "selector":
        from .agents.selector import create_selector_agent
        agent = create_selector_agent()
        default_prompt = SELECTOR_TEST_PROMPT
        max_turns = 120
    elif agent_name == "bridge":
        from .agents.bridge import create_bridge_agent
        agent = create_bridge_agent()
        default_prompt = BRIDGE_TEST_PROMPT
        max_turns = 80
    elif agent_name == "orchestrator":
        from .agents.orchestrator import create_orchestrator_agent
        agent = create_orchestrator_agent()
        default_prompt = ORCHESTRATOR_TEST_PROMPT.format(
            user_request=user_request or "构建一个液压泵驱动双作用缸的简单液压系统"
        )
        max_turns = 60
    elif agent_name == "diagnosis":
        from .agents.diagnosis import create_diagnosis_agent
        agent = create_diagnosis_agent()
        default_prompt = DIAGNOSIS_TEST_PROMPT
        max_turns = 30
    elif agent_name == "parameter":
        from .agents.parameter import create_parameter_agent
        agent = create_parameter_agent()
        default_prompt = PARAMETER_TEST_PROMPT
        max_turns = 60
    elif agent_name == "builder":
        from .agents.builder import create_builder_agent
        agent = create_builder_agent()
        default_prompt = "Builder Agent 目前使用程序化构建 (build_model_direct)，LLM 测试暂不支持。建议直接运行完整流水线。"
        print(default_prompt)
        return
    else:
        print(f"未知 Agent: {agent_name}")
        print(f"可用 Agent: selector, bridge, orchestrator, diagnosis, parameter, builder")
        return

    # 确定 prompt
    if interactive:
        print(f"\n{'='*60}")
        print(f"  交互模式 — [{agent_name.upper()}] Agent 测试")
        print(f"{'='*60}")
        print(f"请输入 prompt (输入空行结束, Ctrl+C 退出):\n")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print("\n")
        prompt = "\n".join(lines).strip()
        if not prompt:
            prompt = default_prompt
            print(f"使用默认 prompt...\n")
    else:
        prompt = default_prompt

    # 运行
    print(f"\n{'='*60}")
    print(f"  [{agent_name.upper()}] Agent 测试")
    print(f"{'='*60}")
    print(f"\n  Prompt 前 200 字符: {prompt[:200]}...\n")

    print(f"  [执行中...]")
    try:
        result = await Runner.run(agent, prompt, run_config=run_cfg, max_turns=max_turns)
        output = result.final_output
        print(f"  [完成] output 长度: {len(output)} chars\n")
    except Exception as e:
        print(f"  [失败] {e}")
        return

    print(f"{'='*60}")
    print(f"  Agent 输出:")
    print(f"{'='*60}\n")
    print(output)
    print()

    # 尝试解析 JSON 并分析
    print(f"{'='*60}")
    print(f"  分析:")
    print(f"{'='*60}")
    try:
        data = json.loads(output)
        if agent_name == "selector":
            sel = data.get("selected", [])
            fail = data.get("failures", [])
            print(f"  选中元件: {len(sel)} 个")
            for s in sel:
                alias = s.get("alias", "?")
                icon = s.get("icon_key", "?")
                ports = s.get("port_count", 0)
                print(f"    [{s.get('requirement_index','?')}] {alias} → {icon} (端口数={ports})")
            if fail:
                print(f"  失败: {len(fail)} 个")
                for f in fail:
                    print(f"    [{f.get('requirement_index','?')}] {f.get('functional_description','?')}: {f.get('reason','?')}")
        elif agent_name == "bridge":
            conns = data.get("connections", [])
            bridges = data.get("bridge_components", [])
            is_valid = data.get("is_valid", False)
            issues = data.get("validation_issues", [])
            print(f"  连接数: {len(conns)}")
            print(f"  桥接元件数: {len(bridges)}")
            print(f"  验证状态: {'✅ 通过' if is_valid else '❌ 未通过'}")
            for b in bridges:
                print(f"    桥接: {b.get('alias','?')} ({b.get('icon_key','?')}) — {b.get('reason','?')[:60]}")
            for c in conns[:5]:
                print(f"    {c.get('from_alias','?')}.port{c.get('from_port','?')} → {c.get('to_alias','?')}.port{c.get('to_port','?')}")
            if issues:
                print(f"  问题: {len(issues)} 个")
                for i in issues[:3]:
                    print(f"    [{i.get('code','?')}] {str(i)[:100]}")
        elif agent_name == "orchestrator":
            desc = data.get("system_description", "")
            comps = data.get("component_requirements", [])
            print(f"  系统描述: {desc[:100]}")
            print(f"  元件需求: {len(comps)} 个")
            for c in comps:
                print(f"    [{c.get('index','?')}] {c.get('functional_description','?')} ({c.get('suggested_library','?')})")
        elif agent_name == "diagnosis":
            errors = data.get("errors", [])
            print(f"  错误数: {len(errors)}")
            for e in errors:
                print(f"    [{e.get('code','?')}] {e.get('message','?')[:100]}")
                print(f"    目标Agent: {e.get('target_agent','?')}")
                print(f"    建议: {e.get('suggested_fix','?')[:100]}")
        elif agent_name == "parameter":
            assigns = data.get("assignments", [])
            print(f"  参数设置: {len(assigns)} 个元件")
            for a in assigns:
                params = a.get("params", {})
                print(f"    {a.get('alias','?')}: {len(params)} 个参数")
                for k, v in list(params.items())[:3]:
                    print(f"      {k}={v}")
    except json.JSONDecodeError:
        print("  ⚠️ 输出不是有效 JSON，无法分析")

    print()
    print(f"  Raw output 已显示在上方。")


# ============================================================
# CLI 入口
# ============================================================
def _check_api_key():
    from .model_providers import DEEPSEEK_API_KEY, KIMI_API_KEY
    ds_key = os.environ.get("DEEPSEEK_API_KEY") or DEEPSEEK_API_KEY
    kimi_key = os.environ.get("KIMI_API_KEY") or KIMI_API_KEY
    if not (ds_key or kimi_key):
        print("错误: 未配置任何 API Key")
        sys.exit(1)
    print(f"[API] 可用 Key: {'DeepSeek' if ds_key else ''}{' + ' if ds_key and kimi_key else ''}{'Kimi' if kimi_key else ''}")


async def main():
    _check_api_key()

    if len(sys.argv) < 2:
        print("用法: uv run python -m amesim_multi_agent.test_single_agent <agent_name> [需求描述] [--interactive]")
        print()
        print("Agent 名称: selector, bridge, orchestrator, diagnosis, parameter")
        print()
        print("示例:")
        print('  uv run python -m amesim_multi_agent.test_single_agent selector')
        print('  uv run python -m amesim_multi_agent.test_single_agent bridge')
        print('  uv run python -m amesim_multi_agent.test_single_agent orchestrator "构建一个质量-弹簧-阻尼系统"')
        print('  uv run python -m amesim_multi_agent.test_single_agent selector --interactive')
        sys.exit(1)

    agent_name = sys.argv[1].lower()
    interactive = "--interactive" in sys.argv

    # 过滤掉标志参数
    args = []
    skip_next = False
    for a in sys.argv[2:]:
        if skip_next:
            skip_next = False
            continue
        if a == "--interactive":
            continue
        args.append(a)
    user_request = " ".join(args)

    await test_agent(agent_name, user_request, interactive)


if __name__ == "__main__":
    asyncio.run(main())
