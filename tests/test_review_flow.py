"""
测试人工审核流程 — 独立运行，不需要 Kimi API

用法: python multi_agent/tests/test_review_flow.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amesim_multi_agent.protocols import (
    TopologyPlan,
    ComponentRequirement,
)

# ── 构造一个模拟的 TopologyPlan ──
mock_plan = TopologyPlan(
    user_request="构建一个质量块-弹簧-阻尼器系统",
    system_description="一维机械振动系统。质量块受弹簧力和阻尼力作用，基座固定在地面。",
    physical_domains=["mechanical_1d", "signal"],
    component_requirements=[
        ComponentRequirement(
            index=1,
            functional_description="固定基座 (零速度参考点)",
            physical_domain="mechanical_1d",
            suggested_library="libmec",
            keywords=["固定", "ground", "zero speed"],
            topological_role="sink",
        ),
        ComponentRequirement(
            index=2,
            functional_description="线性弹簧",
            physical_domain="mechanical_1d",
            suggested_library="libmec",
            keywords=["弹簧", "spring", "linear"],
            topological_role="transfer",
        ),
        ComponentRequirement(
            index=3,
            functional_description="质量块 (带位移传感器输出)",
            physical_domain="mechanical_1d",
            suggested_library="libmec",
            keywords=["质量块", "mass", "displacement"],
            topological_role="storage",
        ),
        ComponentRequirement(
            index=4,
            functional_description="线性阻尼器",
            physical_domain="mechanical_1d",
            suggested_library="libmec",
            keywords=["阻尼器", "damper", "linear"],
            topological_role="transfer",
        ),
        ComponentRequirement(
            index=5,
            functional_description="固定基座 (第二个地面参考点)",
            physical_domain="mechanical_1d",
            suggested_library="libmec",
            keywords=["固定", "ground", "zero speed"],
            topological_role="sink",
        ),
        ComponentRequirement(
            index=6,
            functional_description="阶跃力输入信号",
            physical_domain="signal",
            suggested_library="libsig",
            keywords=["信号", "signal", "step", "force"],
            topological_role="source",
        ),
    ],
    topology_description=(
        "Ground_L → Spring → Mass → Damper → Ground_R; "
        "信号源连接Mass上的力输入端口"
    ),
    physical_flow_diagram="""\
┌───────────────────────────────────────────────────────────────┐
│  质量块-弹簧-阻尼器 物理拓扑图                                  │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │Ground_L  │──→│ Spring   │──→│  Mass    │──→│ Damper   │  │
│  │(Sink:机) │   │(R:机械域)│   │(C:机械域)│   │(R:机械域)│  │
│  └──────────┘   └──────────┘   └────┬─────┘   └────┬─────┘  │
│                                     │               │        │
│                              ┌──────▼──┐     ┌──────▼──┐     │
│                              │Sig_Force│     │Ground_R │     │
│                              │(Source) │     │(Sink:机)│     │
│                              └─────────┘     └─────────┘     │
│                                                                │
│  物理域: mechanical_1d, signal                                 │
│  因果链: Sink → R → C → R → Sink + Signal→C                   │
└───────────────────────────────────────────────────────────────┘""",
    notes="弹簧刚度默认 1000 N/m，阻尼系数 10 N·s/m，质量 1 kg",
)


# ── 内联审核函数 (避免导入 orchestration_runner 的重依赖) ──
def _display_topology_plan(plan):
    """以可读格式显示拓扑规划结果"""
    print("\n" + "=" * 70)
    print("  📋 物理拓扑规划 — 请审核")
    print("=" * 70)
    print(f"\n  ▸ 系统概述:")
    for line in plan.system_description.split("\n"):
        print(f"    {line}")
    print(f"\n  ▸ 物理域: {', '.join(plan.physical_domains)}")
    if plan.physical_flow_diagram:
        print(f"\n  ▸ 物理流程图:")
        for line in plan.physical_flow_diagram.split("\n"):
            print(f"    {line}")
    print(f"\n  ▸ 拓扑关系: {plan.topology_description}")
    print(f"\n  ▸ 元件需求清单 ({len(plan.component_requirements)} 个):")
    for req in plan.component_requirements:
        lib = req.suggested_library or "—"
        print(f"    [{req.index}] {req.functional_description}")
        print(f"        角色={req.topological_role}, 域={req.physical_domain}, 库={lib}")
        print(f"        关键词: {', '.join(req.keywords)}")
    if plan.notes:
        print(f"\n  ▸ 备注: {plan.notes}")


def _review_topology_plan(plan, plan_data):
    """人工审核拓扑规划。返回 (approved, feedback)"""
    _display_topology_plan(plan)
    print("\n  ─────────────────────────────────────────")
    while True:
        choice = input("\n  [A] 审核通过, 继续  |  [E] 提供修改意见  |  [Q] 退出\n  > ").strip().upper()
        if choice == "A":
            print("\n  ✅ 审核通过，进入元件选型阶段...")
            return True, ""
        elif choice == "Q":
            print("\n  ⏹ 用户退出审核")
            return False, ""
        elif choice == "E":
            feedback = input("\n  请输入修改意见 (直接回车取消):\n  > ").strip()
            if not feedback:
                continue
            print(f"\n  🔄 收到修改意见，重新规划...")
            return False, feedback
        else:
            print("  无效选项，请输入 A / E / Q")


# ── 导入审核函数 ──
# (使用上面内联定义的版本)

print("=" * 70)
print("  测试1: _display_topology_plan() — 格式化展示")
print("=" * 70)
_display_topology_plan(mock_plan)

print("\n" + "=" * 70)
print("  测试2: 用 unittest.mock 模拟用户输入 [A] 通过")
print("=" * 70)

from unittest.mock import patch

with patch("builtins.input", return_value="A"):
    approved, feedback = _review_topology_plan(mock_plan, {})
    print(f"\n  结果: approved={approved}, feedback='{feedback}'")
    assert approved, "审核应该通过!"
    assert feedback == "", "feedback应为空!"

print("\n  ✅ 模拟[A]通过 — 测试通过")

print("\n" + "=" * 70)
print("  测试3: 模拟用户输入 [E] 修改 → 提供意见")
print("=" * 70)

with patch("builtins.input", side_effect=["E", "增加一个位移传感器输出"]):
    approved, feedback = _review_topology_plan(mock_plan, {})
    print(f"\n  结果: approved={approved}, feedback='{feedback}'")
    assert not approved, "审核不应该通过!"
    assert "位移传感器" in feedback, "feedback应包含修改意见!"

print("  ✅ 模拟[E]修改 — 测试通过")

print("\n" + "=" * 70)
print("  测试4: 模拟用户输入 [Q] 退出")
print("=" * 70)

with patch("builtins.input", return_value="Q"):
    approved, feedback = _review_topology_plan(mock_plan, {})
    print(f"\n  结果: approved={approved}, feedback='{feedback}'")
    assert not approved, "审核不应该通过!"
    assert feedback == "", "退出时feedback应为空!"

print("  ✅ 模拟[Q]退出 — 测试通过")

print("\n" + "=" * 70)
print("  测试5: 模拟无效输入后选 [A] (容错测试)")
print("=" * 70)

with patch("builtins.input", side_effect=["X", "", "a"]):
    approved, feedback = _review_topology_plan(mock_plan, {})
    print(f"\n  结果: approved={approved}, feedback='{feedback}'")
    assert approved, "最终应通过!"

print("  ✅ 无效输入容错 — 测试通过")

print("\n" + "=" * 70)
print("  全部5项测试通过! ✅")
print("=" * 70)
