r"""
独立测试: 质量-弹簧-阻尼系统 (Mass-Spring-Damper)

不依赖 LLM / Agents SDK，直接使用 amesim_builder 构建模型。
用于验证整个构建流水线是否能正常工作。

运行方式:
    "%AME%\python" multi_agent\amesim_multi_agent\test_mass_spring_damper.py

如果不在 Amesim Python 环境中，将跳过实际构建，仅输出预生成的 model_spec。
"""
import json
import sys
import os

# 项目根
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# =====================================================================
# 模型规格定义
# =====================================================================
MODEL_SPEC = {
    "model_name": "MassSpringDamper",
    "components": [
        {
            "icon_name": "mass_friction_endstops",
            "alias": "Mass",
            "position": [400, 200],
            "submodel": "MAS000",
            "library": "libmec",
            "rotations": 0,
        },
        {
            "icon_name": "spring01",
            "alias": "Spring",
            "position": [200, 200],
            "submodel": "LSTP00A",
            "library": "libmec",
            "rotations": 0,
        },
        {
            "icon_name": "damper01",
            "alias": "Damper",
            "position": [600, 200],
            "submodel": "LSTP00A",
            "library": "libmec",
            "rotations": 0,
        },
        {
            "icon_name": "zerospeedsource",
            "alias": "Ground_Left",
            "position": [50, 200],
            "submodel": "V001",
            "library": "libmec",
            "rotations": 0,
        },
        {
            "icon_name": "zerospeedsource",
            "alias": "Ground_Right",
            "position": [800, 200],
            "submodel": "V001",
            "library": "libmec",
            "rotations": 0,
        },
        {
            "icon_name": "signal03",
            "alias": "Force_Signal",
            "position": [400, 400],
            "submodel": "UD00",
            "library": "libsig",
            "rotations": 0,
        },
        {
            "icon_name": "forcecon",
            "alias": "Actuator",
            "position": [400, 300],
            "submodel": "FORC",
            "library": "libmec",
            "rotations": 0,
        },
    ],
    "connections": [
        {
            "from_alias": "Ground_Left",
            "from_port": 0,
            "to_alias": "Spring",
            "to_port": 0,
            "type": "line",
            "line_alias": "wire_gnd_spr",
            "waypoints": [[100, 200], [150, 200]],
        },
        {
            "from_alias": "Spring",
            "from_port": 1,
            "to_alias": "Mass",
            "to_port": 0,
            "type": "line",
            "line_alias": "wire_spr_mass",
            "waypoints": [[250, 200], [350, 200]],
        },
        {
            "from_alias": "Mass",
            "from_port": 1,
            "to_alias": "Damper",
            "to_port": 0,
            "type": "line",
            "line_alias": "wire_mass_dmp",
            "waypoints": [[500, 200], [550, 200]],
        },
        {
            "from_alias": "Damper",
            "from_port": 1,
            "to_alias": "Ground_Right",
            "to_port": 0,
            "type": "line",
            "line_alias": "wire_dmp_gnd",
            "waypoints": [[700, 200], [750, 200]],
        },
        {
            "from_alias": "Force_Signal",
            "from_port": 0,
            "to_alias": "Actuator",
            "to_port": 0,
            "type": "line",
            "line_alias": "wire_sig_act",
            "waypoints": [[400, 380], [400, 350]],
        },
        {
            "from_alias": "Actuator",
            "from_port": 1,
            "to_alias": "Mass",
            "to_port": 0,
            "type": "line",
            "line_alias": "wire_act_mass",
            "waypoints": [[400, 280], [400, 250]],
        },
    ],
    "parameters": {
        "mass@Mass": "10",
        "k@Spring": "1000",
        "b@Damper": "50",
        "amplitude@Force_Signal": "100",
    },
    "stop_time": "10",
    "interval": "0.01",
}


def print_model_spec():
    """打印模型规格（安全模式，不需要 Amesim 环境）。"""
    print("=" * 60)
    print("  质量-弹簧-阻尼系统 Model Spec")
    print("=" * 60)
    print()
    print(json.dumps(MODEL_SPEC, ensure_ascii=False, indent=2))
    print()
    print("系统拓扑:")
    print("  Ground_Left → Spring → Mass → Damper → Ground_Right")
    print("                           ↑")
    print("                    Actuator ← Force_Signal")
    print()
    print("运行方式:")
    print(f'  "%AME%\\python" {__file__}')


def build_and_run():
    """实际构建并运行模型（需要 Amesim Python 环境）。"""
    from amesim_builder.models import Component, Connection, LineConnection, CircuitDefinition
    from amesim_builder.core import AmeBuilder

    spec = MODEL_SPEC
    model_name = spec["model_name"]

    # 创建模型目录 并 切换工作目录（在此之后创建 / 保存的 .ame 文件都会落在 model_dir）
    model_dir = os.path.join(PROJECT_ROOT, "ame_models", model_name)
    os.makedirs(model_dir, exist_ok=True)
    os.chdir(model_dir)

    # 构建 Component 列表
    components = []
    for c in spec["components"]:
        lib = c.get("library", "")
        sp = f"$AME/{lib}/submodels" if lib else ""
        components.append(Component(
            icon_name=c["icon_name"],
            alias=c["alias"],
            position=tuple(c["position"]),
            submodel=c.get("submodel"),
            submodel_path=sp,
            rotations=c.get("rotations", 0),
            flip=c.get("flip", False),
        ))

    # 构建 Connection 列表
    connections = []
    for conn in spec["connections"]:
        connections.append(LineConnection(
            from_component=conn["from_alias"],
            from_port=conn["from_port"],
            to_component=conn["to_alias"],
            to_port=conn["to_port"],
            line_alias=conn["line_alias"],
            waypoints=tuple(tuple(p) for p in conn["waypoints"]),
        ))

    circuit = CircuitDefinition(
        name=model_name,
        components=components,
        connections=connections,
    )

    # 执行构建
    print("=" * 60)
    print(f"  构建模型: {model_name}")
    print("=" * 60)

    with AmeBuilder() as builder:
        builder.build_from_definition(circuit)
        builder.check_and_assign_submodels()

        # 保存到模型目录（已在 os.chdir(model_dir) 中）
        builder.save()

    print()
    print("=" * 60)
    print(f"  模型构建完成!")
    print(f"  文件: {os.path.join(model_dir, model_name + '.ame')}")
    print("=" * 60)
    print()
    print("下一步:")
    print(f"  1. 在 Amesim GUI 中打开: {model_name}.ame")
    print(f"  2. 设置参数后运行仿真 (Simulation → Run)")


if __name__ == "__main__":
    # 检测是否在 Amesim Python 环境中
    try:
        import ame_apy  # noqa: F401
        IN_AMESIM = True
    except ImportError:
        IN_AMESIM = False

    if IN_AMESIM:
        build_and_run()
    else:
        print("[安全模式] 不在 Amesim Python 环境中，仅输出 model_spec。")
        print()
        print_model_spec()
