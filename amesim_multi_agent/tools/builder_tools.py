"""
模型构建与运行工具 — 封装 amesim_builder.core 为 Agent @function_tool

提供:
  - build_and_run_model: 根据 JSON 规格构建 Amesim 模型并运行仿真
  - take_model_screenshot: 在 Amesim GUI 中打开模型并截图

注意: 这些工具必须在 Amesim Python 环境中运行 (%AME%/python)。
"""
from __future__ import annotations

import json
import subprocess
import time
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from agents import function_tool

# ===== 常量 =====
AMESIM_EXE = os.environ.get("AMESIM_EXE", r"D:\AMESIM24\Amesim\win64\AMESim.exe")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
AME_MODELS_DIR = PROJECT_ROOT / "ame_models"


@function_tool
def generate_build_script_only(model_spec: str) -> str:
    """仅生成Python构建脚本，不执行Amesim API。

    用于手动模式：生成脚本后由用户决定何时在Amesim Python环境中执行。

    Args:
        model_spec: 模型规格 JSON 字符串 (与 build_and_run_model 格式相同)

    Returns:
        JSON 格式的结果 (status, script_path, message)
    """
    spec = json.loads(model_spec)
    model_name = spec["model_name"]
    mode = spec.get("mode", "manual")

    # 创建模型目录
    model_dir = AME_MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # 生成构建脚本
    script_path = model_dir / f"build_{model_name.lower().replace(' ', '_')}.py"
    _generate_build_script(spec, script_path, model_dir)

    # Amesim API Python 路径
    #   终端里 %AME%\python 实际调用的是 python.bat → sys/python/win64/python.exe
    #   但要使用 ame_apy 模块构建模型，必须用 AMEPython.exe (win64/AMEPython.exe)
    AME_PYTHON_EXE = os.path.join(os.path.dirname(AMESIM_EXE), "AMEPython.exe")
    AME_PYTHON_EXE = os.path.normpath(AME_PYTHON_EXE)

    return json.dumps({
        "status": "build_only",
        "model_name": model_name,
        "mode": mode,
        "script_path": str(script_path),
        "stdout": "",
        "stderr": "",
        "returncode": 0,
        "message": (
            f"构建脚本已生成: {script_path}\n"
            f"手动运行方式: \"{AME_PYTHON_EXE}\" {script_path}"
        ),
    }, ensure_ascii=False, indent=2)


@function_tool
def repair_build_script(script_path: str, error_info: str, fixes: str) -> str:
    """读取现有构建脚本并准备修复内容，供 Builder 修复循环使用。

    本工具是 READER 类型 — 不修改原文件，仅读取并返回内容供 LLM 分析。
    Builder 修复循环逻辑:
      1. 执行构建脚本捕获错误
      2. 调用本工具读取脚本 + 错误信息 + 修复建议
      3. LLM 根据返回内容生成修复后的脚本
      4. Builder 将修复后的脚本写回或执行

    Args:
        script_path: 现有构建脚本的完整路径
        error_info: 构建执行时的错误信息
        fixes: 修复建议或说明

    Returns:
        JSON 格式的修复包 (script_content, error_info, applied_fixes)
    """
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            script_content = f.read()
    except FileNotFoundError:
        return json.dumps({
            "script_content": "",
            "error_info": error_info,
            "applied_fixes": fixes,
            "error": f"脚本文件未找到: {script_path}",
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "script_content": "",
            "error_info": error_info,
            "applied_fixes": fixes,
            "error": f"读取脚本失败: {str(e)}",
        }, ensure_ascii=False, indent=2)

    return json.dumps({
        "script_content": script_content,
        "error_info": error_info,
        "applied_fixes": fixes,
    }, ensure_ascii=False, indent=2)


def _build_and_run_model_impl(model_spec: str) -> str:
    """纯函数 — 根据模型规格 JSON 构建并运行 Amesim 仿真模型。

    供 build_model_direct (程序化路径) 和 build_and_run_model (@function_tool 包装) 共同调用。

    规格 JSON 格式:
    {
        "model_name": "MyModel",
        "mode": "auto",
        "components": [
            {
                "icon_name": "mass_friction_endstops",
                "alias": "Mass",
                "position": [100, 50],
                "submodel": "MAS000",
                "library": "libmec",
                "rotations": 0
            }
        ],
        "connections": [
            {
                "from_alias": "Mass", "from_port": 1,
                "to_alias": "Spring", "to_port": 0,
                "type": "line",
                "waypoints": [[200, 50], [250, 80]]
            }
        ],
        "bridge_components": [],
        "parameters": {"mass@Mass": "10"},
        "stop_time": "10",
        "interval": "0.01"
    }

    mode=auto 时自动执行Amesim API构建；mode=manual 时仅生成脚本。

    Args:
        model_spec: 如上格式的 JSON 字符串

    Returns:
        JSON 格式的执行结果 (status, stdout, stderr, ame_path, message 或 error)
    """
    spec = json.loads(model_spec)
    model_name = spec["model_name"]
    mode = spec.get("mode", "auto")

    # 创建模型目录
    model_dir = AME_MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # 生成 Python 构建脚本 (始终生成)
    script_path = model_dir / f"build_{model_name.lower().replace(' ', '_')}.py"
    _generate_build_script(spec, script_path, model_dir)

    # 手动模式: 只生成脚本
    if mode == "manual":
        return json.dumps({
            "status": "build_only",
            "model_name": model_name,
            "mode": mode,
            "script_path": str(script_path),
            "stdout": "",
            "stderr": "",
            "returncode": 0,
            "ame_path": f"{model_dir / model_name}.ame (未生成，需手动运行脚本)",
            "message": f"构建脚本已生成: {script_path}。请手动运行: \"%AME%\\python\" {script_path}",
        }, ensure_ascii=False, indent=2)

    # 自动模式: 通过 Amesim API Python (AMEPython.exe) 执行构建脚本
    #   AMEPython.exe 内含 ame_apy 模块，不需要额外环境变量
    ame_python = os.path.join(os.path.dirname(AMESIM_EXE), "AMEPython.exe")
    ame_python = os.path.normpath(ame_python)

    if not os.path.exists(ame_python):
        # 尝试环境变量
        ame_python = os.environ.get("AME_PYTHON", "")
        if not ame_python or not os.path.exists(ame_python):
            return json.dumps({
                "status": "error",
                "model_name": model_name,
                "mode": mode,
                "script_path": str(script_path),
                "stdout": "",
                "stderr": (
                    f"未找到 Amesim API Python (AMEPython.exe)。\n"
                    f"尝试路径: {ame_python}\n"
                    f"请确认 Amesim 2024 已正确安装。\n"
                    f"或手动运行: \"{ame_python}\" {script_path}"
                ),
                "returncode": -1,
                "message": "未找到Amesim Python解释器",
            }, ensure_ascii=False, indent=2)

    try:
        result = subprocess.run(
            [ame_python, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(model_dir),
        )
        output = result.stdout
        error_out = result.stderr

        if result.returncode == 0:
            ame_path = model_dir / f"{model_name}.ame"
            return json.dumps({
                "status": "success",
                "model_name": model_name,
                "mode": mode,
                "ame_path": str(ame_path),
                "script_path": str(script_path),
                "stdout": output,
                "stderr": error_out,
                "returncode": 0,
                "message": f"模型 {model_name} 构建成功。模型文件: {ame_path}",
            }, ensure_ascii=False, indent=2)
        else:
            return json.dumps({
                "status": "error",
                "model_name": model_name,
                "mode": mode,
                "ame_path": "",
                "script_path": str(script_path),
                "stdout": output,
                "stderr": error_out,
                "returncode": result.returncode,
                "message": f"模型 {model_name} 构建失败 (returncode={result.returncode})",
            }, ensure_ascii=False, indent=2)

    except subprocess.TimeoutExpired:
        return json.dumps({
            "status": "error",
            "model_name": model_name,
            "mode": mode,
            "stdout": "",
            "stderr": "仿真超时 (300秒)，请检查模型复杂度或 stop_time 设置",
            "returncode": -1,
            "message": "构建超时",
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "model_name": model_name,
            "mode": mode,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "message": f"构建异常: {e}",
        }, ensure_ascii=False, indent=2)


@function_tool
def build_and_run_model(model_spec: str) -> str:
    """根据模型规格 JSON 构建并运行 Amesim 仿真模型。

    Agent 工具入口 — 内部委托给 _build_and_run_model_impl。
    """
    return _build_and_run_model_impl(model_spec)


def _generate_build_script(spec: dict, script_path: Path, model_dir: Path):
    """从规格 JSON 生成独立的 Python 构建脚本。"""
    model_name = spec["model_name"]

    lines = [
        '"""Auto-generated Amesim build script."""',
        "import os",
        "import sys",
        "# 添加项目根到 sys.path",
        f"sys.path.insert(0, r'{PROJECT_ROOT}')",
        f"os.chdir(r'{model_dir}')",
        "",
        "from amesim_builder.models import (",
        "    Component, Connection, LineConnection, CircuitDefinition,",
        ")",
        "from amesim_builder.core import AmeBuilder",
        "",
        "# ===== 元件定义 =====",
        "components = [",
    ]

    for c in spec.get("components", []):
        lib = c.get("library", "")
        sm_path = f'"$AME/{lib}/submodels"' if lib else '""'
        sm = f'"{c["submodel"]}"' if c.get("submodel") else "None"
        rot = c.get("rotations", 0)
        flip = c.get("flip", False)
        pos = tuple(c["position"])
        lines.append(
            f"    Component("
            f'icon_name="{c["icon_name"]}", '
            f'alias="{c["alias"]}", '
            f"position={pos}, "
            f"submodel={sm}, "
            f"submodel_path={sm_path}, "
            f"rotations={rot}, flip={flip}),"
        )

    lines.append("]")
    lines.append("")
    lines.append("# ===== 连接定义 =====")
    lines.append("connections = [")

    for conn in spec.get("connections", []):
        ctype = conn.get("type", "direct")
        if ctype == "line":
            wp = tuple(tuple(p) for p in conn.get("waypoints", [[0, 0]]))
            ls = f'"{conn["line_submodel"]}"' if conn.get("line_submodel") else "None"
            ll = f'"$AME/{conn["line_library"]}/submodels"' if conn.get("line_library") else "None"
            lines.append(
                f"    LineConnection("
                f'"{conn["from_alias"]}", {conn["from_port"]}, '
                f'"{conn["to_alias"]}", {conn["to_port"]}, '
                f'"{conn.get("line_alias", "line1")}", '
                f"{wp}, "
                f"line_submodel={ls}, "
                f"line_submodel_path={ll}),"
            )
        else:
            lines.append(
                f"    Connection("
                f'"{conn["from_alias"]}", {conn["from_port"]}, '
                f'"{conn["to_alias"]}", {conn["to_port"]}),'
            )

    lines.append("]")
    lines.append("")
    lines.append("# ===== 构建电路 =====")
    lines.append(f'circuit = CircuitDefinition(name="{model_name}", components=components, connections=connections)')
    lines.append("")
    lines.append("# ===== 参数 =====")
    params = spec.get("parameters", {})
    lines.append(f"params = {json.dumps(params)}")
    lines.append("")
    lines.append("# ===== 执行 =====")
    lines.append('if __name__ == "__main__":')
    lines.append("    with AmeBuilder() as builder:")
    lines.append("        builder.build_from_definition(circuit)")
    lines.append("        builder.check_and_assign_submodels()")
    lines.append("        if params:")
    lines.append("            builder.set_parameters(params)")
    lines.append("        builder.save()")
    lines.append(f'        print("\\n[DONE] 模型 {model_name} 构建完成 (未运行仿真)")')
    lines.append("")

    script_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[BuilderTool] 构建脚本已生成: {script_path}")


@function_tool
def take_model_screenshot(model_name: str) -> str:
    """在 Amesim GUI 中打开指定模型并截图。

    注意: 此操作会启动 Amesim 窗口界面，可能耗时 15-30 秒。

    Args:
        model_name: 模型名称（对应 ame_models/<ModelName>/<ModelName>.ame）

    Returns:
        JSON 格式的结果 (screenshot_path 或 error)
    """
    ame_file = AME_MODELS_DIR / model_name / f"{model_name}.ame"

    if not ame_file.exists():
        return json.dumps({
            "status": "error",
            "message": f"模型文件不存在: {ame_file}",
        }, ensure_ascii=False, indent=2)

    if not os.path.exists(AMESIM_EXE):
        return json.dumps({
            "status": "error",
            "message": f"Amesim 可执行文件不存在: {AMESIM_EXE}",
        }, ensure_ascii=False, indent=2)

    # 启动 Amesim
    proc = subprocess.Popen(
        [AMESIM_EXE, str(ame_file)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 等待 GUI 加载
    wait_seconds = 20
    print(f"[BuilderTool] 等待 Amesim GUI 加载 ({wait_seconds}s)...")
    time.sleep(wait_seconds)

    # 截图
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = AME_MODELS_DIR / model_name / f"{model_name}_screenshot_{timestamp}.png"

    ps_script = f'''
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bitmap = New-Object System.Drawing.Bitmap($screen.Width, $screen.Height)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen($screen.Location, [System.Drawing.Point]::Empty, $screen.Size)
$bitmap.Save("{screenshot_path}")
$graphics.Dispose()
$bitmap.Dispose()
'''
    temp_ps1 = AME_MODELS_DIR / model_name / "_temp_screenshot.ps1"
    temp_ps1.write_text(ps_script, encoding="utf-8")

    result = subprocess.run(
        ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", str(temp_ps1)],
        capture_output=True, text=True,
    )
    temp_ps1.unlink(missing_ok=True)

    if result.returncode == 0:
        return json.dumps({
            "status": "success",
            "model_name": model_name,
            "screenshot_path": str(screenshot_path),
        }, ensure_ascii=False, indent=2)
    else:
        return json.dumps({
            "status": "error",
            "message": f"截图失败: {result.stderr}",
        }, ensure_ascii=False, indent=2)
