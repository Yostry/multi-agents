"""
Diagnosis Agent 工具 — 构建输出解析与错误分类

提供两条路径共享的同一诊断逻辑:
  路径A (Agent流水线): 通过 @function_tool 装饰的 classify_error / parse_build_output
  路径B (独立构建脚本): 通过 classify_error_core / parse_build_output_core (纯Python, 无需agents)

核心函数 (无外部依赖, 可独立导入):
  - classify_error_core: 对单个错误消息分类, 返回 dict
  - parse_build_output_core: 解析 stdout/stderr, 返回 dict
  - print_diagnosis_report: 打印人类可读的诊断报告 (独立脚本用)

Agent 工具 (需 agents 包):
  - parse_build_output: @function_tool 包装
  - classify_error: @function_tool 包装
  - diagnose_error: @function_tool 包装 (仅 Agent 路径使用)
"""
from __future__ import annotations

import json
import re
import sys
import os

# ============================================================
# 错误模式表 — 纯数据, 无依赖, 两条路径共享
# ============================================================
_ERROR_PATTERNS: list[tuple[str, str, str]] = [
    # ICON_NOT_FOUND
    (r"(?:cannot find|cannot load|unknown|icon not found).*?(?:icon|component)",
     "ICON_NOT_FOUND", "selector"),
    (r"(?:icon|component)\s+[\"']?(\w+)[\"']?\s+(?:not found|does not exist)",
     "ICON_NOT_FOUND", "selector"),
    # SUBMODEL_NOT_FOUND
    (r"(?:submodel|sub-model)\s+[\"']?(\w+)[\"']?\s+(?:not found|does not exist|cannot)",
     "SUBMODEL_NOT_FOUND", "selector"),
    (r"cannot assign submodel",
     "SUBMODEL_NOT_FOUND", "selector"),
    # SUBMODEL_INCOMPATIBLE (子模型与元件不兼容，需Selector重新选型)
    (r"submodel.*?not compatible with component",
     "SUBMODEL_NOT_FOUND", "selector"),
    (r"submodel is not compatible",
     "SUBMODEL_NOT_FOUND", "selector"),
    # PORT_CONNECT_FAIL
    (r"(?:cannot connect|cannot link|connection failed).*?port",
     "PORT_CONNECT_FAIL", "bridge"),
    (r"port\s+\d+\s+(?:not found|does not exist|is not available)",
     "PORT_CONNECT_FAIL", "bridge"),
    (r"port already connected",
     "PORT_CONNECT_FAIL", "bridge"),
    # CAUSALITY_CONFLICT
    (r"causality (?:conflict|error|incompatible)",
     "CAUSALITY_CONFLICT", "bridge"),
    (r"ports are not compatible",
     "CAUSALITY_CONFLICT", "bridge"),
    (r"incompatible port (?:type|io)",
     "CAUSALITY_CONFLICT", "bridge"),
    # PARAM_NOT_FOUND
    (r"(?:parameter|param)\s+[\"']?(\w+)[\"']?\s+(?:not found|does not exist|unknown)",
     "PARAM_NOT_FOUND", "parameter"),
    (r"cannot set parameter",
     "PARAM_NOT_FOUND", "parameter"),
    (r"fail(?:ed)? to locate parameter",
     "PARAM_NOT_FOUND", "parameter"),
    # PARAM_OUT_OF_RANGE
    (r"(?:value|parameter).*?(?:out of range|exceeds|too large|too small)",
     "PARAM_OUT_OF_RANGE", "parameter"),
    (r"(?:min|max)(?:imum)?\s+(?:value|limit)\s+exceeded",
     "PARAM_OUT_OF_RANGE", "parameter"),
    # COMPILE_ERROR
    (r"compil(?:ation|e)\s+(?:error|failed|failure)",
     "COMPILE_ERROR", "builder"),
    (r"generate\s*code\s*(?:error|failed)",
     "COMPILE_ERROR", "builder"),
    # BUILD_FAILED — 模型构建/保存阶段错误
    (r"not fully connected",
     "BUILD_FAILED", "builder"),
    (r"fail(?:ed)? to save circuit",
     "BUILD_FAILED", "builder"),
    (r"can'?t open.*?\.sim",
     "BUILD_FAILED", "builder"),
    (r"no active circuit",
     "BUILD_FAILED", "builder"),
    (r"circuit (?:is not opened|active.*?failed)",
     "BUILD_FAILED", "builder"),
    (r"failed to set.*?circuit active",
     "BUILD_FAILED", "builder"),
    # SIMULATION_ERROR
    (r"simulation\s+(?:error|failed|failure)",
     "SIMULATION_ERROR", "builder"),
    (r"simulation failed to run",
     "SIMULATION_ERROR", "builder"),
    (r"(?:solver|integration)\s+(?:error|failed)",
     "SIMULATION_ERROR", "builder"),
    (r"division by zero",
     "SIMULATION_ERROR", "builder"),
    # LIBRARY_PATH_ERROR
    (r"(?:library|lib)\s+(?:path|not found|cannot find)",
     "LIBRARY_PATH_ERROR", "orchestrator"),
    (r"AMEAddPathsToPathsList.*?(?:error|failed)",
     "LIBRARY_PATH_ERROR", "orchestrator"),
    # BUILD_FAILED — 模型构建/保存阶段错误 (扩展)
    # 子模型不存在 (如 TPFOR04 、THHF0 等拼写/版本错误)
    (r"submodel\s+\w+\s+does not exist",
     "SUBMODEL_NOT_FOUND", "selector"),
    (r"submodel\s+\w+\s+(?:not found|is invalid|does not match)",
     "SUBMODEL_NOT_FOUND", "selector"),
    (r"cannot find submodel",
     "SUBMODEL_NOT_FOUND", "selector"),
    # 编码错误 (中文路径/非ASCII字符在Amesim C API中)
    (r"(?:unicode|ascii|utf|codec|encoding).*?(?:error|can'?t)",
     "BUILD_FAILED", "builder"),
    (r"(?:cannot|can'?t).*?(?:encode|decode)",
     "BUILD_FAILED", "builder"),
    # 元件添加失败 (API层错误)
    (r"(?:cannot|fail).*?(?:add|create).*?(?:component|circuit)",
     "BUILD_FAILED", "builder"),
    (r"AMEAddComponent.*?(?:error|fail)",
     "BUILD_FAILED", "builder"),
    # 文件/路径错误
    (r"(?:can'?t|cannot)\s+(?:create|write|save|open)\s+(?:file|circuit)",
     "BUILD_FAILED", "builder"),
    (r"(?:no such file|path not found|cannot find path)",
     "BUILD_FAILED", "builder"),
    # 信号端口方向错误 (开关/SQW的port 0是输出不是输入)
    (r"signal.*?(?:direction|input|output).*?(?:error|mismatch|wrong)",
     "PORT_CONNECT_FAIL", "bridge"),
    (r"port\s+\d+\s+is\s+(?:output|input).*?(?:cannot|error|fail)",
     "PORT_CONNECT_FAIL", "bridge"),
    # 编译前约束：端口必须连接才能编译 (热端口/流体端口)
    (r"(?:must be connected|unconnected.*?port|required.*?connection)",
     "COMPILE_ERROR", "builder"),
    (r"thermal.*?port.*?must.*?(?:connect|be\s+connected)",
     "COMPILE_ERROR", "builder"),
    # 仿真运行失败
    (r"simulation\s+(?:aborted|terminated|stopped)",
     "SIMULATION_ERROR", "builder"),
    (r"(?:initialization|init)\s+(?:error|failed)",
     "SIMULATION_ERROR", "builder"),
    (r"run\s+(?:failed|aborted|error)",
     "SIMULATION_ERROR", "builder"),
]

_WARNING_PATTERNS: list[tuple[str, str, str]] = [
    (r"(?:warning|warn)\s*[:\-]", "BUILD_WARNING", "builder"),
    (r"(?:deprecated|obsolete)", "DEPRECATED", "builder"),
    (r"default\s+(?:value|parameter).*?used", "PARAM_DEFAULT_USED", "parameter"),
]


# ============================================================
# 核心函数 — 纯 Python, 无需 agents, 两条路径共享
# ============================================================

def classify_error_core(error_message: str) -> dict:
    """对单个错误消息进行分类，确定错误类型和目标修正Agent。

    纯 Python 实现，无外部依赖，可被 Agent 工具和独立构建脚本共同调用。

    Args:
        error_message: 错误消息文本

    Returns:
        dict: {code, target_agent, confidence, error_message, [note]}
    """
    msg_lower = error_message.lower()

    for pattern, code, target in _ERROR_PATTERNS:
        if re.search(pattern, msg_lower, re.IGNORECASE):
            return {
                "code": code,
                "target_agent": target,
                "confidence": "high",
                "error_message": error_message[:200],
            }

    for pattern, code, target in _WARNING_PATTERNS:
        if re.search(pattern, msg_lower, re.IGNORECASE):
            return {
                "code": code,
                "target_agent": target,
                "confidence": "medium",
                "level": "warning",
                "error_message": error_message[:200],
            }

    # 未匹配到已知模式 — 通过关键词启发式判断
    if any(kw in msg_lower for kw in ["save", "circuit", ".sim", "active circuit",
                                         "encoding", "unicode", "ascii", "codec",
                                         "cannot create", "cannot write", "path",
                                         "no such file", "permission denied"]):
        guess_target = "builder"
        guess_code = "BUILD_FAILED"
    elif any(kw in msg_lower for kw in ["signal", "direction", "mismatch", "output port", "input port"]):
        guess_target = "bridge"
        guess_code = "PORT_CONNECT_FAIL"
    elif any(kw in msg_lower for kw in ["port", "connect"]):
        guess_target = "bridge"
        guess_code = "PORT_CONNECT_FAIL"
    elif any(kw in msg_lower for kw in ["param", "value", "locate"]):
        guess_target = "parameter"
        guess_code = "PARAM_NOT_FOUND"
    elif any(kw in msg_lower for kw in ["icon", "submodel"]):
        guess_target = "selector"
        guess_code = "SUBMODEL_NOT_FOUND"
    elif any(kw in msg_lower for kw in ["compil", "simulat", "must be connected", "unconnected",
                                         "initialization", "init fail", "aborted", "terminated"]):
        guess_target = "builder"
        guess_code = "COMPILE_ERROR"
    elif any(kw in msg_lower for kw in ["run fail", "run aborted", "run error"]):
        guess_target = "builder"
        guess_code = "SIMULATION_ERROR"
    elif any(kw in msg_lower for kw in ["add component", "create component", "AMEAddComponent"]):
        guess_target = "builder"
        guess_code = "BUILD_FAILED"
    else:
        guess_target = "builder"
        guess_code = "UNKNOWN_ERROR"

    return {
        "code": guess_code,
        "target_agent": guess_target,
        "confidence": "low",
        "error_message": error_message[:200],
        "note": "未匹配到已知错误模式，根据关键词推测",
    }


def parse_build_output_core(stdout: str, stderr: str) -> dict:
    """解析 Builder 的 stdout/stderr，提取错误和警告行。

    纯 Python 实现，无外部依赖。

    Args:
        stdout: 标准输出文本
        stderr: 标准错误文本

    Returns:
        dict: {total_lines, error_count, warning_count, errors, warnings}
    """
    combined = (stdout or "") + "\n" + (stderr or "")
    lines = combined.split("\n")

    error_lines = []
    warning_lines = []

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        if not line_lower:
            continue

        is_error = any(kw in line_lower for kw in [
            "error", "failed", "failure", "cannot", "unable",
            "exception", "traceback", "invalid",
        ])
        is_warning = any(kw in line_lower for kw in [
            "warning", "warn", "deprecated",
        ])

        if is_error and not is_warning:
            error_lines.append({
                "line_num": i + 1,
                "content": line.strip()[:200],
                "source": "stderr" if line in (stderr or "") else "stdout",
            })
        elif is_warning:
            warning_lines.append({
                "line_num": i + 1,
                "content": line.strip()[:200],
                "source": "stderr" if line in (stderr or "") else "stdout",
            })

    return {
        "total_lines": len(lines),
        "error_count": len(error_lines),
        "warning_count": len(warning_lines),
        "errors": error_lines[:20],
        "warnings": warning_lines[:10],
    }


def print_diagnosis_report(
    errors: list[str],
    stage_name: str = "build",
    show_all: bool = True,
) -> None:
    """打印人类可读的诊断报告 (独立构建脚本使用)。

    对每个错误调用 classify_error_core 分类，并按目标 Agent 汇总显示。

    Args:
        errors: 错误消息字符串列表
        stage_name: 当前阶段名称 (用于报告标题)
        show_all: 是否显示所有错误 (默认 True); False 时只显示摘要
    """
    if not errors:
        return

    print(f"\n{'='*60}")
    print(f"  🔍 Diagnosis 错误诊断报告 (阶段: {stage_name})")
    print(f"{'='*60}")
    print(f"  共 {len(errors)} 个错误\n")

    # 分类每个错误
    classified = []
    for i, err in enumerate(errors):
        result = classify_error_core(err)
        classified.append((i + 1, err, result))

    # 按目标 Agent 分组汇总
    by_agent: dict[str, list] = {}
    for num, err, result in classified:
        agent = result["target_agent"]
        by_agent.setdefault(agent, []).append((num, err, result))

    for agent, items in by_agent.items():
        codes = set(r["code"] for _, _, r in items)
        high_conf = sum(1 for _, _, r in items if r["confidence"] == "high")
        print(f"  📌 目标Agent: [{agent}]  ({len(items)} 个错误, "
              f"错误码: {', '.join(sorted(codes))}, "
              f"高置信度: {high_conf}/{len(items)})")

        if show_all:
            for num, err, result in items:
                conf_mark = "✓" if result["confidence"] == "high" else (
                    "~" if result["confidence"] == "medium" else "?"
                )
                print(f"    [{conf_mark}] #{num} [{result['code']}] {err[:120]}")

    print(f"\n  💡 建议: 根据上述分类, 优先处理高置信度错误, "
          f"按 target_agent 逐项修正。")
    print(f"{'='*60}\n")


# ============================================================
# Agent function_tool 包装 — 仅在多 Agent 环境中使用
# ============================================================

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from agents import function_tool  # noqa: E402
    _HAS_AGENTS = True
except ImportError:
    _HAS_AGENTS = False
    def function_tool(fn):  # type: ignore[no-redef]
        """无 agents 时的空装饰器桩"""
        return fn


if _HAS_AGENTS:
    @function_tool
    def parse_build_output(stdout: str, stderr: str) -> str:
        """[Agent工具] 解析Builder的stdout/stderr，提取所有错误和警告行。"""
        return json.dumps(
            parse_build_output_core(stdout, stderr),
            ensure_ascii=False, indent=2,
        )

    @function_tool
    def classify_error(error_message: str) -> str:
        """[Agent工具] 对单个错误消息进行分类，确定错误类型和目标修正Agent。"""
        return json.dumps(
            classify_error_core(error_message),
            ensure_ascii=False, indent=2,
        )

else:
    # 无 agents 环境: 导出 JSON 序列化版本 (保持接口兼容)
    def parse_build_output(stdout: str, stderr: str) -> str:
        """[独立脚本] 解析Builder的stdout/stderr。"""
        return json.dumps(
            parse_build_output_core(stdout, stderr),
            ensure_ascii=False, indent=2,
        )

    def classify_error(error_message: str) -> str:
        """[独立脚本] 对单个错误消息进行分类。"""
        return json.dumps(
            classify_error_core(error_message),
            ensure_ascii=False, indent=2,
        )


@function_tool
def diagnose_error(
    stage: str,
    error_type: str,
    error_summary: str,
    affected_components: str = "",
    suggestion: str = "",
) -> str:
    """接收任意子Agent的错误报告，生成结构化诊断结果。

    当Selector/Bridge/Parameter/Builder任一个子Agent运行出错时调用此工具。

    Args:
        stage: 出错阶段名称，如 "selector"、"bridge"、"parameter"、"builder"
        error_type: 错误大类，如 "json_parse_failed"、"validation_failed"、
                    "not_found"、"build_failed"、"max_turns_exceeded"、
                    "stage_data_missing"、"connection_failed"、"causality_conflict"、
                    "param_failed"、"simulation_failed"
        error_summary: 错误描述摘要
        affected_components: 受影响的元件列表 (逗号分隔)，如 "Mass,Spring"
        suggestion: 初步修复建议 (可选)

    Returns:
        JSON格式的诊断结果，含 error_code, target_agent, severity, action
    """
    # 根据 stage + error_type 推断错误码和目标Agent
    stage_to_agent = {
        "selector": "selector",
        "bridge": "bridge",
        "parameter": "parameter",
        "builder": "builder",
        "diagnosis": "orchestrator",
    }

    error_type_to_code = {
        "json_parse_failed": "AGENT_OUTPUT_INVALID",
        "validation_failed": "VALIDATION_FAILED",
        "not_found": "COMPONENT_NOT_FOUND",
        "build_failed": "BUILD_FAILED",
        "max_turns_exceeded": "MAX_TURNS_EXCEEDED",
        "connection_failed": "PORT_CONNECT_FAIL",
        "causality_conflict": "CAUSALITY_CONFLICT",
        "param_failed": "PARAM_NOT_FOUND",
        "stage_data_missing": "STAGE_DATA_MISSING",
        "simulation_failed": "SIMULATION_ERROR",
        "unknown_error": "UNKNOWN_ERROR",
    }

    code = error_type_to_code.get(error_type, "UNKNOWN_ERROR")
    target = stage_to_agent.get(stage, "builder")

    # 严重程度
    if error_type in ("build_failed", "max_turns_exceeded", "stage_data_missing", "simulation_failed"):
        severity = "critical"
    elif error_type in ("not_found", "validation_failed", "connection_failed"):
        severity = "error"
    else:
        severity = "warning"

    # 建议动作
    if error_type == "stage_data_missing":
        suggested_action = "replan"  # 上游阶段数据缺失，必须重新规划
    elif severity == "critical":
        suggested_action = "replan"
    elif error_type in ("not_found", "validation_failed"):
        suggested_action = "retry_with_adjustment"
    else:
        suggested_action = "retry"

    comp_list = [c.strip() for c in affected_components.split(",") if c.strip()]

    return json.dumps({
        "stage": stage,
        "error_type": error_type,
        "error_code": code,
        "target_agent": target,
        "severity": severity,
        "suggested_action": suggested_action,
        "error_summary": error_summary[:500],
        "affected_components": comp_list,
        "suggestion": suggestion[:300],
        "timestamp": "",  # 由调用方填充
    }, ensure_ascii=False, indent=2)


# ============================================================
# Python Traceback 诊断 — 解析 traceback + 读取代码上下文
# ============================================================

_TRACEBACK_FILE_RE = re.compile(
    r'File\s+"([^"]+)",\s+line\s+(\d+),\s+in\s+(\w+)'
)
_TRACEBACK_ERROR_RE = re.compile(
    r'^([A-Za-z_]\w*(?:\.\w+)*(?:Error|Exception|Warning|Exit))\s*:?\s*(.*)',
    re.MULTILINE,
)

# 已知可修复的 Python 错误模式 → (fix_strategy, description)
_KNOWN_PYTHON_BUG_PATTERNS: list[tuple[str, str, str]] = [
    (
        r"'FunctionTool'\s+object\s+is\s+not\s+callable",
        "FUNCTIONTOOL_DIRECT_CALL",
        "@function_tool 装饰的函数被直接 () 调用。修复: 抽离纯函数实现, @function_tool 包装只做代理, 程序化路径调用纯函数。"
    ),
    (
        r"non-default\s+argument.*?follows\s+default\s+argument",
        "DATACLASS_FIELD_ORDER",
        "dataclass 中无默认值字段排在了有默认值字段后面。修复: 给该字段添加 default 值, 或调整字段顺序使无默认值字段在前。"
    ),
]


def parse_python_traceback(tb_text: str) -> dict:
    """解析 Python traceback 文本, 提取结构化错误信息。

    Args:
        tb_text: Python traceback 完整文本

    Returns:
        dict: {
            error_type, error_message, error_file, error_line,
            error_function, stack_frames, full_traceback, matched_bug_type
        }
    """
    # 提取所有 File 引用
    stack_frames: list[dict] = []
    for m in _TRACEBACK_FILE_RE.finditer(tb_text):
        stack_frames.append({
            "file": m.group(1),
            "line": int(m.group(2)),
            "function": m.group(3),
        })

    # 提取最终错误类型和消息
    error_type = "Unknown"
    error_message = tb_text.strip().split("\n")[-1] if tb_text.strip() else ""
    for m in _TRACEBACK_ERROR_RE.finditer(tb_text):
        error_type = m.group(1)
        error_message = m.group(2).strip()

    # 错误发生的文件 (最后一个 File 引用)
    error_file = None
    error_line = 0
    error_function = ""
    if stack_frames:
        last = stack_frames[-1]
        error_file = last["file"]
        error_line = last["line"]
        error_function = last["function"]

    # 匹配已知 bug 模式
    matched_bug_type = None
    matched_bug_desc = ""
    for pattern, bug_type, description in _KNOWN_PYTHON_BUG_PATTERNS:
        if re.search(pattern, tb_text, re.IGNORECASE):
            matched_bug_type = bug_type
            matched_bug_desc = description
            break

    return {
        "error_type": error_type,
        "error_message": error_message[:500],
        "error_file": error_file,
        "error_line": error_line,
        "error_function": error_function,
        "stack_frames": stack_frames,
        "full_traceback": tb_text[:3000],
        "matched_bug_type": matched_bug_type,
        "matched_bug_description": matched_bug_desc,
    }


def read_code_context(file_path: str, line_num: int, context_lines: int = 10) -> str:
    """读取指定文件中某一行附近的代码上下文。

    Args:
        file_path: 文件绝对路径
        line_num: 目标行号 (1-based)
        context_lines: 目标行前后各取多少行

    Returns:
        格式化的代码片段字符串 (含行号), 或错误信息
    """
    if not file_path or not os.path.exists(file_path):
        return f"[无法读取] 文件不存在: {file_path}"

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except Exception as e:
        return f"[无法读取] {file_path}: {e}"

    total = len(all_lines)
    start = max(1, line_num - context_lines)
    end = min(total, line_num + context_lines)

    result_parts = []
    result_parts.append(f"=== {file_path} (共 {total} 行, 显示 L{start}-L{end}) ===")
    for i in range(start, end + 1):
        marker = ">>>" if i == line_num else "   "
        line_content = all_lines[i - 1].rstrip("\n\r")
        result_parts.append(f"{marker} L{i:4d}: {line_content}")

    return "\n".join(result_parts)


@function_tool
def diagnose_python_error(traceback_text: str) -> str:
    """诊断 Python traceback 错误, 读取代码上下文并生成修复建议。

    当子Agent运行抛出 Python 异常 (TypeError, SyntaxError, ImportError 等)
    而非 Amesim 建模错误时, 调用此工具。它会:
    1. 解析 traceback 提取文件路径和行号
    2. 读取错误行的代码上下文
    3. 匹配已知 bug 模式
    4. 生成结构化修复建议

    Args:
        traceback_text: Python traceback 完整文本

    Returns:
        JSON 格式的诊断结果, 含 fix_suggestion, code_context, affected_files
    """
    parsed = parse_python_traceback(traceback_text)

    # 读取代码上下文
    code_context = ""
    if parsed["error_file"] and parsed["error_line"]:
        code_context = read_code_context(
            parsed["error_file"], parsed["error_line"], context_lines=12
        )

    # 根据匹配的 bug 类型生成修复建议
    bug_type = parsed["matched_bug_type"] or ""
    bug_desc = parsed["matched_bug_description"] or ""

    if bug_type == "FUNCTIONTOOL_DIRECT_CALL":
        fix_strategy = (
            "该错误表示一个被 @function_tool 装饰的函数被当作普通函数直接调用了。"
            "FunctionTool 是一个 dataclass 对象, 不能直接 () 调用。\n"
            "修复步骤:\n"
            "1. 在工具文件中, 将函数体抽离为 _xxx_impl 纯函数 (不加装饰器)\n"
            "2. @function_tool 包装只做一行代理: return _xxx_impl(args)\n"
            "3. 在调用方, 改为 import 并调用 _xxx_impl 纯函数"
        )
    elif bug_type == "DATACLASS_FIELD_ORDER":
        fix_strategy = (
            "该错误表示 dataclass 中字段顺序不正确: 无默认值的字段必须排在"
            "有默认值的字段前面。\n"
            "修复步骤:\n"
            "1. 找到报错的 @dataclass 类定义\n"
            "2. 给报错字段添加默认值 (如 = field(default_factory=list)), 或\n"
            "3. 将该字段移到所有有默认值的字段之前"
        )
    else:
        fix_strategy = (
            f"未匹配到已知 bug 模式。错误类型: {parsed['error_type']}。"
            f"请根据 traceback 和代码上下文手动分析。"
        )

    return json.dumps({
        "error_type": parsed["error_type"],
        "error_message": parsed["error_message"],
        "error_file": parsed["error_file"],
        "error_line": parsed["error_line"],
        "error_function": parsed["error_function"],
        "stack_frames": parsed["stack_frames"][-5:],  # 最近 5 帧
        "matched_bug_type": bug_type,
        "matched_bug_description": bug_desc,
        "fix_strategy": fix_strategy,
        "code_context": code_context,
        "affected_files": [f["file"] for f in parsed["stack_frames"]],
        "confidence": "high" if bug_type else "medium",
    }, ensure_ascii=False, indent=2)
