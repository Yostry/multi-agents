"""
Agent间数据协议 (Shared Data Models)

定义6个Agent之间传递的结构化数据类。
所有数据类均为纯Python dataclass，支持JSON序列化/反序列化，
可在程序化编排中直接传递，也可序列化后传给LLM作为上下文。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# 1. TopologyPlan — Orchestrator 输出
# ============================================================
@dataclass
class ComponentRequirement:
    """单个元件的需求描述 (Orchestrator对Selector的指令)"""
    index: int                             # 需求编号
    functional_description: str            # 功能描述 (如 "质量块，有摩擦和限位")
    physical_domain: str                   # 物理域 (如 "mechanical_1d")
    suggested_library: Optional[str] = None  # 建议的库 (如 "libmec")
    quantity: int = 1                      # 需要的数量
    keywords: list[str] = field(default_factory=list)  # 搜索关键词
    # 拓扑角色: source, sink, storage, transfer, sensor, actuator
    topological_role: str = ""


@dataclass
class TopologyPlan:
    """Orchestrator的需求分解和物理拓扑规划结果"""
    user_request: str                      # 原始用户需求
    system_description: str                # 系统整体描述
    physical_domains: list[str]            # 涉及的物理域列表
    component_requirements: list[ComponentRequirement]  # 元件需求清单
    topology_description: str              # 拓扑关系文字描述 (供Bridge Agent理解)
    physical_flow_diagram: str = ""        # 物理流程图文本 (供人工审核可视化)
    model_name: str = ""                   # ASCII安全的英文模型名 (Orchestrator LLM生成)
    notes: str = ""                        # 额外注意事项

    def to_json(self) -> str:
        """序列化为JSON字符串"""
        def _convert(obj):
            if isinstance(obj, TopologyPlan):
                return {
                    "user_request": obj.user_request,
                    "system_description": obj.system_description,
                    "physical_domains": obj.physical_domains,
                    "component_requirements": [_convert(c) for c in obj.component_requirements],
                    "model_name": obj.model_name,
                    "topology_description": obj.topology_description,
                    "physical_flow_diagram": obj.physical_flow_diagram,
                    "notes": obj.notes,
                }
            if isinstance(obj, ComponentRequirement):
                return {
                    "index": obj.index,
                    "functional_description": obj.functional_description,
                    "physical_domain": obj.physical_domain,
                    "suggested_library": obj.suggested_library,
                    "quantity": obj.quantity,
                    "keywords": obj.keywords,
                    "topological_role": obj.topological_role,
                }
            return obj
        return json.dumps(_convert(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "TopologyPlan":
        """从JSON字符串反序列化"""
        d = json.loads(s)
        reqs = [
            ComponentRequirement(
                index=r["index"],
                functional_description=r["functional_description"],
                physical_domain=r["physical_domain"],
                suggested_library=r.get("suggested_library"),
                quantity=r.get("quantity", 1),
                keywords=r.get("keywords", []),
                topological_role=r.get("topological_role", ""),
            )
            for r in d["component_requirements"]
        ]
        return cls(
            user_request=d["user_request"],
            system_description=d["system_description"],
            physical_domains=d["physical_domains"],
            component_requirements=reqs,
            topology_description=d["topology_description"],
            model_name=d.get("model_name", ""),
            physical_flow_diagram=d.get("physical_flow_diagram", ""),
            notes=d.get("notes", ""),
        )


# ============================================================
# 2. ComponentSelection — Selector 输出
# ============================================================
@dataclass
class SelectedComponent:
    """Selector选定的单个元件"""
    requirement_index: int                 # 对应 ComponentRequirement.index
    icon_key: str                          # "library:icon_name" 格式
    icon_name: str                         # 图标名称
    library: str                           # 所属库
    alias: str                             # 元件别名 (如 "Mass", "Spring")
    position: tuple[int, int]              # 画布坐标 (x, y)
    recommended_submodel: str              # 推荐子模型ID
    submodel_ids: list[str]                # 所有可用子模型ID列表
    domain: str = ""                       # 物理域中文名
    submodel_path: str = ""                # 子模型路径 (如 "$AME/libmec/submodels")
    rotations: int = 0                     # 旋转次数
    flip: bool = False                     # 是否翻转
    status: str = "found"                  # found | not_found
    # NEW FIELDS for enhanced Selector architecture
    selected_submodel_id: Optional[str] = None  # 用户最终选择的子模型ID
    confidence: Optional[float] = None     # 选型置信度 (0.0-1.0)
    selection_method: str = "search"       # 选型方式: "search" | "deep_reasoning" | "secondary_port_match"


@dataclass
class SelectionFailure:
    """未找到元件的失败记录"""
    requirement_index: int
    functional_description: str
    keywords_used: list[str]
    reason: str                            # 失败原因


@dataclass
class ComponentSelection:
    """Selector的完整选型结果"""
    selected: list[SelectedComponent]      # 成功选定的元件
    failures: list[SelectionFailure]       # 未找到的元件
    has_failures: bool = False             # 是否存在未找到的元件

    def to_json(self) -> str:
        def _convert(obj):
            if isinstance(obj, ComponentSelection):
                return {
                    "selected": [_convert(s) for s in obj.selected],
                    "failures": [_convert(f) for f in obj.failures],
                    "has_failures": obj.has_failures,
                }
            if isinstance(obj, SelectedComponent):
                return {
                    "requirement_index": obj.requirement_index,
                    "icon_key": obj.icon_key,
                    "icon_name": obj.icon_name,
                    "library": obj.library,
                    "alias": obj.alias,
                    "position": list(obj.position),
                    "recommended_submodel": obj.recommended_submodel,
                    "submodel_ids": obj.submodel_ids,
                    "domain": obj.domain,
                    "submodel_path": obj.submodel_path,
                    "rotations": obj.rotations,
                    "flip": obj.flip,
                    "status": obj.status,
                }
            if isinstance(obj, SelectionFailure):
                return {
                    "requirement_index": obj.requirement_index,
                    "functional_description": obj.functional_description,
                    "keywords_used": obj.keywords_used,
                    "reason": obj.reason,
                }
            return obj
        return json.dumps(_convert(self), ensure_ascii=False, indent=2)


# ============================================================
# NEW: Enhanced Selector Architecture Data Models
# ============================================================
@dataclass
class RefinedRequirement:
    """需求深化 Agent 输出"""
    requirement_index: int
    original_description: str
    refined_constraints: dict  # e.g. {"flow_rate_range": "10-100 L/min"}
    questions_asked: list[str] = field(default_factory=list)
    user_answers: list[str] = field(default_factory=list)


@dataclass
class ApprovalRequest:
    """审批节点 — 请求用户确认"""
    requirement_index: int
    recommended: dict  # {icon_key, submodel, confidence, reasoning}
    question: str  # e.g. "请选择: [A] 确认推荐 [B] 选备选 [C] 重新搜索"
    alternatives: list[dict] = field(default_factory=list)  # 默认放最后

    def to_json(self) -> str:
        """序列化为JSON字符串"""
        return json.dumps({
            "requirement_index": self.requirement_index,
            "recommended": self.recommended,
            "question": self.question,
            "alternatives": self.alternatives,
        }, ensure_ascii=False, indent=2)


@dataclass
class ApprovalResponse:
    """审批节点 — 用户响应"""
    requirement_index: int
    selected: Optional[dict] = None  # {icon_key, submodel} | None
    action: str = ""  # "approved" | "alternative" | "reject" | "manual"

    def to_json(self) -> str:
        """序列化为JSON字符串"""
        return json.dumps({
            "requirement_index": self.requirement_index,
            "selected": self.selected,
            "action": self.action,
        }, ensure_ascii=False, indent=2)


@dataclass
class SubmodelCompat:
    """子模型兼容性数据"""
    icon_key: str
    valid_submodels: list[str] = field(default_factory=list)


@dataclass
class PortPhysics:
    """端口物理意义"""
    index: int
    physical_quantity: str  # flow_rate, pressure, torque, etc.
    direction: str  # input, output, bidirectional
    domain: str = ""  # hydraulic, mechanical, thermal, etc.


@dataclass
class GeneratedSkill:
    """自动生成的 Skill"""
    model_name: str
    created_at: str  # ISO format datetime
    component_selections: list[dict] = field(default_factory=list)
    topology_description: str = ""
    reasoning_chain: list[str] = field(default_factory=list)
    skill_file_path: str = ""


# ============================================================
# 3. ConnectionPlan — Bridge 输出
# ============================================================
@dataclass
class PortConnection:
    """单个端口连接"""
    from_alias: str                        # 源元件别名
    from_port: int                         # 源端口号 (0-indexed)
    to_alias: str                          # 目标元件别名
    to_port: int                           # 目标端口号 (0-indexed)
    connection_type: str = "line"          # line | direct
    line_alias: str = ""                   # 折线别名
    waypoints: list[tuple[int, int]] = field(default_factory=list)
    line_submodel: Optional[str] = None    # 折线子模型
    line_submodel_path: Optional[str] = None
    port_type_from: str = ""               # 源端口物理类型
    port_type_to: str = ""                 # 目标端口物理类型


@dataclass
class BridgeComponent:
    """跨域桥接元件"""
    icon_key: str                          # "library:icon_name"
    icon_name: str
    library: str
    alias: str                             # 自动生成的别名 (如 "bridge_1")
    position: tuple[int, int]
    submodel: str
    submodel_path: str
    reason: str                            # 桥接原因 (如 "连接Thermal域与TPF域")


@dataclass
class ConnectionPlan:
    """Bridge的完整连接规划结果"""
    connections: list[PortConnection]      # 端口连接列表
    bridge_components: list[BridgeComponent]  # 需要插入的桥接元件
    terminal_components: list[BridgeComponent] = field(default_factory=list)  # 悬空端口补全的终端/边界元件
    validation_issues: list[dict] = field(default_factory=list)  # 验证过程发现的问题
    is_valid: bool = True                  # 是否通过所有验证

    def to_json(self) -> str:
        result = {
            "connections": [
                {
                    "from_alias": c.from_alias,
                    "from_port": c.from_port,
                    "to_alias": c.to_alias,
                    "to_port": c.to_port,
                    "connection_type": c.connection_type,
                    "line_alias": c.line_alias,
                    "waypoints": [list(w) for w in c.waypoints],
                    "line_submodel": c.line_submodel,
                    "line_submodel_path": c.line_submodel_path,
                    "port_type_from": c.port_type_from,
                    "port_type_to": c.port_type_to,
                }
                for c in self.connections
            ],
            "bridge_components": [
                {
                    "icon_key": b.icon_key,
                    "icon_name": b.icon_name,
                    "library": b.library,
                    "alias": b.alias,
                    "position": list(b.position),
                    "submodel": b.submodel,
                    "submodel_path": b.submodel_path,
                    "reason": b.reason,
                }
                for b in self.bridge_components
            ],
            "terminal_components": [
                {
                    "icon_key": t.icon_key,
                    "icon_name": t.icon_name,
                    "library": t.library,
                    "alias": t.alias,
                    "position": list(t.position),
                    "submodel": t.submodel,
                    "submodel_path": t.submodel_path,
                    "reason": t.reason,
                }
                for t in self.terminal_components
            ],
            "validation_issues": self.validation_issues,
            "is_valid": self.is_valid,
        }
        return json.dumps(result, ensure_ascii=False, indent=2)


# ============================================================
# 4. ParameterAssignment — Parameter Agent 输出
# ============================================================
@dataclass
class ComponentParams:
    """单个元件的参数设置"""
    alias: str                             # 元件别名
    icon_name: str                         # 图标名
    submodel_id: str                       # 子模型ID
    params: dict[str, str] = field(default_factory=dict)  # {varname: value}
    # 每个参数的详细信息 (供诊断使用)
    params_detail: list[dict] = field(default_factory=list)


@dataclass
class ParameterAssignment:
    """Parameter Agent的完整参数设置结果"""
    assignments: list[ComponentParams]     # 每个元件的参数
    warnings: list[str] = field(default_factory=list)  # 参数设置过程中的警告

    def to_dict(self) -> dict[str, str]:
        """转换为 {param_name@alias: value} 格式, 供Builder直接使用"""
        result = {}
        for comp in self.assignments:
            for varname, value in comp.params.items():
                result[f"{varname}@{comp.alias}"] = value
        return result

    def to_json(self) -> str:
        result = {
            "assignments": [
                {
                    "alias": a.alias,
                    "icon_name": a.icon_name,
                    "submodel_id": a.submodel_id,
                    "params": a.params,
                    "params_detail": a.params_detail,
                }
                for a in self.assignments
            ],
            "warnings": self.warnings,
        }
        return json.dumps(result, ensure_ascii=False, indent=2)


# ============================================================
# 5. BuildResult — Builder 输出
# ============================================================
@dataclass
class BuildResult:
    """Builder的模型构建结果"""
    status: str                            # success | error | build_only
    model_name: str                        # 模型名称
    ame_path: str = ""                     # .ame 文件路径
    script_path: str = ""                  # 构建脚本路径
    stdout: str = ""                       # 标准输出 (终端信息)
    stderr: str = ""                       # 标准错误 (错误信息)
    returncode: int = 0                    # 进程返回码
    mode: str = "auto"                     # auto | manual
    message: str = ""                      # 摘要信息

    def to_json(self) -> str:
        return json.dumps({
            "status": self.status,
            "model_name": self.model_name,
            "ame_path": self.ame_path,
            "script_path": self.script_path,
            "stdout": self.stdout[-3000:] if len(self.stdout) > 3000 else self.stdout,
            "stderr": self.stderr[-2000:] if len(self.stderr) > 2000 else self.stderr,
            "returncode": self.returncode,
            "mode": self.mode,
            "message": self.message,
        }, ensure_ascii=False, indent=2)


# ============================================================
# 6. DiagnosisReport — Diagnosis Agent 输出
# ============================================================
@dataclass
class ErrorItem:
    """单个错误/警告条目"""
    level: str                             # error | warning
    code: str                              # 错误代码 (如 ICON_NOT_FOUND)
    message: str                           # 错误描述
    target_agent: str                      # 需要修正的Agent (selector|bridge|parameter|builder|orchestrator)
    suggested_fix: str = ""                # 修复建议
    affected_components: list[str] = field(default_factory=list)  # 受影响的元件


@dataclass
class DiagnosisReport:
    """Diagnosis Agent的完整诊断报告"""
    errors: list[ErrorItem]               # 必须处理的错误
    warnings: list[ErrorItem]             # 可选处理的警告
    has_errors: bool = False              # 是否存在需处理的错误
    summary: str = ""                     # 总结

    def get_retry_targets(self) -> list[tuple[str, list[ErrorItem]]]:
        """按目标Agent分组错误, 返回 [(agent_name, [errors]), ...]"""
        groups: dict[str, list[ErrorItem]] = {}
        for e in self.errors:
            groups.setdefault(e.target_agent, []).append(e)
        return list(groups.items())

    def to_json(self) -> str:
        return json.dumps({
            "errors": [
                {
                    "level": e.level,
                    "code": e.code,
                    "message": e.message,
                    "target_agent": e.target_agent,
                    "suggested_fix": e.suggested_fix,
                    "affected_components": e.affected_components,
                }
                for e in self.errors
            ],
            "warnings": [
                {
                    "level": w.level,
                    "code": w.code,
                    "message": w.message,
                    "target_agent": w.target_agent,
                    "suggested_fix": w.suggested_fix,
                    "affected_components": w.affected_components,
                }
                for w in self.warnings
            ],
            "has_errors": self.has_errors,
            "summary": self.summary,
        }, ensure_ascii=False, indent=2)


# ============================================================
# 组合数据: 完整构建规格 (供Builder使用)
# ============================================================
# ============================================================
# 7. PipelineIntermediateData — 进程中传递的中间数据
# ============================================================
@dataclass
class ComponentPortDetail:
    """单个元件的完整端口详情 (供 Bridge/Parameter 使用)"""
    alias: str                             # 元件别名
    icon_key: str                          # library:icon_name
    icon_name: str = ""
    library: str = ""
    submodel_id: str = ""
    submodel_path: str = ""
    total_ports: int = 0
    ports: list[dict] = field(default_factory=list)
    # 每个 port: {port_index, port_tag, variables: [{norm, io, title, units}]}


@dataclass
class ComponentParamDef:
    """单个元件的参数定义 (供 Parameter Agent 查询后使用)"""
    alias: str                             # 元件别名
    submodel_id: str = ""
    params: list[dict] = field(default_factory=list)
    # 每个 param: {varname, title, default, min, max, units}


@dataclass
class PipelineIntermediateData:
    """进程内存中持有的完整中间数据。

    在各 Agent 阶段之间传递，避免重复从文件读取。
    所有数据同时写入文件 (PipelineContext)，此 dataclass 用于进程内快速访问。
    """
    model_name: str = ""
    # Orchestrator 产出
    topology_plan: Optional["TopologyPlan"] = None
    # Selector 产出
    selected_components: list[SelectedComponent] = field(default_factory=list)
    component_port_details: list[ComponentPortDetail] = field(default_factory=list)
    component_param_defs: list[ComponentParamDef] = field(default_factory=list)
    # NEW: Enhanced Selector architecture fields
    refined_requirements: list["RefinedRequirement"] = field(default_factory=list)
    approval_log: list["ApprovalResponse"] = field(default_factory=list)
    secondary_selections: list[SelectedComponent] = field(default_factory=list)
    # Bridge 产出
    connections: list["PortConnection"] = field(default_factory=list)
    bridge_components: list["BridgeComponent"] = field(default_factory=list)
    terminal_components: list["BridgeComponent"] = field(default_factory=list)
    # Parameter 产出
    non_default_params: dict[str, str] = field(default_factory=dict)
    params_detail: list[dict] = field(default_factory=list)
    # Builder 产出
    build_result: Optional["BuildResult"] = None
    build_script_content: str = ""
    # 元信息
    source_ame_path: str = ""
    extracted_json: Optional[dict] = None


# ============================================================
# 组合数据: 完整构建规格 (供Builder使用)
# ============================================================
@dataclass
class BuildSpecification:
    """传递给Builder的完整构建规格"""
    model_name: str
    components: list[dict]                 # 元件定义列表 (从SelectedComponent转换)
    connections: list[dict]                # 连接定义列表 (从PortConnection转换)
    bridge_components: list[dict]          # 桥接元件列表
    parameters: dict[str, str]             # {param_name@alias: value}
    stop_time: str = "10"
    interval: str = "0.01"
    mode: str = "auto"                     # auto | manual
    script_content: str = ""               # LLM 生成的构建脚本内容

    def to_json(self) -> str:
        return json.dumps({
            "model_name": self.model_name,
            "components": self.components,
            "connections": self.connections,
            "bridge_components": self.bridge_components,
            "parameters": self.parameters,
            "stop_time": self.stop_time,
            "interval": self.interval,
            "mode": self.mode,
        }, ensure_ascii=False, indent=2)
