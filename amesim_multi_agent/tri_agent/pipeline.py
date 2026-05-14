"""
TriAgentPipeline — 三 Agent 流水线主协调器

编排 Orchestrator → Connector(Phase 1+2) → Parameter-Runner 的完整流程。
包含验证-修复循环和人工审核关卡。

用法:
    python -m amesim_multi_agent.tri_agent.pipeline "构建一个液氮储存系统"
    python -m amesim_multi_agent.tri_agent.pipeline "需求" --skip-review --verbose
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field

from agents import Runner

from ..model_providers import create_run_config, get_multi_provider, DEEPSEEK_MODEL
from .agents.orchestrator import create_tri_orchestrator_agent
from .agents.connector import (
    create_connector_phase1_agent,
    create_connector_phase2_agent,
)
from .agents.parameter_runner import create_parameter_runner_agent
from .topology_graph import TopologyGraph, TorsionBarModel, ComponentEntry, ConnectionEntry
from .experience_memory import ExperienceMemory


# ============================================================
# 配置
# ============================================================

MAX_VALIDATION_ITERATIONS = 3     # 验证-修复循环上限
MAX_BUILD_ATTEMPTS = 3            # 构建重试上限
MAX_EXPANSION_LAYERS = 4          # 螺旋展开最大层数


# ============================================================
# 流水线状态
# ============================================================

@dataclass
class PipelineState:
    """流水线执行状态。"""
    user_request: str = ""
    model_name: str = ""

    # 各阶段产出
    topology_graph: TopologyGraph | None = None
    phase1_result: dict | None = None
    torsionbar_model: TorsionBarModel | None = None
    runner_result: dict | None = None

    # 追踪
    validation_iterations: int = 0
    build_attempts: int = 0
    total_llm_calls: int = 0
    errors: list[str] = field(default_factory=list)

    skip_review: bool = False
    verbose: bool = False
    experience_db_path: str = ""

    @property
    def is_topology_ready(self) -> bool:
        return self.topology_graph is not None and len(self.topology_graph.nodes) > 0

    @property
    def is_components_ready(self) -> bool:
        return self.torsionbar_model is not None and self.torsionbar_model.is_complete()

    def log(self, msg: str):
        if self.verbose:
            print(f"  [Pipeline] {msg}")


# ============================================================
# 流水线
# ============================================================

class TriAgentPipeline:
    """三 Agent 建模流水线。

    用法:
        pipeline = TriAgentPipeline(skip_review=False, verbose=True)
        result = await pipeline.run("构建一个液压系统...")
    """

    def __init__(
        self,
        skip_review: bool = False,
        verbose: bool = False,
        experience_db_path: str = "",
    ):
        self.state = PipelineState(
            skip_review=skip_review,
            verbose=verbose,
            experience_db_path=experience_db_path,
        )
        self.memory: ExperienceMemory | None = None
        if experience_db_path:
            self.memory = ExperienceMemory(experience_db_path)
            self.memory.initialize()

    async def run(self, user_request: str) -> PipelineState:
        """执行完整三 Agent 流水线。"""
        self.state.user_request = user_request

        # ── 阶段 1: 主管 Agent (拓扑规划) ──
        if not await self._run_orchestrator():
            return self.state
        if not self.state.skip_review:
            self._print_topology_review()
            if not self._await_approval():
                return self.state

        # ── 阶段 2: 连接 Agent Phase 1 (核心元件) ──
        if not await self._run_connector_phase1():
            return self.state

        # ── 阶段 3: 连接 Agent Phase 2 (螺旋展开) ──
        if not await self._run_connector_phase2():
            return self.state

        # ── 阶段 4: 主管验证 ──
        if not await self._validate_and_fix():
            return self.state

        # ── 阶段 5: 代码构建 (不经过 LLM) ──
        if not self._build_model_code():
            return self.state

        self._print_final_summary()
        return self.state

    # ---- Phase Handlers ----

    async def _run_orchestrator(self) -> bool:
        """执行主管 Agent。"""
        self.state.log("Starting Orchestrator Agent...")
        agent = create_tri_orchestrator_agent()
        run_config = create_run_config()

        prompt = self._build_orchestrator_prompt()
        try:
            result = await Runner.run(agent, prompt, run_config=run_config)
        except Exception as e:
            self.state.errors.append(f"Orchestrator: {e}")
            return False

        raw = result.final_output or ""
        try:
            graph_dict = self._extract_json(raw)
            self.state.topology_graph = TopologyGraph.from_dict(graph_dict)
            self.state.model_name = self.state.topology_graph.model_name
            self.state.total_llm_calls += 1
            print(f"\n[OK]TopologyGraph: {self.state.topology_graph.summary()}")
            return True
        except Exception as e:
            self.state.errors.append(f"Orchestrator parse: {e}")
            return False

    async def _run_connector_phase1(self) -> bool:
        """执行连接 Agent Phase 1。"""
        self.state.log("Starting Connector Phase 1...")
        agent = create_connector_phase1_agent()
        run_config = create_run_config()

        prompt = self._build_phase1_prompt()
        try:
            result = await Runner.run(agent, prompt, run_config=run_config)
        except Exception as e:
            self.state.errors.append(f"Connector Phase1: {e}")
            return False

        raw = result.final_output or ""
        try:
            self.state.phase1_result = self._extract_json(raw)
            self.state.total_llm_calls += 1

            # 初始化 TorsionBarModel
            core = self.state.phase1_result.get("core_components", [])
            self.state.torsionbar_model = TorsionBarModel(
                model_name=self.state.model_name,
                components=[
                    ComponentEntry(
                        icon_name=c["icon_name"],
                        alias=c["alias"],
                        position=tuple(c["position"]),
                        submodel=c.get("selected_submodel_id", ""),
                        submodel_path=c.get("submodel_path", ""),
                        layer=0,
                    )
                    for c in core
                ],
                connections=[],
            )
            print(f"[OK]Phase 1: {len(core)} core components selected")
            return True
        except Exception as e:
            self.state.errors.append(f"Connector Phase1 parse: {e}")
            return False

    async def _run_connector_phase2(self) -> bool:
        """执行连接 Agent Phase 2 (螺旋展开)。"""
        self.state.log("Starting Connector Phase 2...")
        agent = create_connector_phase2_agent()
        run_config = create_run_config()

        prompt = self._build_phase2_prompt()
        try:
            result = await Runner.run(agent, prompt, run_config=run_config)
        except Exception as e:
            self.state.errors.append(f"Connector Phase2: {e}")
            return False

        raw = result.final_output or ""
        try:
            phase2_result = self._extract_json(raw)
            self.state.total_llm_calls += 1

            # 合并到 TorsionBarModel
            for layer_data in phase2_result.get("layers_added", []):
                self.state.torsionbar_model.components.extend([
                    ComponentEntry(
                        icon_name=c["icon_name"],
                        alias=c["alias"],
                        position=tuple(c["position"]),
                        submodel=c.get("submodel_id", ""),
                        submodel_path=c.get("submodel_path", ""),
                        layer=layer_data["layer"],
                    )
                    for c in layer_data.get("components", [])
                    if not c.get("is_terminal", False)
                ])
                self.state.torsionbar_model.components.extend([
                    ComponentEntry(
                        icon_name=c["icon_name"],
                        alias=c["alias"],
                        position=tuple(c["position"]),
                        submodel=c.get("submodel_id", ""),
                        submodel_path=c.get("submodel_path", ""),
                        layer=layer_data["layer"],
                    )
                    for c in layer_data.get("terminals", [])
                ])
                self.state.torsionbar_model.connections.extend([
                    ConnectionEntry(
                        from_component=c["from_component"],
                        from_port=c["from_port"],
                        to_component=c["to_component"],
                        to_port=c["to_port"],
                        type=c.get("type", "line"),
                        line_alias=c.get("line_alias", ""),
                        waypoints=[tuple(w) for w in c.get("waypoints", [])] if c.get("waypoints") else [],
                    )
                    for c in layer_data.get("connections", [])
                ])
                self.state.torsionbar_model.bridge_components.extend(
                    layer_data.get("bridges", [])
                )

            print(f"[OK]Phase 2: {self.state.torsionbar_model.summary()}")
            return True
        except Exception as e:
            self.state.errors.append(f"Connector Phase2 parse: {e}")
            return False

    async def _validate_and_fix(self) -> bool:
        """主管验证连接结果, 必要时发回定点修复。"""
        for iteration in range(MAX_VALIDATION_ITERATIONS):
            issues = self._validate_model()
            if not issues:
                print(f"[OK]Validation passed (iteration {iteration + 1})")
                return True

            print(f"  [!] {len(issues)} validation issues, fixing...")
            fixed = await self._targeted_fix(issues)
            if not fixed:
                return False

        print(f"  [[FAIL]] Max validation iterations ({MAX_VALIDATION_ITERATIONS}) reached")
        return False

    def _validate_model(self) -> list[dict]:
        """验证 TorsionBarModel 是否符合 TopologyGraph 要求。"""
        issues = []
        model = self.state.torsionbar_model
        graph = self.state.topology_graph

        # 检查: 所有核心节点都有对应元件
        core_aliases = {c.alias for c in model.components if c.layer == 0}
        for node in graph.core_nodes:
            found = any(
                node.label.replace("_", " ").lower() in alias.lower()
                or alias.lower().startswith(node.label.lower()[:8])
                for alias in core_aliases
            )
            if not found:
                issues.append({
                    "type": "MISSING_CORE_COMPONENT",
                    "node_id": node.node_id,
                    "label": node.label,
                    "message": f"No component selected for core node {node.node_id} ({node.label})",
                })

        # 检查: 非终端、非核心的元件是否有连接
        terminal_patterns = (
            "ground", "zero", "constant", "sigconst", "sink", "source",
            "term_", "signal_", "atmosphere", "ambient", "reservoir",
        )
        for comp in model.components:
            if comp.layer == 0:
                continue  # skip core components
            alias_lower = comp.alias.lower()
            icon_lower = comp.icon_name.lower()
            if any(p in alias_lower or p in icon_lower for p in terminal_patterns):
                continue  # terminal components are expected to be one-sided
            has_conn = any(
                c.from_component == comp.alias or c.to_component == comp.alias
                for c in model.connections
            )
            if not has_conn:
                issues.append({
                    "type": "ISOLATED_COMPONENT",
                    "alias": comp.alias,
                    "icon_name": comp.icon_name,
                    "message": f"Component {comp.alias} ({comp.icon_name}) has no connections",
                })

        return issues

    async def _targeted_fix(self, issues: list[dict]) -> bool:
        """定点修复验证发现的问题。

        简单问题用代码修复 (删孤立元件); 复杂问题发回 LLM。
        """
        fixes_applied = 0
        for issue in issues:
            if issue["type"] == "ISOLATED_COMPONENT":
                alias = issue.get("alias", "")
                # 删除没有连接的非核心孤立元件
                removed = [
                    c for c in self.state.torsionbar_model.components
                    if c.alias == alias
                ]
                if removed:
                    self.state.torsionbar_model.components = [
                        c for c in self.state.torsionbar_model.components
                        if c.alias != alias
                    ]
                    self.state.log(f"FIX: removed isolated component {alias}")
                    fixes_applied += 1

            elif issue["type"] == "MISSING_CORE_COMPONENT":
                # 核心元件缺失 — 需要 LLM 重新选择, 但当前先用简单回退
                self.state.log(f"WARN: core component missing for {issue.get('label')}, will retry phase 1")
                return await self._run_connector_phase1()

        return fixes_applied > 0 or len(issues) == 0

    async def _run_parameter_runner(self) -> bool:
        """执行参数运行 Agent。"""
        self.state.log("Starting Parameter-Runner Agent...")
        agent = create_parameter_runner_agent()
        run_config = create_run_config()

        prompt = self._build_runner_prompt()
        try:
            result = await Runner.run(agent, prompt, run_config=run_config)
        except Exception as e:
            self.state.errors.append(f"Parameter-Runner: {e}")
            return False

        raw = result.final_output or ""
        try:
            self.state.runner_result = self._extract_json(raw)
            self.state.total_llm_calls += 1

            br = self.state.runner_result.get("build_result", {})
            status = br.get("status", "error")
            print(f"[OK]Parameter-Runner: build={status}")
            return status == "success"
        except Exception as e:
            self.state.errors.append(f"Parameter-Runner parse: {e}")
            return False

    async def _record_to_memory(self):
        """将成功的建模案例写入经验记忆库。"""
        if not self.memory:
            return
        try:
            case_id = self.memory.record_success({
                "model_name": self.state.model_name,
                "user_request": self.state.user_request,
                "topology_graph_json": self.state.topology_graph.to_json(),
                "torsionbar_json": self.state.torsionbar_model.to_json(),
                "component_count": len(self.state.torsionbar_model.components),
                "physical_domains": self.state.topology_graph.physical_domains,
                "total_llm_calls": self.state.total_llm_calls,
                "total_tokens": 0,
                "tags": [self.state.topology_graph.complexity],
            })
            print(f"[OK]Recorded to experience memory (case #{case_id})")
        except Exception as e:
            self.state.log(f"Memory write failed: {e}")

    def _build_model_code(self) -> bool:
        """代码驱动构建: 将 TorsionBarModel 转换为 builder spec 并调用 build_model_direct()。

        不经过 LLM — 纯代码执行。
        需要 Amesim 环境 (AMEPython.exe) 才可实际生成 .ame 文件。
        """
        model = self.state.torsionbar_model
        if not model or not model.is_complete():
            self.state.log("No complete model to build")
            return False

        # 0. KB 交叉验证: 修复 LLM 编造的 icon_name
        self._validate_icons_in_kb(model)

        # 1. 转换 components: ComponentEntry → builder spec dict
        spec_components = []
        for c in model.components:
            comp_dict = {
                "icon_name": c.icon_name,
                "alias": c.alias,
                "position": list(c.position),
            }
            if c.submodel:
                comp_dict["submodel"] = c.submodel
            # 从 submodel_path 提取 library 名 (如 "$AME/libmec/submodels" → "libmec")
            if c.submodel_path:
                parts = c.submodel_path.replace("$AME/", "").split("/")
                if parts:
                    comp_dict["library"] = parts[0]
            if c.rotations:
                comp_dict["rotations"] = c.rotations
            comp_dict["layer"] = c.layer
            spec_components.append(comp_dict)

        # 2. 转换 connections: ConnectionEntry → builder spec dict
        spec_connections = []
        for conn in model.connections:
            conn_dict = {
                "from_alias": conn.from_component,
                "from_port": conn.from_port,
                "to_alias": conn.to_component,
                "to_port": conn.to_port,
                "type": conn.type,
            }
            if conn.waypoints:
                conn_dict["waypoints"] = [list(w) for w in conn.waypoints]
            if conn.line_alias:
                conn_dict["line_alias"] = conn.line_alias
            spec_connections.append(conn_dict)

        # 3. Call build_model_direct
        print(f"\n--- Building model: {model.model_name} ---")
        print(f"  Components: {len(spec_components)} | Connections: {len(spec_connections)}")

        try:
            from ..agents.builder import build_model_direct

            result_json = build_model_direct(
                model_name=model.model_name,
                components=spec_components,
                connections=spec_connections,
                parameters=model.non_default_params or {},
                bridge_components=model.bridge_components,
                stop_time="10",
                interval="0.01",
                mode="manual",  # 仅生成脚本, 手动在 Amesim 环境运行
            )
            result = json.loads(result_json)
            self.state.runner_result = result

            status = result.get("status", "error")
            print(f"  Build result: {status}")

            if status in ("success", "completed", "build_only"):
                ame_path = result.get("ame_path", "")
                script_path = result.get("script_path", "")
                if ame_path:
                    print(f"  .ame file: {ame_path}")
                if script_path:
                    print(f"  Script: {script_path}")

                # Write to experience memory
                if self.memory:
                    try:
                        self.memory.record_success({
                            "model_name": model.model_name,
                            "user_request": self.state.user_request,
                            "topology_graph_json": self.state.topology_graph.to_json(),
                            "torsionbar_json": model.to_json(),
                            "build_script_path": script_path,
                            "ame_file_path": ame_path,
                            "component_count": len(model.components),
                            "physical_domains": self.state.topology_graph.physical_domains,
                            "total_llm_calls": self.state.total_llm_calls,
                            "tags": [self.state.topology_graph.complexity],
                        })
                    except Exception as e:
                        self.state.log(f"Memory write: {e}")

                return True

            else:
                # Build error
                stderr = result.get("stderr", "")
                message = result.get("message", "")
                self.state.errors.append(f"Build failed: {message}")
                if stderr:
                    print(f"  Stderr: {stderr[:500]}")
                return False

        except ImportError as e:
            print(f"  [WARN] Cannot import build_model_direct: {e}")
            print(f"  Run under Amesim Python environment to build .ame files.")
            print(f"  The model definition (TorsionBar JSON) is complete and ready to build.")
            # Save TorsionBarModel as JSON for later manual build
            save_path = os.path.join(
                os.path.dirname(__file__), "..", "..",
                f"_torsionbar_{model.model_name}.json"
            )
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(model.to_json())
            print(f"  Model saved to: {save_path}")
            self.state.runner_result = {
                "status": "build_only",
                "model_name": model.model_name,
                "message": f"TorsionBar JSON saved to {save_path}",
            }
            return True  # Model spec is valid, just can't build here

        except Exception as e:
            self.state.errors.append(f"Build exception: {e}")
            return False

    def _validate_icons_in_kb(self, model: TorsionBarModel) -> None:
        """交叉验证: 检查并修复 LLM 编造的 icon_name。

        加载 knowledge_base/_registry.json, 对每个 component 的 icon_name
        做精确匹配和模糊回退, 替换编造的名为真实 icons。
        """
        self.state.log("Running KB icon validation...")
        try:
            reg_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "..",
                "knowledge_base", "json", "components", "_registry.json"
            )
            if not os.path.exists(reg_path):
                self.state.log("KB registry not found, skipping icon validation")
                return

            with open(reg_path, encoding="utf-8") as f:
                reg = json.load(f)

            icon_index = reg.get("icon_index", {})
            if not icon_index:
                return

            # 硬编码映射: LLM 常见编造名 → 真实 Amesim icon
            HARD_MAP = {
                "spring": "spring01", "damper": "damper01", "damper2": "damper02",
                "mass": "mass2port", "mass_body": "mass2port",
                "mass_friction_endstops": "mass2port", "mass_with_friction": "mass2port",
                "ground": "zeroforcesource", "mecground": "zeroforcesource",
                "mecground0": "zeroforcesource", "mechanical_ground": "zeroforcesource",
                "fixed_ground": "zeroforcesource", "wall": "zeroforcesource",
                "force_source": "forcesource", "force_signal": "sigstep",
                "force_input": "sigstep", "mecforce": "forcesource",
                "mecforcesource": "forcesource", "signal_force": "sigforcecon",
                "signalforce": "sigforcecon", "speed_source": "omegacon",
                "velocity_source": "omegacon", "displacement_sensor": "displacementsensor",
                "velocity_sensor": "velocitysensor", "acceleration_sensor": "accelerationsensor",
                "position_sensor": "displacementsensor", "angle_sensor": "anglesensor",
                "sensor": "displacementsensor",
                "fousq": "forcesource", "fousrce": "forcesource",
            }

            fixes = 0
            for comp in model.components:
                icon = comp.icon_name

                # 0: 大小写规范化 (Amesim icons 全小写)
                icon_lower = icon.lower()
                if icon != icon_lower and icon_lower in icon_index:
                    comp.icon_name = icon_lower
                    fixes += 1
                    self.state.log(f"KB fix (case): {icon} -> {icon_lower}")
                    continue

                # 精确匹配
                if icon in icon_index:
                    continue

                candidates = []

                # 1: 硬编码映射
                if icon in HARD_MAP:
                    candidates.append(HARD_MAP[icon])

                # 2: 通用前缀清理
                if not candidates:
                    for prefix in ("mec", "mec_", "tpf", "tpf_", "sig", "sig_",
                                   "hydr", "hydr_", "pn", "pn_", "ele", "ele_", "th", "th_"):
                        if icon.startswith(prefix):
                            cleaned = icon[len(prefix):]
                            cleaned = cleaned.lstrip("_").rstrip("0123456789_")
                            if cleaned in icon_index:
                                candidates.append(cleaned)
                            elif cleaned in HARD_MAP:
                                candidates.append(HARD_MAP[cleaned])
                            if candidates:
                                break

                # 3: 后缀匹配
                if not candidates:
                    for suffix in ("01", "00", "02", "1", "2", "0"):
                        trial = f"{icon}{suffix}"
                        if trial in icon_index:
                            candidates.append(trial)

                # 4: 子串匹配
                if not candidates and len(icon) >= 4:
                    icon_lower = icon.lower().replace("_", " ")
                    for real_icon in icon_index.keys():
                        real_lower = real_icon.lower().replace("_", " ")
                        if icon_lower in real_lower:
                            candidates.append(real_icon)
                            break

                # 5: 最后手段 — 尝试任何 icon 名称中包含相似字母的
                if not candidates and len(icon) >= 3:
                    first3 = icon[:3].lower()
                    for real_icon in icon_index.keys():
                        if real_icon.lower().startswith(first3):
                            candidates.append(real_icon)
                            break

                if candidates:
                    matched = candidates[0]
                    comp.icon_name = matched
                    r = reg["icon_index"].get(matched)
                    if isinstance(r, list) and r:
                        entry = r[0]
                        comp.submodel_path = f"$AME/{entry.get('lib', 'unknown')}/submodels"
                    fixes += 1
                    self.state.log(f"KB fix: {icon} -> {matched}")
                else:
                    self.state.log(f"KB WARN: no match for icon '{icon}'")

            # 同时清除 LLM 编造的 submodel IDs — 验证是否在 KB submodel_index 中
            submodel_index = reg.get("submodel_index", {})
            submodel_fixes = 0
            for comp in model.components:
                if comp.submodel:
                    if comp.submodel not in submodel_index:
                        comp.submodel = ""
                        submodel_fixes += 1

            if submodel_fixes:
                self.state.log(f"Cleared {submodel_fixes} invented submodel IDs (auto-assign)")
                fixes += submodel_fixes

            if fixes:
                print(f"  [KB] Fixed {fixes} invented icon/submodel names")
            else:
                self.state.log("KB validation: all icons OK or no matches found")

        except Exception as e:
            self.state.log(f"KB validation error: {e}")
            import traceback
            traceback.print_exc()

    def _build_orchestrator_prompt(self) -> str:
        return f"""Analyze this modeling request and generate a TopologyGraph.

USER REQUEST:
{self.state.user_request}

{"SKIP_REVIEW: Make reasonable assumptions. Do NOT ask clarifying questions." if self.state.skip_review else "Ask up to 3 questions if critical information is missing."}

Output ONLY the TopologyGraph JSON."""

    def _build_phase1_prompt(self) -> str:
        icon_hints = self._get_icon_hints_for_domains()
        return f"""Phase 1: Select core Amesim components.

TOPOLOGY GRAPH:
{self.state.topology_graph.to_json()}

{icon_hints}

For each is_core=true node, select an Amesim component from the available icons above.
Use ONLY the exact icon names listed. Do NOT invent new icon names.
Use 4-dimension verification (ports, parameters, description, topology fit).
Output ONLY the Phase 1 result JSON."""

    def _build_phase2_prompt(self) -> str:
        return f"""Phase 2: Spiral expansion.

TOPOLOGY GRAPH:
{self.state.topology_graph.to_json()}

PHASE 1 CORE COMPONENTS:
{json.dumps(self.state.phase1_result, ensure_ascii=False, indent=2) if self.state.phase1_result else '{}'}

Detect dangling ports and expand outward. Max {MAX_EXPANSION_LAYERS} layers.
Output ONLY the Phase 2 result JSON."""

    def _build_runner_prompt(self) -> str:
        return f"""Process this model: assign submodels, set parameters, build, and run.

MODEL:
{self.state.torsionbar_model.to_json() if self.state.torsionbar_model else '{}'}

Output ONLY the Parameter-Runner result JSON."""

    def _get_icon_hints_for_domains(self) -> str:
        """从 KB 中提取相关域的常用 icon 列表, 嵌入 prompt 防止 LLM 编造。"""
        try:
            reg_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "..",
                "knowledge_base", "json", "components", "_registry.json"
            )
            if not os.path.exists(reg_path):
                return ""

            with open(reg_path, encoding="utf-8") as f:
                reg = json.load(f)
            icon_index = reg.get("icon_index", {})

            # 获取当前拓扑涉及的域对应的库
            domains = self.state.topology_graph.physical_domains if self.state.topology_graph else []
            domain_libs = {
                "mechanical_1d": "libmec", "two_phase_flow": "libtpf",
                "thermal": "libth", "hydraulic": "libhydr",
                "pneumatic": "libpn", "signal": "libsig", "electric": "libeb",
            }
            target_libs = set()
            for d in domains:
                lib = domain_libs.get(d, "")
                if lib:
                    target_libs.add(lib)

            if not target_libs:
                return ""

            # 为每个目标库采集常用 icons (最多 30 个/库)
            lines = ["## AVAILABLE AMESIM ICONS (use EXACT names only)"]
            for lib_name in sorted(target_libs):
                icons_in_lib = []
                for icon_name, entries in icon_index.items():
                    if isinstance(entries, list):
                        for entry in entries:
                            if entry.get("lib") == lib_name:
                                label = entry.get("label", "")[:60]
                                icons_in_lib.append((icon_name, label))
                                break

                # 按名称排序, 取前 30
                icons_in_lib.sort(key=lambda x: x[0])
                icons_in_lib = icons_in_lib[:30]

                lines.append(f"\n### {lib_name} ({len(icons_in_lib)} icons shown)")
                for icon_name, label in icons_in_lib:
                    lines.append(f"  - `{icon_name}`: {label}")

            return "\n".join(lines)

        except Exception:
            return ""

    @staticmethod
    def _extract_json(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        import re
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            raw = match.group(0)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # 尝试修复常见 JSON 错误
                repaired = TriAgentPipeline._repair_json(raw)
                if repaired:
                    return json.loads(repaired)
        raise ValueError(f"Cannot extract JSON from output (first 200 chars: {text[:200]})")

    @staticmethod
    def _repair_json(text: str) -> str | None:
        """修复 LLM 输出中常见的 JSON 格式错误。"""
        import re
        fixed = text
        # 1. 单引号替换为双引号 (保守: 仅替换键值对模式)
        fixed = re.sub(r"'([^']+)':", r'"\1":', fixed)
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
        # 2. 移除尾部多余逗号 (JSON 不允许)
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        # 3. 修复未闭合的引号
        try:
            json.loads(fixed)
            return fixed
        except json.JSONDecodeError:
            pass
        return None

    # ---- UI Methods ----

    def _print_topology_review(self):
        graph = self.state.topology_graph
        print(f"\n{'─'*60}")
        print(f"  TOPOLOGY REVIEW: {graph.model_name}")
        print(f"{'─'*60}")
        print(graph.to_mermaid())
        print(f"{'─'*60}")
        print(f"  [A]pprove  [E]dit  [Q]uit")
        print(f"{'─'*60}")

    @staticmethod
    def _await_approval() -> bool:
        try:
            choice = input("  > ").strip().lower()
            if choice == "a":
                return True
            elif choice == "e":
                feedback = input("  Edit feedback: ").strip()
                print(f"  Feedback recorded: {feedback}")
                return True  # 简化: 接受编辑但继续
            else:
                return False
        except (EOFError, KeyboardInterrupt):
            return False

    def _print_final_summary(self):
        print(f"\n{'='*70}")
        print(f"  PIPELINE COMPLETE")
        print(f"  Model: {self.state.model_name}")
        print(f"  Topology nodes: {len(self.state.topology_graph.nodes)}")
        print(f"  Components: {len(self.state.torsionbar_model.components)}")
        print(f"  Connections: {len(self.state.torsionbar_model.connections)}")
        print(f"  LLM calls: {self.state.total_llm_calls}")
        if self.state.errors:
            print(f"  Errors: {len(self.state.errors)}")
            for e in self.state.errors:
                print(f"    - {e}")
        print(f"{'='*70}\n")


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tri-Agent Pipeline")
    parser.add_argument("request", nargs="?", help="Natural language modeling request")
    parser.add_argument("--file", "-f", help="Read request from file")
    parser.add_argument("--skip-review", action="store_true", help="Skip human review")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--experience-db", default="", help="Experience memory DB path")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            request = f.read().strip()
    elif args.request:
        request = args.request
    else:
        parser.print_help()
        sys.exit(1)

    pipeline = TriAgentPipeline(
        skip_review=args.skip_review,
        verbose=args.verbose,
        experience_db_path=args.experience_db,
    )

    state = asyncio.run(pipeline.run(request))

    status = state.runner_result.get("status", "") if state.runner_result else ""
    build_status = state.runner_result.get("build_result", {}).get("status", "") if state.runner_result else ""
    if status in ("build_only", "success") or build_status in ("success", "build_only"):
        print("[OK]Pipeline SUCCESS")
        sys.exit(0)
    else:
        print("[FAIL] Pipeline FAILED")
        if state.errors:
            for e in state.errors:
                print(f"  - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
