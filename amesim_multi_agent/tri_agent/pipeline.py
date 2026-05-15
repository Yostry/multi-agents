"""
TriAgentPipeline — 三 Agent 流水线主协调器

编排 Orchestrator → Connector(Phase 1+2) → Parameter-Runner 的完整流程。
使用现有的 @function_tool 工具, 不重复造轮子。

用法:
    python -m amesim_multi_agent.tri_agent.pipeline "构建一个液氮储存系统"
    python -m amesim_multi_agent.tri_agent.pipeline "需求" --skip-review --verbose
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field

from agents import Runner

from ..model_providers import create_run_config
from .agents.connector import create_connector_phase1_with_tools, create_connector_phase2_with_tools
from .agents.orchestrator import create_tri_orchestrator_agent
from .agents.parameter_runner import create_parameter_runner_with_tools
from .topology_graph import TopologyGraph, TorsionBarModel, ComponentEntry, ConnectionEntry
from .experience_memory import ExperienceMemory
from .direct_builder import DirectBuilder

# ── 现有工具 (不重复造轮子) ──
from ..tools.selector_tools import (
    batch_search_and_select, search_component,
    get_component_detail, recommend_submodel,
)
from ..tools.bridge_tools import (
    batch_query_ports, verify_all_components_exist,
    detect_dangling_ports, find_port_connections,
    find_bridge_paths, find_junction_component,
    validate_connection_plan, find_dangling_terminal_candidates,
)
from ..tools.parameter_tools import batch_query_params, query_submodel_params
from ..tools.builder_tools import build_and_run_model, repair_build_script
from ..tools.diagnosis_tools import diagnose_error, classify_error, parse_build_output


MAX_VALIDATION_ITERATIONS = 3
MAX_BUILD_ATTEMPTS = 3
MAX_EXPANSION_LAYERS = 4


@dataclass
class PipelineState:
    user_request: str = ""
    model_name: str = ""
    topology_graph: TopologyGraph | None = None
    phase1_result: dict | None = None
    torsionbar_model: TorsionBarModel | None = None
    runner_result: dict | None = None
    validation_iterations: int = 0
    total_llm_calls: int = 0
    errors: list[str] = field(default_factory=list)
    skip_review: bool = False
    verbose: bool = False

    @property
    def is_topology_ready(self) -> bool:
        return self.topology_graph is not None and len(self.topology_graph.nodes) > 0

    @property
    def is_components_ready(self) -> bool:
        return self.torsionbar_model is not None and self.torsionbar_model.is_complete()

    def log(self, msg: str):
        if self.verbose:
            print(f"  [Pipeline] {msg}")


class TriAgentPipeline:

    def __init__(self, skip_review=False, verbose=False, experience_db_path=""):
        self.state = PipelineState(skip_review=skip_review, verbose=verbose)
        self.memory = None
        if experience_db_path:
            self.memory = ExperienceMemory(experience_db_path)
            self.memory.initialize()

    # ================================================================
    # 主流程
    # ================================================================

    async def run(self, user_request: str) -> PipelineState:
        self.state.user_request = user_request

        if not await self._run_orchestrator():
            return self.state
        if not self.state.skip_review:
            if not self._await_approval():
                return self.state

        # DirectBuilder: 简单模型走确定性 KB 搜索, 复杂模型走 LLM
        if self._try_direct_build():
            pass  # DirectBuilder 成功
        elif not await self._run_connector_phase1():
            return self.state
        elif not await self._run_connector_phase2():
            return self.state

        if not await self._validate_and_fix():
            return self.state

        if not self._build_model_code():
            return self.state

        self._generate_preview()

        self._print_final_summary()
        return self.state

    # ================================================================
    # Phase: Orchestrator
    # ================================================================

    async def _run_orchestrator(self) -> bool:
        self.state.log("Starting Orchestrator Agent...")
        agent = create_tri_orchestrator_agent()

        prompt = (
            f"Analyze this modeling request and generate a TopologyGraph.\n\n"
            f"USER REQUEST:\n{self.state.user_request}\n\n"
            f"{'SKIP_REVIEW: Make reasonable assumptions.' if self.state.skip_review else 'Ask up to 3 questions if critical info is missing.'}\n\n"
            f"Output ONLY the TopologyGraph JSON."
        )
        try:
            result = await Runner.run(agent, prompt, run_config=create_run_config())
        except Exception as e:
            self.state.errors.append(f"Orchestrator: {e}")
            return False

        try:
            graph_dict = self._extract_json(result.final_output or "")
            self.state.topology_graph = TopologyGraph.from_dict(graph_dict)
            self.state.model_name = self.state.topology_graph.model_name
            self.state.total_llm_calls += 1
            print(f"\n[OK]TopologyGraph: {self.state.topology_graph.summary()}")
            return True
        except Exception as e:
            self.state.errors.append(f"Orchestrator parse: {e}")
            return False

    def _try_direct_build(self) -> bool:
        """用 DirectBuilder 确定性构建 (简单模型, 无 LLM)。

        若成功, 直接设置 torsionbar_model; 若失败, 返回 False 让 LLM fallback 接管。
        """
        graph = self.state.topology_graph
        if not graph or graph.component_count > 8:
            return False

        try:
            builder = DirectBuilder()
            model = builder.build(graph)
            if model and model.is_complete():
                self.state.torsionbar_model = model
                self.state.total_llm_calls += 0  # 不计入 LLM 调用
                print(f"[OK]DirectBuilder: {model.summary()}")
                return True
        except Exception as e:
            self.state.log(f"DirectBuilder failed: {e}")

        return False

    # ================================================================
    # Phase: Connector Phase 1 (核心元件 + KB 搜索工具)
    # ================================================================

    async def _run_connector_phase1(self) -> bool:
        self.state.log("Starting Connector Phase 1...")
        agent = create_connector_phase1_with_tools(
            batch_search_and_select=batch_search_and_select,
            search_component=search_component,
            get_component_detail=get_component_detail,
            recommend_submodel=recommend_submodel,
        )
        prompt = (
            f"Phase 1: Select core Amesim components.\n\n"
            f"TOPOLOGY GRAPH:\n{self.state.topology_graph.to_json()}\n\n"
            f"For each is_core=true node, use the search tools to find the best Amesim component.\n"
            f"Use 4-dimension verification. Output ONLY the Phase 1 result JSON."
        )
        try:
            result = await Runner.run(agent, prompt, run_config=create_run_config())
        except Exception as e:
            self.state.errors.append(f"Connector Phase1: {e}")
            return False

        try:
            self.state.phase1_result = self._extract_json(result.final_output or "")
            self.state.total_llm_calls += 1

            core = self.state.phase1_result.get("core_components", [])
            self.state.torsionbar_model = TorsionBarModel(
                model_name=self.state.model_name,
                components=[
                    ComponentEntry(icon_name=c["icon_name"], alias=c["alias"],
                                   position=tuple(c["position"]),
                                   submodel=c.get("selected_submodel_id", ""),
                                   submodel_path=c.get("submodel_path", ""), layer=0)
                    for c in core
                ],
                connections=[],
            )
            print(f"[OK]Phase 1: {len(core)} core components selected")
            return True
        except Exception as e:
            self.state.errors.append(f"Connector Phase1 parse: {e}")
            return False

    # ================================================================
    # Phase: Connector Phase 2 (螺旋展开 + 端口连接工具)
    # ================================================================

    async def _run_connector_phase2(self) -> bool:
        self.state.log("Starting Connector Phase 2...")
        agent = create_connector_phase2_with_tools(
            batch_query_ports=batch_query_ports,
            verify_all_components_exist=verify_all_components_exist,
            detect_dangling_ports=detect_dangling_ports,
            find_port_connections=find_port_connections,
            find_bridge_paths=find_bridge_paths,
            find_junction_component=find_junction_component,
            validate_connection_plan=validate_connection_plan,
            find_dangling_terminal_candidates=find_dangling_terminal_candidates,
            get_component_detail=get_component_detail,
        )
        prompt = (
            f"Phase 2: Spiral expansion.\n\n"
            f"TOPOLOGY GRAPH:\n{self.state.topology_graph.to_json()}\n\n"
            f"PHASE 1 CORE COMPONENTS:\n"
            f"{json.dumps(self.state.phase1_result, ensure_ascii=False, indent=2) if self.state.phase1_result else '{}'}\n\n"
            f"Use the tools to detect dangling ports, find port connections, and expand outward.\n"
            f"Max {MAX_EXPANSION_LAYERS} layers. Output ONLY the Phase 2 result JSON."
        )
        try:
            result = await Runner.run(agent, prompt, run_config=create_run_config())
        except Exception as e:
            self.state.errors.append(f"Connector Phase2: {e}")
            return False

        try:
            phase2_result = self._extract_json(result.final_output or "")
            self.state.total_llm_calls += 1

            for layer_data in phase2_result.get("layers_added", []):
                layer = layer_data["layer"]
                for c in layer_data.get("components", []):
                    if not c.get("is_terminal", False):
                        self.state.torsionbar_model.components.append(ComponentEntry(
                            icon_name=c["icon_name"], alias=c["alias"],
                            position=tuple(c["position"]),
                            submodel=c.get("submodel_id", ""),
                            submodel_path=c.get("submodel_path", ""), layer=layer))
                for c in layer_data.get("terminals", []):
                    self.state.torsionbar_model.components.append(ComponentEntry(
                        icon_name=c["icon_name"], alias=c["alias"],
                        position=tuple(c["position"]),
                        submodel=c.get("submodel_id", ""),
                        submodel_path=c.get("submodel_path", ""), layer=layer))
                for c in layer_data.get("connections", []):
                    self.state.torsionbar_model.connections.append(ConnectionEntry(
                        from_component=c["from_component"], from_port=c["from_port"],
                        to_component=c["to_component"], to_port=c["to_port"],
                        type=c.get("type", "line"), line_alias=c.get("line_alias", ""),
                        waypoints=[tuple(w) for w in c.get("waypoints", [])] if c.get("waypoints") else []))
                self.state.torsionbar_model.bridge_components.extend(
                    layer_data.get("bridges", []))

            print(f"[OK]Phase 2: {self.state.torsionbar_model.summary()}")
            return True
        except Exception as e:
            self.state.errors.append(f"Connector Phase2 parse: {e}")
            return False

    # ================================================================
    # Phase: Validation + Fix
    # ================================================================

    async def _validate_and_fix(self) -> bool:
        for i in range(MAX_VALIDATION_ITERATIONS):
            issues = self._validate_model()
            if not issues:
                print(f"[OK]Validation passed (iteration {i + 1})")
                return True
            print(f"  [!] {len(issues)} validation issues, fixing...")
            if not self._targeted_fix(issues):
                return False
        print(f"  [[FAIL]] Max validation iterations reached")
        return False

    def _validate_model(self) -> list[dict]:
        issues = []
        model = self.state.torsionbar_model
        graph = self.state.topology_graph

        core_aliases = {c.alias for c in model.components if c.layer == 0}
        for node in graph.core_nodes:
            found = any(node.label.replace("_", " ").lower() in alias.lower()
                        or alias.lower().startswith(node.label.lower()[:8])
                        for alias in core_aliases)
            if not found:
                issues.append({"type": "MISSING_CORE_COMPONENT", "node_id": node.node_id,
                               "label": node.label,
                               "message": f"No component for core node {node.node_id} ({node.label})"})

        terminal = ("ground", "zero", "constant", "sigconst", "sink", "source",
                     "term_", "signal_", "atmosphere", "ambient", "reservoir")
        for comp in model.components:
            if comp.layer == 0:
                continue
            alias_lower = comp.alias.lower()
            if any(p in alias_lower or p in comp.icon_name.lower() for p in terminal):
                continue
            if not any(c.from_component == comp.alias or c.to_component == comp.alias
                       for c in model.connections):
                issues.append({"type": "ISOLATED_COMPONENT", "alias": comp.alias,
                               "icon_name": comp.icon_name,
                               "message": f"Component {comp.alias} ({comp.icon_name}) has no connections"})
        return issues

    def _targeted_fix(self, issues: list[dict]) -> bool:
        fixes = 0
        for issue in issues:
            if issue["type"] == "ISOLATED_COMPONENT":
                alias = issue.get("alias", "")
                self.state.torsionbar_model.components = [
                    c for c in self.state.torsionbar_model.components if c.alias != alias]
                self.state.log(f"FIX: removed isolated {alias}")
                fixes += 1
            elif issue["type"] == "MISSING_CORE_COMPONENT":
                self.state.log(f"WARN: core component missing, retrying phase 1")
                return False  # 触发外层重试
        return fixes > 0 or len(issues) == 0

    # ================================================================
    # Phase: Code Build
    # ================================================================

    def _build_model_code(self) -> bool:
        model = self.state.torsionbar_model
        if not model or not model.is_complete():
            return False

        # 清除 LLM 编造的 submodel IDs
        for comp in model.components:
            if comp.submodel:
                valid = (4 <= len(comp.submodel) <= 15 and comp.submodel[0].isupper()
                         and any(c.isdigit() for c in comp.submodel) and "." not in comp.submodel)
                if not valid:
                    comp.submodel = ""

        # 转换 components
        spec_components = []
        for c in model.components:
            d = {"icon_name": c.icon_name, "alias": c.alias, "position": list(c.position)}
            if c.submodel:
                d["submodel"] = c.submodel
            if c.submodel_path:
                parts = c.submodel_path.replace("$AME/", "").split("/")
                if parts:
                    d["library"] = parts[0]
            if c.rotations:
                d["rotations"] = c.rotations
            spec_components.append(d)

        # 转换 connections
        spec_connections = []
        for conn in model.connections:
            d = {"from_alias": conn.from_component, "from_port": conn.from_port,
                 "to_alias": conn.to_component, "to_port": conn.to_port, "type": conn.type}
            if conn.waypoints:
                d["waypoints"] = [list(w) for w in conn.waypoints]
            spec_connections.append(d)

        print(f"\n--- Building model: {model.model_name} ---")
        print(f"  Components: {len(spec_components)} | Connections: {len(spec_connections)}")

        try:
            from ..agents.builder import build_model_direct
            result = json.loads(build_model_direct(
                model_name=model.model_name, components=spec_components,
                connections=spec_connections,
                parameters=model.non_default_params or {},
                bridge_components=model.bridge_components,
                mode="manual"))
        except ImportError:
            self._save_torsionbar_fallback(model)
            return True
        except Exception as e:
            self.state.errors.append(f"Build: {e}")
            return False

        self.state.runner_result = result
        status = result.get("status", "error")
        print(f"  Build result: {status}")

        if status in ("success", "completed", "build_only"):
            if result.get("script_path"):
                print(f"  Script: {result['script_path']}")
            return True
        else:
            self.state.errors.append(f"Build failed: {result.get('message', '')}")
            return False

    def _save_torsionbar_fallback(self, model):
        save_path = os.path.join(os.path.dirname(__file__), "..", "..",
                                 f"_torsionbar_{model.model_name}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(model.to_json())
        print(f"  Model saved to: {save_path}")
        self.state.runner_result = {"status": "build_only", "model_name": model.model_name,
                                     "message": f"TorsionBar JSON saved to {save_path}"}

    # ================================================================
    # Helpers
    # ================================================================

    @staticmethod
    def _extract_json(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        import re
        m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            raw = m.group(0)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # 修复常见 JSON 错误
                raw = re.sub(r"'([^']+)':", r'"\1":', raw)
                raw = re.sub(r":\s*'([^']*)'", r': "\1"', raw)
                raw = re.sub(r',\s*}', '}', raw)
                raw = re.sub(r',\s*]', ']', raw)
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    pass
        raise ValueError(f"Cannot extract JSON (first 200 chars: {text[:200]})")

    @staticmethod
    def _await_approval() -> bool:
        try:
            return input("  [A]pprove / [Q]uit > ").strip().lower() == "a"
        except (EOFError, KeyboardInterrupt):
            return False

    def _generate_preview(self):
        """调用 AMEPreview.exe 生成模型拓扑预览图。

        需要有效的 .ame 文件。若 build mode 为 manual (仅生成脚本),
        .ame 文件尚未生成, 则跳过预览。
        """
        ame_root = os.environ.get("AME", "D:/AMESIM24/Amesim")
        preview_exe = os.path.join(ame_root, "win64", "AMEPreview.exe")

        if not os.path.exists(preview_exe):
            self.state.log("AMEPreview.exe not found, skipping preview")
            return

        # 查找 .ame 文件
        ame_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                               "ame_models", self.state.model_name)
        ame_path = os.path.join(ame_dir, f"{self.state.model_name}.ame")

        if not os.path.exists(ame_path):
            self.state.log(f"No .ame file yet, skipping preview. "
                           f"Run build script with AMEPython.exe to generate it.")
            return

        preview_path = os.path.join(ame_dir, f"{self.state.model_name}_preview.png")

        try:
            result = subprocess.run(
                [preview_exe, ame_path, preview_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and os.path.exists(preview_path):
                size_kb = os.path.getsize(preview_path) / 1024
                print(f"  Preview: {preview_path} ({size_kb:.0f} KB)")
                self.state.runner_result = self.state.runner_result or {}
                self.state.runner_result["preview_path"] = preview_path
            else:
                self.state.log(f"Preview failed: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            self.state.log("Preview generation timed out")
        except Exception as e:
            self.state.log(f"Preview error: {e}")

    def _print_final_summary(self):
        m = self.state.torsionbar_model
        g = self.state.topology_graph
        print(f"\n{'='*70}")
        print(f"  PIPELINE COMPLETE")
        print(f"  Model: {self.state.model_name}")
        print(f"  Nodes: {len(g.nodes)} | Components: {len(m.components)} | Connections: {len(m.connections)}")
        print(f"  LLM calls: {self.state.total_llm_calls}")
        if self.state.runner_result and self.state.runner_result.get("preview_path"):
            print(f"  Preview: {self.state.runner_result['preview_path']}")
        if self.state.errors:
            print(f"  Errors: {len(self.state.errors)}")
            for e in self.state.errors:
                print(f"    - {e}")
        print(f"{'='*70}\n")


# ================================================================
# CLI
# ================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="Tri-Agent Pipeline")
    p.add_argument("request", nargs="?", help="Modeling request")
    p.add_argument("--file", "-f", help="Read request from file")
    p.add_argument("--skip-review", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            req = f.read().strip()
    elif args.request:
        req = args.request
    else:
        p.print_help(); sys.exit(1)

    pipeline = TriAgentPipeline(skip_review=args.skip_review, verbose=args.verbose)
    state = asyncio.run(pipeline.run(req))

    ok = False
    if state.runner_result:
        ok = state.runner_result.get("status") in ("success", "build_only")
    if ok:
        print("[OK]Pipeline SUCCESS"); sys.exit(0)
    else:
        print("[FAIL] Pipeline FAILED")
        for e in state.errors:
            print(f"  - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
