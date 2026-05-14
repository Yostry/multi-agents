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

        # ── 阶段 5: 参数运行 Agent ──
        if not await self._run_parameter_runner():
            return self.state

        # ── 记录到经验记忆库 ──
        await self._record_to_memory()

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
            print(f"\n✓ TopologyGraph: {self.state.topology_graph.summary()}")
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
            print(f"✓ Phase 1: {len(core)} core components selected")
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

            print(f"✓ Phase 2: {self.state.torsionbar_model.summary()}")
            return True
        except Exception as e:
            self.state.errors.append(f"Connector Phase2 parse: {e}")
            return False

    async def _validate_and_fix(self) -> bool:
        """主管验证连接结果, 必要时发回定点修复。"""
        for iteration in range(MAX_VALIDATION_ITERATIONS):
            issues = self._validate_model()
            if not issues:
                print(f"✓ Validation passed (iteration {iteration + 1})")
                return True

            print(f"  [!] {len(issues)} validation issues, fixing...")
            fixed = await self._targeted_fix(issues)
            if not fixed:
                return False

        print(f"  [✗] Max validation iterations ({MAX_VALIDATION_ITERATIONS}) reached")
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

        # 检查: 每个相关元件至少有一条连接
        for comp in model.components:
            has_conn = any(
                c.from_component == comp.alias or c.to_component == comp.alias
                for c in model.connections
            )
            if not has_conn and comp.layer > 0:
                issues.append({
                    "type": "ISOLATED_COMPONENT",
                    "alias": comp.alias,
                    "message": f"Component {comp.alias} has no connections",
                })

        return issues

    async def _targeted_fix(self, issues: list[dict]) -> bool:
        """发回连接 Agent 定点修复单个问题。"""
        agent = create_connector_phase2_agent()
        run_config = create_run_config()

        prompt = f"""TARGETED FIX REQUEST: The following issues were found in your model.
Fix ONLY these issues. Do NOT rebuild the entire model.

CURRENT MODEL:
{self.state.torsionbar_model.to_json()}

ISSUES TO FIX:
{json.dumps(issues, ensure_ascii=False, indent=2)}

RULES:
- You may ADD new components or connections
- You may REMOVE non-core components (layer > 0)
- NEVER remove or modify core components (layer = 0)
- Output ONLY the fix result JSON with added/removed items."""

        try:
            result = await Runner.run(agent, prompt, run_config=run_config)
            raw = result.final_output or ""
            fix_result = self._extract_json(raw)
            # 应用修复...
            self.state.total_llm_calls += 1
            return True
        except Exception as e:
            self.state.errors.append(f"Targeted fix: {e}")
            return False

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
            print(f"✓ Parameter-Runner: build={status}")
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
            print(f"✓ Recorded to experience memory (case #{case_id})")
        except Exception as e:
            self.state.log(f"Memory write failed: {e}")

    # ---- Helper Methods ----

    def _build_orchestrator_prompt(self) -> str:
        return f"""Analyze this modeling request and generate a TopologyGraph.

USER REQUEST:
{self.state.user_request}

{"SKIP_REVIEW: Make reasonable assumptions. Do NOT ask clarifying questions." if self.state.skip_review else "Ask up to 3 questions if critical information is missing."}

Output ONLY the TopologyGraph JSON."""

    def _build_phase1_prompt(self) -> str:
        return f"""Phase 1: Select core Amesim components.

TOPOLOGY GRAPH:
{self.state.topology_graph.to_json()}

For each is_core=true node, search and select the best Amesim component.
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

    @staticmethod
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

    if state.runner_result and state.runner_result.get("build_result", {}).get("status") == "success":
        print("✓ Pipeline SUCCESS")
        sys.exit(0)
    else:
        print("✗ Pipeline FAILED")
        if state.errors:
            for e in state.errors:
                print(f"  - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
