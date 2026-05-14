"""
Self-Play 引擎

从成功案例变异生成新测试案例, 运行完整流水线, 发现弱点并提取新经验。
日常关闭, 需要优化 Agent 效果时手动开启。

用法:
    # 在所有成功案例上运行
    python -m amesim_multi_agent.tri_agent.self_play_engine --all --strategies domain_swap,scale_change --max-cases 20

    # 在指定案例上运行
    python -m amesim_multi_agent.tri_agent.self_play_engine --case-id 5 --strategy domain_swap

    # 干跑 (只生成变异, 不执行流水线)
    python -m amesim_multi_agent.tri_agent.self_play_engine --all --dry-run

    # 分析已有结果
    python -m amesim_multi_agent.tri_agent.self_play_engine --report
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field

from .experience_memory import ExperienceMemory, SuccessCase


# ============================================================
# 变异策略定义
# ============================================================

MUTATION_STRATEGIES = {
    "domain_swap": {
        "label": "Domain Swap",
        "description": "Replace one physical domain with another. E.g. hydraulic pump → pneumatic pump.",
        "examples": ["hydraulic→pneumatic", "thermal→two_phase_flow"],
    },
    "scale_change": {
        "label": "Scale Change",
        "description": "Scale physical parameters by factor. E.g. 10kg mass → 1000kg mass.",
        "examples": ["pressure: 3→30 bar", "flow: 0.1→10 kg/s"],
    },
    "topology_add": {
        "label": "Topology Add",
        "description": "Add an extra component. E.g. add a sensor between existing components.",
        "examples": ["add pressure sensor after pump", "add temperature sensor at tank outlet"],
    },
    "topology_remove": {
        "label": "Topology Remove",
        "description": "Remove a non-critical component. E.g. remove a damper.",
        "examples": ["remove bypass valve", "remove secondary sensor"],
    },
    "requirement_rewrite": {
        "label": "Requirement Rewrite",
        "description": "Paraphrase the original request with different wording.",
        "examples": ["hydraulic pump system→oil pressure supply circuit"],
    },
    "combined": {
        "label": "Combined",
        "description": "Apply 2+ strategies together.",
        "examples": ["domain_swap + scale_change"],
    },
}


# ============================================================
# 引擎
# ============================================================

@dataclass
class SelfPlayResult:
    """一次 self-play 运行的结果。"""
    case_id: int = 0
    source_case_id: int = 0
    mutation_type: str = ""
    mutated_request: str = ""
    result_status: str = "unknown"          # success / partial / failure
    metrics: dict = field(default_factory=dict)
    error_message: str = ""
    duration_seconds: float = 0.0


class SelfPlayEngine:
    """Self-play 案例变异引擎。

    用法:
        engine = SelfPlayEngine(memory)
        results = await engine.run(
            strategies=["domain_swap", "scale_change"],
            max_cases=20,
            dry_run=False,
        )
    """

    def __init__(self, memory: ExperienceMemory):
        self.memory = memory
        self.results: list[SelfPlayResult] = []
        self._pipeline = None  # 延迟导入避免循环

    async def run(
        self,
        case_ids: list[int] | None = None,
        strategies: list[str] | None = None,
        max_cases: int = 10,
        dry_run: bool = False,
    ) -> list[SelfPlayResult]:
        """执行 self-play 循环。"""
        strategies = strategies or list(MUTATION_STRATEGIES.keys())

        # 加载案例
        cases = await self._load_cases(case_ids)
        if not cases:
            print("[SelfPlay] No cases found in experience memory.")
            return []

        print(f"[SelfPlay] Loaded {len(cases)} cases")
        print(f"[SelfPlay] Strategies: {strategies}")
        print(f"[SelfPlay] Max mutations: {max_cases}")
        print(f"[SelfPlay] Dry run: {dry_run}")

        mutation_count = 0
        for case in cases:
            if mutation_count >= max_cases:
                break
            for strategy in strategies:
                if mutation_count >= max_cases:
                    break
                mutation_count += 1
                result = await self._run_single(case, strategy, dry_run)
                self.results.append(result)

                # 记录到数据库
                self.memory.record_self_play_case({
                    "source_case_id": case.id,
                    "mutation_type": strategy,
                    "mutated_request": result.mutated_request,
                    "expected_topology_json": {},
                    "actual_topology_json": {},
                    "result_status": result.result_status,
                    "metrics_json": result.metrics,
                })

        self._print_summary()
        return self.results

    async def _load_cases(self, case_ids: list[int] | None = None) -> list[SuccessCase]:
        """加载要变异的案例。"""
        if case_ids:
            cases = []
            for cid in case_ids:
                # 简化: 通过查询加载; 完整实现需 get_case_by_id
                cases.extend(self.memory.query_similar_cases(f"id:{cid}"))
            return cases
        else:
            return self.memory.query_by_domain("")  # all domains

    async def _run_single(
        self,
        case: SuccessCase,
        strategy: str,
        dry_run: bool,
    ) -> SelfPlayResult:
        """对单个案例执行单个变异策略。"""
        start = time.time()
        result = SelfPlayResult(
            source_case_id=case.id,
            mutation_type=strategy,
        )

        # 生成变异需求
        mutated_request = self._mutate_request(case.user_request, strategy)
        result.mutated_request = mutated_request

        print(f"\n  [{strategy}] {case.model_name}")
        print(f"    Original: {case.user_request[:80]}...")
        print(f"    Mutated:  {mutated_request[:80]}...")

        if dry_run:
            result.result_status = "dry_run"
            result.metrics = {"mutation_generated": True}
            return result

        # 实际运行流水线 (需要导入 pipeline)
        try:
            from .pipeline import TriAgentPipeline
            pipeline = TriAgentPipeline(skip_review=True, verbose=False)
            state = await pipeline.run(mutated_request)

            if state.runner_result:
                build_ok = state.runner_result.get("build_result", {}).get("status") == "success"
                result.result_status = "success" if build_ok else "partial"
            else:
                result.result_status = "failure"

            result.metrics = {
                "topology_nodes": len(state.topology_graph.nodes) if state.topology_graph else 0,
                "components_selected": len(state.torsionbar_model.components) if state.torsionbar_model else 0,
                "llm_calls": state.total_llm_calls,
                "errors": state.errors[:5],
            }
        except Exception as e:
            result.result_status = "failure"
            result.error_message = str(e)
            result.metrics = {"exception": str(e)}

        result.duration_seconds = time.time() - start
        return result

    def _mutate_request(self, original: str, strategy: str) -> str:
        """应用变异策略 (简化版 — 规则替换 + 随机扰动)。

        完整版应使用 LLM 进行智能变异, 此处为基础实现。
        """
        mutations = {
            "domain_swap": self._mutate_domain_swap,
            "scale_change": self._mutate_scale_change,
            "topology_add": self._mutate_topology_add,
            "topology_remove": self._mutate_topology_remove,
            "requirement_rewrite": self._mutate_requirement_rewrite,
            "combined": self._mutate_combined,
        }
        mutator = mutations.get(strategy, lambda x: x)
        return mutator(original)

    # ---- 变异实现 ----

    def _mutate_domain_swap(self, text: str) -> str:
        swaps = [
            ("hydraulic", "pneumatic"),
            ("液氮", "液氧"),
            ("LN2", "LO2"),
            ("液压", "气动"),
        ]
        result = text
        for old, new in random.sample(swaps, min(2, len(swaps))):
            result = result.replace(old, new)
        return result

    def _mutate_scale_change(self, text: str) -> str:
        """尝试修改数字参数 (简化版)。"""
        import re
        def scale(m):
            val = float(m.group(1))
            factor = random.choice([0.1, 0.5, 2.0, 5.0, 10.0])
            return f"{val * factor:.1f}"
        result = re.sub(r'(\d+\.?\d*)\s*(bar|MPa|L|kg|°C|K)', lambda m: f"{scale(m)} {m.group(2)}", text)
        return result

    def _mutate_topology_add(self, text: str) -> str:
        additions = [
            " 在系统中添加一个压力传感器用于监测。",
            " Add a temperature sensor at the outlet.",
            " Include a flow meter after the pump.",
        ]
        return text + random.choice(additions)

    def _mutate_topology_remove(self, text: str) -> str:
        return text + " (注意: 简化模型, 省略次要附件)"

    def _mutate_requirement_rewrite(self, text: str) -> str:
        """简单的同义改写 (完整版用 LLM)。"""
        prefixes = [
            "I need to model the following: ",
            "Please build a simulation model for: ",
            "Construct an Amesim model of: ",
        ]
        return random.choice(prefixes) + text

    def _mutate_combined(self, text: str) -> str:
        """组合两种变异。"""
        result = self._mutate_domain_swap(text)
        result = self._mutate_scale_change(result)
        return result

    def _print_summary(self):
        """打印运行总结。"""
        total = len(self.results)
        success = sum(1 for r in self.results if r.result_status == "success")
        partial = sum(1 for r in self.results if r.result_status == "partial")
        failure = sum(1 for r in self.results if r.result_status == "failure")

        print(f"\n{'='*60}")
        print(f"  SELF-PLAY SUMMARY")
        print(f"  Total: {total} | Success: {success} | Partial: {partial} | Failure: {failure}")
        print(f"  Success rate: {success/total*100:.0f}%" if total > 0 else "")
        print(f"{'='*60}")

        by_strategy = {}
        for r in self.results:
            by_strategy.setdefault(r.mutation_type, []).append(r.result_status)
        for strategy, statuses in by_strategy.items():
            s_ok = statuses.count("success")
            print(f"  {strategy}: {s_ok}/{len(statuses)} passed")


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Self-Play Engine")
    parser.add_argument("--all", action="store_true", help="Run on all success cases")
    parser.add_argument("--case-id", type=int, help="Run on a specific case")
    parser.add_argument("--strategies", default="domain_swap,scale_change",
                        help="Comma-separated strategy names")
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate mutations only, don't execute pipeline")
    parser.add_argument("--report", action="store_true",
                        help="Analyze existing self-play results")
    parser.add_argument("--db", default="", help="Experience DB path")
    args = parser.parse_args()

    memory = ExperienceMemory(args.db if args.db else None)
    memory.initialize()

    if args.report:
        stats = memory.stats()
        print(f"Experience DB Stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return

    if not args.all and not args.case_id:
        parser.print_help()
        return

    case_ids = [args.case_id] if args.case_id else None
    strategies = [s.strip() for s in args.strategies.split(",")]

    engine = SelfPlayEngine(memory)
    asyncio.run(engine.run(
        case_ids=case_ids,
        strategies=strategies,
        max_cases=args.max_cases,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
