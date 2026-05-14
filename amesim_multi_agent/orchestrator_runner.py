"""
Orchestrator Runner — 程序化编排主控制器

负责按序调用6个Agent，管理数据传递、Diagnosis→Router错误路由、重试容错。

架构:
    User Request
        ↓
    Orchestrator.plan_topology() → TopologyPlan
        ↓
    【人工审核关卡】— 显示物理流程图，等待审核确认或修改
        ├── 通过 → 继续
        └── 修改 → 回到 Orchestrator 重新规划
        ↓
    Selector.select() → ComponentSelection
        ├── 出错 → Diagnosis.diagnose() → Router.decide()
        │         ├── auto:  自动执行 retry/replan/escalate
        │         └── manual: 人工审核路由决策
        ↓
    Bridge.connect() → ConnectionPlan
        ├── 出错 → Diagnosis → Router (同上)
        ↓
    Parameter.assign() → ParameterAssignment
        ├── 出错 → Diagnosis → Router (同上)
        ↓
    Builder.build() → BuildResult
        ├── 出错 → Diagnosis → Router (同上)
        ↓
    Diagnosis.analyze() → DiagnosisReport (Builder失败时深度诊断)
        ├── errors → 路由到对应Agent重试
        └── success → DONE

用法:
    # 自动模式 (默认)
    uv run python -m amesim_multi_agent.orchestrator_runner "构建一个朗肯循环发电模型"

    # 人工审核模式 (每个错误路由决策需人工确认)
    uv run python -m amesim_multi_agent.orchestrator_runner "构建一个朗肯循环发电模型" --review-mode manual
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import os
import subprocess
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents import Runner, trace
from agents.run import RunConfig

from .agents.orchestrator import create_orchestrator_agent, create_routing_agent
from .agents.selector import create_selector_agent
from .agents.bridge import create_bridge_agent
from .agents.parameter import create_parameter_agent
from .agents.builder import create_builder_agent, build_model_direct
from .agents.diagnosis import create_diagnosis_agent
from .kimi_provider import create_kimi_run_config
from .selector_integration import run_new_selector_pipeline
from .pipeline_context import PipelineContext
from .protocols import (
    TopologyPlan,
    ComponentSelection,
    ConnectionPlan,
    ParameterAssignment,
    BuildResult,
    DiagnosisReport,
    BuildSpecification,
    SelectedComponent,
    SelectionFailure,
)

# ============================================================
# 配置
# ============================================================
MAX_RETRIES_PER_AGENT = 3  # 同一Agent连续失败阈值
MAX_TOTAL_RETRIES = 10      # 全局最大重试次数


@dataclass
class PipelineState:
    """流水线执行状态"""
    user_request: str
    topology_plan: TopologyPlan | None = None
    component_selection: ComponentSelection | None = None
    connection_plan: ConnectionPlan | None = None
    parameter_assignment: ParameterAssignment | None = None
    build_result: BuildResult | None = None
    diagnosis_report: DiagnosisReport | None = None

    # 审核模式: "auto" 自动执行 / "manual" 人工审核每个路由决策
    review_mode: str = "auto"

    # 重试计数
    agent_retry_counts: dict[str, int] = field(default_factory=dict)
    consecutive_failures: dict[str, int] = field(default_factory=dict)
    total_retries: int = 0
    current_stage: str = "init"

    # Router 决策记录 (供 retry 时传递给目标Agent)
    last_router_adjustments: str = ""
    last_router_target: str = ""

    # 历史记录 (用于链路追溯)
    stage_history: list[str] = field(default_factory=list)

    def record_stage(self, stage: str):
        self.current_stage = stage
        self.stage_history.append(stage)

    def increment_failure(self, agent_name: str):
        """记录Agent失败"""
        self.consecutive_failures[agent_name] = (
            self.consecutive_failures.get(agent_name, 0) + 1
        )
        self.total_retries += 1

    def reset_failures(self, agent_name: str):
        """重置Agent失败计数 (该Agent成功后调用)"""
        self.consecutive_failures[agent_name] = 0

    def should_replan(self, agent_name: str) -> bool:
        """判断是否需要重新规划 (同一Agent连续失败3次)"""
        return self.consecutive_failures.get(agent_name, 0) >= MAX_RETRIES_PER_AGENT

    def should_escalate(self) -> bool:
        """判断是否需要人工介入"""
        return self.total_retries >= MAX_TOTAL_RETRIES


# ============================================================
# 辅助函数: 从LLM输出中提取JSON
# ============================================================
def _extract_json(text: str) -> str:
    """从LLM的文本输出中提取JSON块"""
    # 尝试匹配 ```json ... ``` 代码块
    m = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if m:
        return m.group(1).strip()
    # 尝试匹配裸 {...}
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        return m.group(0).strip()
    return text.strip()


def _assemble_circuit_json(build_spec: BuildSpecification, build_result: dict) -> dict:
    """组装电路JSON — 将构建规格和构建结果组合为可导出的电路描述。

    供成功构建路径使用，将 components、connections、parameters、
    bridge_components 和 build_result 信息合并为一个 JSON。
    """
    return {
        "model_name": build_spec.model_name,
        "mode": build_spec.mode,
        "stop_time": build_spec.stop_time,
        "interval": build_spec.interval,
        "components": build_spec.components,
        "connections": build_spec.connections,
        "bridge_components": build_spec.bridge_components,
        "parameters": build_spec.parameters,
        "build_result": {
            "status": build_result.get("status", "success"),
            "ame_path": build_result.get("ame_path", ""),
            "script_path": build_result.get("script_path", ""),
            "mode": build_result.get("mode", "auto"),
            "message": build_result.get("message", ""),
        },
    }


def _display_topology_plan(plan: TopologyPlan):
    """以可读格式显示拓扑规划结果，供人工审核"""
    print("\n" + "=" * 70)
    print("  [ 物理拓扑规划 — 请审核")
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


def _review_topology_plan(plan: TopologyPlan, plan_data: dict) -> tuple[bool, str]:
    """人工审核拓扑规划。

    返回 (approved, feedback):
      - approved=True: 审核通过，进入下一阶段
      - approved=False, feedback非空: 用户提供修改意见，重新规划
      - approved=False, feedback为空: 用户选择退出
    """
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


# ============================================================
# 辅助函数: 错误诊断 → 路由决策 → 执行 (自动/人工双模式)
# ============================================================
def _manual_routing_review(
    stage_name: str,
    diag_report: dict,
    router_decision: dict,
    consecutive_failures: int,
) -> str:
    """人工审核路由决策。显示诊断报告和路由建议，等待用户选择。

    返回用户选择: "execute_auto" | "retry" | "replan" | "escalate" | "abort"
    """
    print("\n" + "=" * 70)
    print(f"  ⚠️ 错误路由审核 — [{stage_name.upper()}] 阶段出错")
    print("=" * 70)

    # 显示诊断报告
    print(f"\n  ▸ 出错阶段: {stage_name}")
    print(f"  ▸ 连续失败次数: {consecutive_failures}")

    errors = diag_report.get("errors", [])
    if errors:
        print(f"\n  ▸ 错误详情 ({len(errors)} 个):")
        for e in errors:
            print(f"    [{e.get('level', 'error').upper()}] {e.get('code', '?')}")
            print(f"    消息: {e.get('message', '')[:120]}")
            print(f"    目标Agent: {e.get('target_agent', '?')}")
            if e.get("suggested_fix"):
                print(f"    建议: {e['suggested_fix'][:120]}")

    # 显示路由推荐
    print(f"\n  ▸ 路由推荐:")
    print(f"    动作: {router_decision.get('action', '?')}")
    print(f"    目标: {router_decision.get('target_agent', '?')}")
    print(f"    理由: {router_decision.get('reason', '?')}")
    adjustments = router_decision.get('adjustments', '')
    if adjustments:
        print(f"    调整建议: {adjustments[:200]}")

    print("\n  ─────────────────────────────────────────")
    while True:
        choice = input(
            "\n  [R] 执行推荐决策  |  [M] 手动修改决策  |  [A] 中止流程\n  > "
        ).strip().upper()

        if choice == "R":
            print(f"\n  ✅ 执行推荐决策: {router_decision.get('action')}")
            return router_decision.get('action', 'retry')
        elif choice == "A":
            print("\n  ⏹ 用户中止流程")
            return "abort"
        elif choice == "M":
            print("\n  可选动作:")
            print("    1. retry    — 让目标Agent重试")
            print("    2. replan   — 回到Orchestrator重新规划")
            print("    3. escalate — 标记为需人工介入")
            manual = input("  请输入动作编号 (1/2/3): ").strip()
            action_map = {"1": "retry", "2": "replan", "3": "escalate"}
            if manual in action_map:
                action = action_map[manual]
                print(f"\n  ✅ 手动选择: {action}")
                return action
            print("  无效编号")
            continue
        else:
            print("  无效选项，请输入 R / M / A")


async def _handle_agent_error(
    stage_name: str,
    error_type: str,
    error_summary: str,
    state: PipelineState,
    run_config: RunConfig,
    diagnosis_agent,
    router_agent,
    affected_components: str = "",
) -> str:
    """处理子Agent错误: Diagnosis分类 → Router决策 → 执行。

    Args:
        stage_name: 出错阶段 (selector/bridge/parameter/builder)
        error_type: 错误大类
        error_summary: 错误描述
        state: 流水线状态
        run_config: 运行配置
        diagnosis_agent: Diagnosis Agent 实例
        router_agent: Router Agent 实例
        affected_components: 受影响的元件 (逗号分隔)

    Returns:
        决策动作: "retry" | "replan" | "escalate" | "abort"
    """
    print(f"\n  🔍 [Diagnosis] 分析 [{stage_name}] 错误...")

    # Step 1: Diagnosis — 调用 diagnose_error 工具
    diag_prompt = f"""请诊断以下子Agent错误:

阶段: {stage_name}
错误类型: {error_type}
错误描述: {error_summary}
受影响元件: {affected_components or "无"}
连续失败次数: {state.consecutive_failures.get(stage_name, 0)}

请调用 diagnose_error 工具进行分析，然后输出 DiagnosisReport JSON。"""

    try:
        diag_data = await _parse_json_output(
            diagnosis_agent, diag_prompt, run_config, f"Diagnosis-{stage_name}", max_turns=10
        )
    except Exception as e:
        print(f"  [Diagnosis] 诊断失败: {e}，使用启发式回退")
        # 回退: 根据连续失败次数简单判断
        if state.should_replan(stage_name):
            return "replan"
        return "retry"

    # 解析诊断报告
    errors = diag_data.get("errors", [])
    has_errors = diag_data.get("has_errors", len(errors) > 0)
    summary = diag_data.get("summary", "")

    if has_errors:
        print(f"  [Diagnosis] 发现 {len(errors)} 个错误:")
        for e in errors[:3]:
            print(f"    [{e.get('code', '?')}] {e.get('message', '')[:100]}")
    else:
        print(f"  [Diagnosis] 诊断完成: {summary[:100]}")

    # Step 2: Router — 决定下一步动作
    print(f"\n  🧭 [Router] 决策中...")

    router_prompt = f"""请根据以下诊断报告，决定下一步动作:

## 诊断报告
{json.dumps(diag_data, ensure_ascii=False, indent=2)}

## 流水线状态
- 阶段: {stage_name}
- 连续失败次数: {state.consecutive_failures.get(stage_name, 0)}
- 全局重试次数: {state.total_retries}
- 审核模式: {state.review_mode}

请输出路由决策JSON (含 action, target_agent, reason, adjustments, review_mode, severity)。"""

    try:
        route_data = await _parse_json_output(
            router_agent, router_prompt, run_config, f"Router-{stage_name}", max_turns=10
        )
    except Exception as e:
        print(f"  [Router] 决策失败: {e}，使用默认retry")
        return "retry" if not state.should_replan(stage_name) else "replan"

    action = route_data.get("action", "retry")
    target = route_data.get("target_agent", stage_name)
    reason = route_data.get("reason", "")
    adjustments = route_data.get("adjustments", "")
    review_mode = route_data.get("review_mode", "auto")
    severity = route_data.get("severity", "error")

    print(f"  [Router] 决策: action={action}, target={target}, review={review_mode}")
    if reason:
        print(f"           理由: {reason[:120]}")

    # Step 3: 执行决策 — 根据 review_mode 分流
    effective_review_mode = state.review_mode if state.review_mode != "auto" else review_mode

    if effective_review_mode == "manual" or severity == "critical":
        # 人工审核模式
        user_choice = _manual_routing_review(
            stage_name, diag_data, route_data,
            state.consecutive_failures.get(stage_name, 0)
        )
        # 如果用户选择执行推荐决策，沿用路由器的决定
        if user_choice == "execute_auto":
            pass  # 使用路由器的 action
        else:
            action = user_choice  # retry / replan / escalate / abort

    # Step 4: 记录 state 中的调整信息 (供 retry 时使用)
    state.last_router_adjustments = adjustments
    state.last_router_target = target

    return action


async def _run_agent(agent, prompt: str, run_config: RunConfig, stage_name: str, max_turns: int = 30) -> str:
    """运行单个Agent并返回其final_output"""
    print(f"\n  [{stage_name}] 执行中...")
    result = await Runner.run(agent, prompt, run_config=run_config, max_turns=max_turns)
    output = result.final_output
    print(f"  [{stage_name}] 完成 (output长度: {len(output)} chars)")
    return output


async def _parse_json_output(
    agent, prompt: str, run_config: RunConfig, stage_name: str, max_turns: int = 30
) -> dict:
    """运行Agent并解析其JSON输出"""
    output = await _run_agent(agent, prompt, run_config, stage_name, max_turns=max_turns)
    try:
        return json.loads(_extract_json(output))
    except json.JSONDecodeError as e:
        print(f"  [{stage_name}] JSON解析失败: {e}")
        print(f"  [{stage_name}] 原始输出前200字符: {output[:200]}")
        # 回退: 尝试直接从输出中找JSON
        raise


# ============================================================
# 主编排函数
# ============================================================
async def run_pipeline(user_request: str, skip_review: bool = False, review_mode: str = "auto") -> PipelineState:
    """运行完整的6-Agent建模流水线。

    Args:
        user_request: 用户的自然语言建模需求
        skip_review: 跳过拓扑规划的人工审核关卡 (用于自动化测试)
        review_mode: 错误路由审核模式 — "auto" 自动执行 / "manual" 人工审核每个路由决策

    Returns:
        PipelineState: 包含所有阶段结果的完整状态
    """
    run_config = create_kimi_run_config()
    state = PipelineState(user_request=user_request, review_mode=review_mode)

    print("=" * 70)
    print("  Amesim 6-Agent 自动建模系统")
    print("=" * 70)
    print(f"\n  用户需求: {user_request}\n")

    # ---- 创建所有Agent ----
    orchestrator = create_orchestrator_agent()
    router = create_routing_agent()
    selector = create_selector_agent()
    bridge = create_bridge_agent()
    parameter = create_parameter_agent()
    builder = create_builder_agent()
    diagnosis = create_diagnosis_agent()

    with trace("Amesim-6Agent-Pipeline"):

        # ====================================================
        # Stage 1: Orchestrator — 拓扑规划 + 人工审核
        # ====================================================
        state.record_stage("orchestrator_planning")
        print("\n" + "=" * 50)
        print("  [1/6] Orchestrator: 需求分解与拓扑规划")
        print("=" * 50)

        plan_prompt = f"""请对以下用户需求进行分解和拓扑规划:

用户需求: {user_request}

请严格按照系统指令中定义的JSON格式输出规划结果，必须包含 physical_flow_diagram 字段。"""

        # ====================================================
        # 主循环: 拓扑规划 ⇄ 执行 ⇄ 回退重规划
        # ====================================================
        _master_retry_count = 0
        _topology_approved = False
        _topology_plan_data = None  # 保留最近一次拓扑规划的原始 dict

        while _master_retry_count < MAX_TOTAL_RETRIES * 2:  # 外层保护
            _master_retry_count += 1

            # ── 拓扑规划阶段 (审核通过后转入执行) ──
            if not _topology_approved:
                while True:
                    plan_data = await _parse_json_output(
                        orchestrator, plan_prompt, run_config, "Orchestrator-Plan"
                    )
                    _topology_plan_data = plan_data  # 保留供 replan 参考

                    state.topology_plan = TopologyPlan(
                        user_request=user_request,
                        system_description=plan_data.get("system_description", ""),
                        physical_domains=plan_data.get("physical_domains", []),
                        component_requirements=[],
                        topology_description=plan_data.get("topology_description", ""),
                        physical_flow_diagram=plan_data.get("physical_flow_diagram", ""),
                        notes=plan_data.get("notes", ""),
                    )

                    from .protocols import ComponentRequirement
                    for req in plan_data.get("component_requirements", []):
                        state.topology_plan.component_requirements.append(ComponentRequirement(
                            index=req.get("index", 0),
                            functional_description=req.get("functional_description", ""),
                            physical_domain=req.get("physical_domain", ""),
                            suggested_library=req.get("suggested_library"),
                            quantity=req.get("quantity", 1),
                            keywords=req.get("keywords", []),
                            topological_role=req.get("topological_role", ""),
                        ))

                    # ── 人工审核关卡 ──
                    if skip_review:
                        print("\n  ⏩ [skip-review] 跳过人工审核，自动通过")
                        _topology_approved = True
                        break

                    approved, feedback = _review_topology_plan(
                        state.topology_plan, plan_data
                    )

                    if approved:
                        _topology_approved = True
                        break  # 审核通过, 进入执行阶段

                    if not feedback:
                        print("  流程终止。")
                        return state

                    # 用户提供修改意见, 重新规划
                    plan_prompt = f"""请根据以下修改意见重新进行拓扑规划:

## 修改意见
{feedback}

## 原始需求
用户需求: {user_request}

## 上次规划 (供参考, 请根据修改意见调整)
{state.topology_plan.to_json()}

请严格按照系统指令中定义的JSON格式输出新的规划结果，必须包含 physical_flow_diagram 字段。"""
                    state.topology_plan.component_requirements = []

                if not _topology_approved:
                    continue  # 不应该到这里, 安全保护

            print(f"  规划: {state.topology_plan.system_description[:100]}...")
            print(f"  物理域: {state.topology_plan.physical_domains}")
            print(f"  元件需求数: {len(state.topology_plan.component_requirements)}")

            # ====================================================
            # 执行阶段 — Stages 2-6 (支持 Diagnosis→回退重试)
            # ====================================================
            # 流水线阶段追踪: "selector" → "bridge" → "parameter" → "builder" → "diagnosis"
            # Diagnosis 可能回退到 "bridge" / "parameter" / "selector" / "orchestrator"
            _pipeline_stage = "selector"
            _full_pipeline_success = False

            while state.total_retries < MAX_TOTAL_RETRIES:
                # ── 退出条件 ──
                if state.should_escalate():
                    print("  [!!!] 达到全局最大重试次数，需要人工介入")
                    break

                # Stage: Selector (NEW LLM deep reasoning pipeline)
                if _pipeline_stage == "selector":
                    model_name = state.user_request[:40].replace(" ", "_").replace("/", "_")
                    from .pipeline_context import PipelineContext as PCtx
                    ctx = PCtx(model_name)
                    next_stage = await run_new_selector_pipeline(
                        state, run_config, ctx, verbose=getattr(state, "verbose", True)
                    )
                    if next_stage == "bridge":
                        _pipeline_stage = "bridge"
                        continue
                    elif next_stage == "orchestrator":
                        _pipeline_stage = "selector"
                        continue
                    else:
                        return state

                # ====================================================
                # Stage: Bridge — 元件连接规划
                # ====================================================
                if _pipeline_stage == "bridge":
                    state.record_stage("bridge")
                    print("\n" + "=" * 50)
                    print("  [Bridge] 元件审查与连接规划")
                    print("=" * 50)

                    components_for_bridge = [
                        {"alias": s.alias, "icon_key": s.icon_key,
                         "icon_name": s.icon_name, "library": s.library,
                         "submodel_id": s.recommended_submodel}
                        for s in state.component_selection.selected
                    ]

                    # 构建 bridge_prompt — 如果是 Diagnosis 回退过来的，带上错误信息
                    bridge_prompt = f"""请对以下Selector选定的元件进行三步审查和连接规划:

    ## 已选定的元件
    {json.dumps(components_for_bridge, ensure_ascii=False, indent=2)}

    ## 拓扑关系描述
    {state.topology_plan.topology_description}

    ## 系统描述
    {state.topology_plan.system_description}"""

                    # 如果是回退, 附加上次构建的错误信息
                    if state.last_router_adjustments:
                        bridge_prompt += f"""

    ## ⚠️ 上次构建错误 (需修正)
    {state.last_router_adjustments}

    请根据错误信息修正连接方案，确保端口兼容。"""
                    if state.build_result and state.build_result.stderr:
                        bridge_prompt += f"""

    ## 上次构建stderr
    {state.build_result.stderr[:800]}"""

                    bridge_prompt += "\n\n请按Step 1(存在性审查) → Step 2(端口规划) → Step 3(跨域桥接)的流程，输出ConnectionPlan JSON。"

                    bridge_success = False
                    bridge_replan = False
                    while True:
                        try:
                            conn_data = await _parse_json_output(
                                bridge, bridge_prompt, run_config, "Bridge", max_turns=80
                            )
                        except json.JSONDecodeError as e:
                            state.increment_failure("bridge")
                            action = await _handle_agent_error(
                                "bridge", "json_parse_failed",
                                f"JSON解析失败: {str(e)[:200]}",
                                state, run_config, diagnosis, router,
                            )
                            if action == "replan":
                                bridge_replan = True; break
                            elif action in ("escalate", "abort"):
                                return state
                            continue
                        except Exception as e:
                            state.increment_failure("bridge")
                            action = await _handle_agent_error(
                                "bridge", "max_turns_exceeded",
                                f"Bridge运行异常: {str(e)[:300]}",
                                state, run_config, diagnosis, router,
                            )
                            if action == "replan":
                                bridge_replan = True; break
                            elif action in ("escalate", "abort"):
                                return state
                            continue

                        from .protocols import PortConnection, BridgeComponent
                        connections = []
                        for c in conn_data.get("connections", []):
                            connections.append(PortConnection(
                                from_alias=c.get("from_alias", ""),
                                from_port=c.get("from_port", 0),
                                to_alias=c.get("to_alias", ""),
                                to_port=c.get("to_port", 0),
                                connection_type=c.get("connection_type", "line"),
                                line_alias=c.get("line_alias", ""),
                                waypoints=[tuple(w) for w in c.get("waypoints", [])],
                                port_type_from=c.get("port_type_from", ""),
                                port_type_to=c.get("port_type_to", ""),
                            ))
                        bridges = []
                        for b in conn_data.get("bridge_components", []):
                            pos = b.get("position", [0, 0])
                            bridges.append(BridgeComponent(
                                icon_key=b.get("icon_key", ""),
                                icon_name=b.get("icon_name", ""),
                                library=b.get("library", ""),
                                alias=b.get("alias", ""),
                                position=tuple(pos) if len(pos) == 2 else (0, 0),
                                submodel=b.get("submodel", ""),
                                submodel_path=b.get("submodel_path", ""),
                                reason=b.get("reason", ""),
                            ))

                        is_valid = conn_data.get("is_valid", True)
                        validation_issues = conn_data.get("validation_issues", [])
                        state.connection_plan = ConnectionPlan(
                            connections=connections, bridge_components=bridges,
                            validation_issues=validation_issues, is_valid=is_valid,
                        )

                        if not is_valid:
                            print(f"  [Bridge] 验证不通过: {len(validation_issues)} 个问题")
                            state.increment_failure("bridge")
                            issues_summary = "; ".join(str(v)[:150] for v in validation_issues[:5])
                            action = await _handle_agent_error(
                                "bridge", "validation_failed",
                                f"连接验证不通过: {issues_summary[:400]}",
                                state, run_config, diagnosis, router,
                            )
                            if action == "replan":
                                bridge_replan = True; break
                            elif action in ("escalate", "abort"):
                                return state
                            adjustments = state.last_router_adjustments or "修正验证问题并重新规划连接"
                            bridge_prompt = f"""上次连接规划验证不通过，请{adjustments}:

    ## 验证问题
    {json.dumps(validation_issues, ensure_ascii=False, indent=2)}

    ## 元件列表 (同上)
    {json.dumps(components_for_bridge, ensure_ascii=False, indent=2)}

    请修正连接方案。"""
                            continue
                        else:
                            state.reset_failures("bridge")
                            print(f"  [Bridge] 规划 {len(connections)} 个连接, {len(bridges)} 个桥接元件")
                            bridge_success = True
                            break

                    if bridge_replan:
                        _pipeline_stage = "orchestrator"; continue
                    if not bridge_success:
                        if state.connection_plan is None:
                            print("\n  ⚠️ [Gate] Bridge 未产出有效连接方案")
                            _pipeline_stage = "orchestrator"; continue
                    _pipeline_stage = "parameter"
                    continue

                # ====================================================
                # Stage: Parameter — 参数设置
                # ====================================================
                if _pipeline_stage == "parameter":
                    state.record_stage("parameter")
                    print("\n" + "=" * 50)
                    print("  [Parameter Agent] 参数查询与设置")
                    print("=" * 50)

                    params_input = [
                        {"alias": s.alias, "library": s.library,
                         "submodel_id": s.recommended_submodel}
                        for s in state.component_selection.selected
                    ]

                    param_prompt = f"""请为以下元件查询并设置参数:

    ## 元件列表
    {json.dumps(params_input, ensure_ascii=False, indent=2)}

    ## 用户需求
    {user_request}

    ## 系统描述
    {state.topology_plan.system_description}

    请使用 batch_query_params 工具查询所有元件的参数，然后根据物理常识推断合理的参数值。"""

                    param_success = False
                    param_replan = False
                    while True:
                        try:
                            param_data = await _parse_json_output(
                                parameter, param_prompt, run_config, "Parameter"
                            )
                        except json.JSONDecodeError as e:
                            state.increment_failure("parameter")
                            action = await _handle_agent_error(
                                "parameter", "json_parse_failed",
                                f"JSON解析失败: {str(e)[:200]}",
                                state, run_config, diagnosis, router,
                            )
                            if action == "replan":
                                param_replan = True; break
                            elif action in ("escalate", "abort"):
                                return state
                            continue
                        except Exception as e:
                            state.increment_failure("parameter")
                            action = await _handle_agent_error(
                                "parameter", "param_failed",
                                f"Parameter运行异常: {str(e)[:300]}",
                                state, run_config, diagnosis, router,
                            )
                            if action == "replan":
                                param_replan = True; break
                            elif action in ("escalate", "abort"):
                                return state
                            continue

                        from .protocols import ComponentParams
                        assignments = []
                        for a in param_data.get("assignments", []):
                            assignments.append(ComponentParams(
                                alias=a.get("alias", ""),
                                icon_name=a.get("icon_name", ""),
                                submodel_id=a.get("submodel_id", ""),
                                params=a.get("params", {}),
                                params_detail=a.get("params_detail", []),
                            ))
                        state.parameter_assignment = ParameterAssignment(
                            assignments=assignments,
                            warnings=param_data.get("warnings", []),
                        )
                        state.reset_failures("parameter")
                        print(f"  [Parameter] 为 {len(assignments)} 个元件设置了参数")
                        param_success = True
                        break

                    if param_replan:
                        _pipeline_stage = "orchestrator"; continue
                    if not param_success:
                        if state.parameter_assignment is None:
                            print("\n  ⚠️ [Gate] Parameter 未产出有效参数设置")
                            _pipeline_stage = "orchestrator"; continue
                    _pipeline_stage = "builder"
                    continue

                # ====================================================
                # Stage: Builder — 模型构建 (带 while 重试)
                # ====================================================
                if _pipeline_stage == "builder":
                    state.record_stage("builder")
                    print("\n" + "=" * 50)
                    print("  [Builder Agent] 模型构建")
                    print("=" * 50)

                    builder_success = False
                    builder_replan = False
                    while True:
                        # 组装完整构建规格
                        all_components = []
                        for s in state.component_selection.selected:
                            all_components.append({
                                "icon_name": s.icon_name, "alias": s.alias,
                                "position": list(s.position),
                                "submodel": s.recommended_submodel,
                                "library": s.library, "rotations": s.rotations,
                                "flip": s.flip,
                            })
                        for b in state.connection_plan.bridge_components:
                            all_components.append({
                                "icon_name": b.icon_name, "alias": b.alias,
                                "position": list(b.position),
                                "submodel": b.submodel, "library": b.library,
                                "rotations": 0, "flip": False,
                            })

                        all_connections = []
                        for c in state.connection_plan.connections:
                            conn_entry = {
                                "from_alias": c.from_alias, "from_port": c.from_port,
                                "to_alias": c.to_alias, "to_port": c.to_port,
                                "type": c.connection_type,
                                "line_alias": c.line_alias or f"wire_{len(all_connections)+1}",
                            }
                            if c.waypoints:
                                conn_entry["waypoints"] = [list(w) for w in c.waypoints]
                            all_connections.append(conn_entry)

                        build_spec = BuildSpecification(
                            model_name=user_request[:40].replace(" ", "_").replace("/", "_"),
                            components=all_components,
                            connections=all_connections,
                            bridge_components=[
                                {"icon_name": b.icon_name, "alias": b.alias,
                                 "position": list(b.position),
                                 "submodel": b.submodel, "library": b.library}
                                for b in state.connection_plan.bridge_components
                            ],
                            parameters=state.parameter_assignment.to_dict(),
                            stop_time="10", interval="0.01", mode="auto",
                        )

                        # ★ Step 3: PRIMARY PATH — build_model_direct(mode="auto")
                        code_ref_result = None
                        try:
                            direct_result = build_model_direct(
                                model_name=build_spec.model_name,
                                components=build_spec.components,
                                connections=build_spec.connections,
                                parameters=build_spec.parameters,
                                bridge_components=build_spec.bridge_components,
                                stop_time=build_spec.stop_time,
                                interval=build_spec.interval,
                                mode="auto",
                            )
                            code_ref_result = json.loads(direct_result)
                            status = code_ref_result.get("status", "error")
                            
                            if status == "success":
                                # ★ SUCCESS PATH — ZERO LLM calls, return immediately
                                print(f"  [Builder] 直接构建成功: {code_ref_result.get('ame_path', 'N/A')}")
                                state.build_result = BuildResult(
                                    status=status,
                                    model_name=code_ref_result.get("model_name", build_spec.model_name),
                                    ame_path=code_ref_result.get("ame_path", ""),
                                    script_path=code_ref_result.get("script_path", ""),
                                    stdout=code_ref_result.get("stdout", ""),
                                    stderr=code_ref_result.get("stderr", ""),
                                    returncode=code_ref_result.get("returncode", 0),
                                    mode=code_ref_result.get("mode", "auto"),
                                    message=code_ref_result.get("message", ""),
                                )
                                # ★ Save extracted circuit JSON for downstream consumption
                                circuit_data = _assemble_circuit_json(build_spec, code_ref_result)
                                ctx = PipelineContext(model_name=build_spec.model_name)
                                ctx.save_extracted_json(circuit_data)
                                print(f"  [Builder] 状态: {state.build_result.status}")
                                print(f"  [Builder] 模型文件: {state.build_result.ame_path}")
                                builder_success = True
                                break  # 跳出 builder while 循环
                            else:
                                # 直接构建失败，进入修复循环
                                print(f"  [Builder] 直接构建失败，进入LLM修复循环...")
                        except Exception as e:
                            print(f"  [Builder] 直接构建异常: {e}")
                            code_ref_result = {"status": "error", "message": str(e), "stderr": str(e)}

                        # ★ Step 4: REPAIR LOOP (only on direct failure, max 3 attempts)
                        if not builder_success:
                            repair_attempt = 0
                            max_repair_attempts = 3
                            while repair_attempt < max_repair_attempts:
                                repair_attempt += 1
                                print(f"  [Builder] 修复尝试 {repair_attempt}/{max_repair_attempts}")
                                
                                # 诊断失败原因
                                diagnosis_prompt = f"""请分析以下模型构建结果中的错误:

    ## Direct BuildResult
    {json.dumps(code_ref_result, ensure_ascii=False, indent=2)}

    ## BuildSpecification
    {json.dumps({
        "model_name": build_spec.model_name,
        "components": build_spec.components,
        "connections": build_spec.connections,
        "bridge_components": build_spec.bridge_components,
        "parameters": build_spec.parameters,
        "stop_time": build_spec.stop_time,
        "interval": build_spec.interval,
    }, ensure_ascii=False, indent=2)}

    请分析错误并提供针对性的修复建议 (fixes)。"""

                                diag_data = {}
                                try:
                                    diag_data = await _parse_json_output(
                                        diagnosis, diagnosis_prompt, run_config, "Diagnosis"
                                    )
                                except json.JSONDecodeError:
                                    diag_data = {
                                        "errors": [{"level": "error", "code": "UNKNOWN_ERROR",
                                                    "message": "诊断Agent未能正常输出",
                                                    "target_agent": "builder",
                                                    "suggested_fix": "检查构建日志"}],
                                        "warnings": [], "has_errors": True,
                                        "summary": "诊断Agent异常",
                                    }

                                # 检查是否有错误需要修复
                                errors = diag_data.get("errors", [])
                                if not errors:
                                    print(f"  [Diagnosis] 无错误可修复")
                                    break

                                # 生成针对性修复 (不是完整重写)
                                fix_prompt = f"""请根据以下诊断结果，生成针对性的修复代码。

    ## 错误信息
    {json.dumps(errors, ensure_ascii=False, indent=2)}

    ## 当前构建规格
    {json.dumps({
        "model_name": build_spec.model_name,
        "components": build_spec.components,
        "connections": build_spec.connections,
        "bridge_components": build_spec.bridge_components,
        "parameters": build_spec.parameters,
        "stop_time": build_spec.stop_time,
        "interval": build_spec.interval,
    }, ensure_ascii=False, indent=2)}

    请生成修复后的构建规格 (仅包含需要修改的部分)，返回JSON格式:
    {{
        "fixes": [
            {{"line": N, "old": "...", "new": "..."}}
        ]
    }}"""

                                try:
                                    fix_data = await _parse_json_output(
                                        builder, fix_prompt, run_config, "Builder"
                                    )
                                    # 应用修复并重新执行
                                    fixes = fix_data.get("fixes", [])
                                    if fixes:
                                        # 读取现有构建脚本
                                        script_path = code_ref_result.get("script_path", "")
                                        if not script_path or not os.path.exists(script_path):
                                            print(f"  [Builder] 脚本路径无效，跳过修复: {script_path}")
                                            code_ref_result = {
                                                "status": "error",
                                                "model_name": build_spec.model_name,
                                                "mode": "auto",
                                                "ame_path": "",
                                                "script_path": script_path,
                                                "stdout": "",
                                                "stderr": "Script path invalid for repair",
                                                "returncode": -1,
                                                "message": "无法应用修复：脚本路径无效",
                                            }
                                            continue

                                        # 逐行应用修复 (old → new 替换)
                                        with open(script_path, 'r', encoding='utf-8') as f:
                                            script_lines = f.readlines()

                                        applied_count = 0
                                        for fix in fixes:
                                            line_no = fix.get("line", 0)
                                            old_text = fix.get("old", "")
                                            new_text = fix.get("new", "")
                                            if 0 < line_no <= len(script_lines):
                                                if old_text in script_lines[line_no - 1]:
                                                    script_lines[line_no - 1] = (
                                                        script_lines[line_no - 1].replace(old_text, new_text)
                                                    )
                                                    applied_count += 1
                                                else:
                                                    print(f"  [Builder] fix line={line_no}: old_text not found in script")
                                            else:
                                                print(f"  [Builder] fix line={line_no}: out of range (1-{len(script_lines)})")

                                        if applied_count == 0:
                                            print(f"  [Builder] 无修复可应用 ({len(fixes)} 条fix均不匹配)")
                                            code_ref_result = {
                                                "status": "error",
                                                "model_name": build_spec.model_name,
                                                "mode": "auto",
                                                "ame_path": "",
                                                "script_path": script_path,
                                                "stdout": "",
                                                "stderr": f"No fixes applied ({len(fixes)} attempted)",
                                                "returncode": -1,
                                                "message": f"修复不匹配：{len(fixes)} 条fix均无法应用到脚本",
                                            }
                                            continue

                                        # 写入修复后的脚本 (新文件，保留原脚本便于对比)
                                        fixed_script_path = script_path.replace(".py", "_fixed.py")
                                        with open(fixed_script_path, 'w', encoding='utf-8') as f:
                                            f.writelines(script_lines)
                                        print(f"  [Builder] 已应用 {applied_count}/{len(fixes)} 条修复 → {fixed_script_path}")

                                        # 通过 AMEPython.exe 执行修复后的脚本
                                        amesim_exe = os.environ.get(
                                            "AMESIM_EXE", r"D:\AMESIM24\Amesim\win64\AMESim.exe"
                                        )
                                        ame_python = os.path.join(os.path.dirname(amesim_exe), "AMEPython.exe")
                                        ame_python = os.path.normpath(ame_python)
                                        model_dir = os.path.dirname(script_path)

                                        if not os.path.exists(ame_python):
                                            retry_data = {
                                                "status": "error",
                                                "model_name": build_spec.model_name,
                                                "mode": "auto",
                                                "ame_path": "",
                                                "script_path": fixed_script_path,
                                                "stdout": "",
                                                "stderr": f"未找到 AMEPython.exe: {ame_python}",
                                                "returncode": -1,
                                                "message": "未找到Amesim Python解释器",
                                            }
                                        else:
                                            try:
                                                proc = subprocess.run(
                                                    [ame_python, fixed_script_path],
                                                    capture_output=True, text=True, timeout=300,
                                                    cwd=model_dir,
                                                )
                                                ame_path = os.path.join(
                                                    model_dir, f"{build_spec.model_name}.ame"
                                                )
                                                retry_data = {
                                                    "status": "success" if proc.returncode == 0 else "error",
                                                    "model_name": build_spec.model_name,
                                                    "mode": "auto",
                                                    "ame_path": ame_path if proc.returncode == 0 else "",
                                                    "script_path": fixed_script_path,
                                                    "stdout": proc.stdout or "",
                                                    "stderr": proc.stderr or "",
                                                    "returncode": proc.returncode,
                                                    "message": (
                                                        f"模型 {build_spec.model_name} 构建成功"
                                                        if proc.returncode == 0
                                                        else f"模型 {build_spec.model_name} 构建失败 (returncode={proc.returncode})"
                                                    ),
                                                }
                                            except Exception as ex:
                                                retry_data = {
                                                    "status": "error",
                                                    "model_name": build_spec.model_name,
                                                    "mode": "auto",
                                                    "ame_path": "",
                                                    "script_path": fixed_script_path,
                                                    "stdout": "",
                                                    "stderr": str(ex),
                                                    "returncode": -1,
                                                    "message": f"执行修复脚本异常: {ex}",
                                                }

                                        if retry_data.get("status") == "success":
                                            print(f"  [Builder] 修复后构建成功")
                                            state.build_result = BuildResult(
                                                status="success",
                                                model_name=retry_data.get("model_name", build_spec.model_name),
                                                ame_path=retry_data.get("ame_path", ""),
                                                script_path=retry_data.get("script_path", ""),
                                                stdout=retry_data.get("stdout", ""),
                                                stderr=retry_data.get("stderr", ""),
                                                returncode=retry_data.get("returncode", 0),
                                                mode="auto",
                                                message=retry_data.get("message", ""),
                                            )
                                            builder_success = True
                                            break
                                        else:
                                            code_ref_result = retry_data
                                            print(f"  [Builder] 修复后仍然失败")
                                            continue
                                    else:
                                        print(f"  [Builder] 无可应用的修复")
                                        break
                                except json.JSONDecodeError as e:
                                    print(f"  [Builder] 修复解析失败: {e}")
                                    continue
                                except Exception as e:
                                    print(f"  [Builder] 修复异常: {e}")
                                    continue

                            if not builder_success:
                                # 修复循环耗尽，标记失败
                                state.increment_failure("builder")
                                if state.should_replan("builder"):
                                    builder_replan = True; break
                                # 继续外层 while 循环重试

                        # Record failure state from code_ref_result for diagnosis
                        state.build_result = BuildResult(
                            status=code_ref_result.get("status", "error"),
                            model_name=code_ref_result.get("model_name", build_spec.model_name),
                            ame_path=code_ref_result.get("ame_path", ""),
                            script_path=code_ref_result.get("script_path", ""),
                            stdout=code_ref_result.get("stdout", ""),
                            stderr=code_ref_result.get("stderr", ""),
                            returncode=code_ref_result.get("returncode", -1),
                            mode=code_ref_result.get("mode", "auto"),
                            message=code_ref_result.get("message", ""),
                        )
                        print(f"  [Builder] 状态: {state.build_result.status}")

                        # Builder 失败 → 运行 Diagnosis 决定下一步
                        state.record_stage("diagnosis")
                        print("\n  [Diagnosis] 分析构建失败原因...")

                        diagnosis_prompt = f"""请分析以下模型构建结果中的错误:

    ## BuildResult
    {state.build_result.to_json()}

    请使用 parse_build_output 解析输出，然后对每个错误调用 classify_error 进行分类。"""

                        diag_data = {}
                        try:
                            diag_data = await _parse_json_output(
                                diagnosis, diagnosis_prompt, run_config, "Diagnosis"
                            )
                        except json.JSONDecodeError:
                            print("  [Diagnosis] JSON解析失败")
                            diag_data = {
                                "errors": [{"level": "error", "code": "UNKNOWN_ERROR",
                                            "message": "诊断Agent未能正常输出",
                                            "target_agent": "builder",
                                            "suggested_fix": "检查构建日志"}],
                                "warnings": [], "has_errors": True,
                                "summary": "诊断Agent异常",
                            }

                        from .protocols import ErrorItem
                        errors = []
                        for e in diag_data.get("errors", []):
                            errors.append(ErrorItem(
                                level=e.get("level", "error"),
                                code=e.get("code", "UNKNOWN"),
                                message=e.get("message", ""),
                                target_agent=e.get("target_agent", "builder"),
                                suggested_fix=e.get("suggested_fix", ""),
                                affected_components=e.get("affected_components", []),
                            ))
                        warnings = []
                        for w in diag_data.get("warnings", []):
                            warnings.append(ErrorItem(
                                level="warning",
                                code=w.get("code", "WARNING"),
                                message=w.get("message", ""),
                                target_agent=w.get("target_agent", ""),
                                suggested_fix=w.get("suggested_fix", ""),
                                affected_components=w.get("affected_components", []),
                            ))
                        state.diagnosis_report = DiagnosisReport(
                            errors=errors, warnings=warnings,
                            has_errors=diag_data.get("has_errors", len(errors) > 0),
                            summary=diag_data.get("summary", ""),
                        )

                        if state.diagnosis_report.has_errors:
                            print(f"  [Diagnosis] 发现 {len(errors)} 个错误, {len(warnings)} 个警告")
                            print(f"  [Diagnosis] 摘要: {state.diagnosis_report.summary}")

                            retry_targets = state.diagnosis_report.get_retry_targets()
                            if not retry_targets:
                                # 没有明确目标 → 默认回退 bridge
                                retry_targets = [("bridge", errors)]

                            for target_agent, target_errors in retry_targets:
                                state.increment_failure(target_agent)
                                print(f"  [Orchestrator] → {target_agent} 需要修正 ({len(target_errors)} 个错误)")

                                if state.should_replan(target_agent):
                                    print(f"  [!] {target_agent} 连续失败3次，回退到Orchestrator重新规划")
                                    builder_replan = True
                                    break

                                # ★ 收集错误信息作为调整建议
                                fix_hints = "; ".join(
                                    f"[{e.code}] {e.message[:80]}" + (f" → {e.suggested_fix[:80]}" if e.suggested_fix else "")
                                    for e in target_errors[:3]
                                )
                                state.last_router_adjustments = (
                                    f"上次构建失败: {fix_hints}"
                                )
                                state.last_router_target = target_agent

                                # ★ 设置回退目标阶段
                                if target_agent == "bridge":
                                    _pipeline_stage = "bridge"
                                elif target_agent == "parameter":
                                    _pipeline_stage = "parameter"
                                elif target_agent == "selector":
                                    _pipeline_stage = "selector"
                                elif target_agent == "orchestrator":
                                    _pipeline_stage = "orchestrator"
                                else:
                                    # builder 自己 → 继续 builder while 循环
                                    _pipeline_stage = "builder"

                                break  # 只处理第一个 target

                            if builder_replan:
                                break  # 跳出 builder while 循环
                            if _pipeline_stage != "builder":
                                break  # 跳出 builder while → 主循环跳转到目标阶段
                            # _pipeline_stage == "builder" → 继续 builder while 循环重试
                            continue
                        else:
                            print(f"  [Diagnosis] 无错误, {len(warnings)} 个警告")
                            # 无诊断错误但构建失败 → 重试 builder
                            continue

                    if builder_replan:
                        _pipeline_stage = "orchestrator"; continue
                    if builder_success:
                        _pipeline_stage = "success"; break
                    # 非成功非 replan → 可能已跳转到 bridge/parameter/selector → 主循环继续
                    continue

                # ====================================================
                # Stage: Orchestrator replan (回退重规划)
                # ====================================================
                if _pipeline_stage == "orchestrator":
                    print("\n  🔄 [Orchestrator] 回退重新规划...")
                    state.record_stage("orchestrator_replan")
                    # 回到拓扑规划循环 (外层 while True)
                    break  # 跳出主执行循环, 回到拓扑规划的 while True

                # ── 未知阶段, 跳出 ──
                if _pipeline_stage == "success":
                    _full_pipeline_success = True
                    break

                # 安全保护: 未知阶段
                print(f"  [!!!] 未知流水线阶段: {_pipeline_stage}")
                break

            # ── 如果跳回拓扑规划, 重新进入外层循环 ──
            if _pipeline_stage == "orchestrator":
                state.topology_plan.component_requirements = []
                _topology_approved = False  # ★ 重置, 下次循环进入拓扑规划
                plan_prompt = f"""请根据以下修改意见重新进行拓扑规划:

## 诊断报告
{state.diagnosis_report.to_json() if state.diagnosis_report else '无'}
上次调整建议: {state.last_router_adjustments}

## 原始需求
用户需求: {user_request}

## 上次规划 (供参考, 请根据诊断结果调整)
{json.dumps(plan_data, ensure_ascii=False)[:800] if plan_data else '无'}

请严格按照系统指令中定义的JSON格式输出新的规划结果，必须包含 physical_flow_diagram 字段。"""
                continue  # 回到最外层拓扑规划 while True

            if _full_pipeline_success:
                # 构建成功 → 结束 (无需单独 Diagnosis 阶段)
                state.record_stage("diagnosis")
                state.diagnosis_report = DiagnosisReport(
                    errors=[], warnings=[], has_errors=False,
                    summary="构建成功，无需修正",
                )
                break  # 跳出拓扑规划循环

            # ── 达到最大重试或 escalate ──
            break

    # ========================================================
    # 最终报告
    # ========================================================
    print("\n" + "=" * 70)
    print("  流水线执行完成")
    print("=" * 70)
    print(f"  历史阶段: {' → '.join(state.stage_history)}")
    print(f"  总重试次数: {state.total_retries}")

    if state.build_result and state.build_result.status == "success":
        print(f"  模型文件: {state.build_result.ame_path}")
    elif state.diagnosis_report:
        print(f"  诊断: {state.diagnosis_report.summary}")

    return state


# ============================================================
# CLI 入口
# ============================================================
def _check_api_key():
    """检查是否至少配置了一个 API Key（DeepSeek 或 Kimi）。"""
    from .model_providers import DEEPSEEK_API_KEY, KIMI_API_KEY

    ds_key = os.environ.get("DEEPSEEK_API_KEY") or DEEPSEEK_API_KEY
    kimi_key = os.environ.get("KIMI_API_KEY") or KIMI_API_KEY

    available = []
    if ds_key:
        available.append(f"DeepSeek (deepseek-chat)")
    if kimi_key:
        available.append(f"Kimi (kimi-for-coding)")

    if not available:
        print("=" * 60)
        print("  错误: 未配置任何 API Key")
        print("=" * 60)
        print()
        print("配置方式 (任选其一):")
        print('  1. 环境变量:')
        print('     $env:DEEPSEEK_API_KEY = "sk-your-deepseek-key"')
        print('     $env:KIMI_API_KEY = "sk-your-kimi-key"')
        print('  2. 代码硬编码: 编辑 amesim_multi_agent/model_providers.py')
        print('     修改 DEEPSEEK_API_KEY / KIMI_API_KEY 变量')
        print()
        print("多 Key 自动切换: DeepSeek → Kimi (任一耗尽自动切换)")
        print()
        sys.exit(1)

    print(f"[API] 可用 Key: {', '.join(available)}")
    print("[API] 切换策略: DeepSeek → Kimi (配额耗尽自动切换)")
    print()


async def main(user_request: str, skip_review: bool = False, review_mode: str = "auto"):
    """主入口：运行6-Agent建模流水线。"""
    _check_api_key()
    state = await run_pipeline(user_request, skip_review=skip_review, review_mode=review_mode)
    return state


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python -m amesim_multi_agent.orchestrator_runner '<需求描述>' [--skip-review] [--review-mode auto|manual]")
        print()
        print("选项:")
        print("  --skip-review           跳过拓扑规划人工审核关卡")
        print("  --review-mode auto       错误路由自动执行 (默认)")
        print("  --review-mode manual     错误路由需人工审核确认")
        print()
        print("示例:")
        print('  uv run python -m amesim_multi_agent.orchestrator_runner "构建一个质量块连弹簧再连阻尼器的系统"')
        print('  uv run python -m amesim_multi_agent.orchestrator_runner "构建一个质量块连弹簧再连阻尼器的系统" --skip-review')
        print('  uv run python -m amesim_multi_agent.orchestrator_runner "构建一个朗肯循环发电模型" --review-mode manual')
        sys.exit(1)

    skip_review = "--skip-review" in sys.argv

    # 解析 --review-mode
    review_mode = "auto"
    for i, arg in enumerate(sys.argv):
        if arg == "--review-mode" and i + 1 < len(sys.argv):
            mode_val = sys.argv[i + 1]
            if mode_val in ("auto", "manual"):
                review_mode = mode_val
            break

    # 过滤掉标志参数
    args = []
    skip_next = False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a == "--skip-review":
            continue
        if a == "--review-mode":
            skip_next = True
            continue
        args.append(a)

    user_request = " ".join(args)
    asyncio.run(main(user_request, skip_review=skip_review, review_mode=review_mode))
