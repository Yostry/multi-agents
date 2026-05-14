"""
Parameter-Runner Agent — 参数运行 Agent 工厂函数

职责: 子模型设定 → 参数赋值 → 构建 → 运行 → 提取结果 → 对比
"""

from __future__ import annotations

from agents import Agent

from ...model_providers import get_multi_provider, DEEPSEEK_MODEL
from ..prompts.parameter_runner_prompt import PARAMETER_RUNNER_SYSTEM_PROMPT


def create_parameter_runner_agent() -> Agent:
    """创建参数运行 Agent。

    Tools 由 pipeline 运行时注入, 包含:
      - batch_query_params (parameter_tools)
      - query_submodel_params (parameter_tools)
      - build_and_run_model (builder_tools)
      - repair_build_script (builder_tools)
      - diagnose_error (diagnosis_tools)
      - classify_error (diagnosis_tools)
      - parse_build_output (diagnosis_tools)
    """
    return Agent(
        name="ParameterRunner",
        instructions=PARAMETER_RUNNER_SYSTEM_PROMPT,
        model=get_multi_provider().get_model(DEEPSEEK_MODEL),
        tools=[],
    )


def create_parameter_runner_with_tools(
    batch_query_params,
    query_submodel_params,
    build_and_run_model,
    repair_build_script,
    diagnose_error,
    classify_error,
    parse_build_output,
) -> Agent:
    """创建带完整工具集的参数运行 Agent。"""
    return Agent(
        name="ParameterRunner",
        instructions=PARAMETER_RUNNER_SYSTEM_PROMPT,
        model=get_multi_provider().get_model(DEEPSEEK_MODEL),
        tools=[
            batch_query_params,
            query_submodel_params,
            build_and_run_model,
            repair_build_script,
            diagnose_error,
            classify_error,
            parse_build_output,
        ],
    )
