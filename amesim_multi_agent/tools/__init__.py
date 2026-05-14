"""Amesim 6-Agent 工具集

条件导入: 在无 agents 包的独立环境中, 仅导入 diagnosis_tools (纯Python)。
在 Agent 环境中, 完整导入所有工具。
"""

# step_tracker 始终可用 (纯Python, 无 agents 依赖)
from .step_tracker import StepTracker, get_tracker, reset_tracker

# diagnosis_tools 始终可用 (纯Python, 无 agents 依赖)
from .diagnosis_tools import (
    parse_build_output,
    classify_error,
    classify_error_core,
    parse_build_output_core,
    print_diagnosis_report,
)

# 以下依赖 agents 包, 仅在 Agent 环境中可用
try:
    from .selector_tools import search_component, get_component_detail, recommend_submodel
    from .bridge_tools import (
        verify_all_components_exist,
        query_component_ports,
        find_port_connections,
        find_bridge_paths,
        validate_connection_plan,
    )
    from .parameter_tools import query_submodel_params, batch_query_params
    from .builder_tools import (
        generate_build_script_only,
        build_and_run_model,
        take_model_screenshot,
    )
    # 兼容旧导入
    from .validator_tools import find_bridges, validate_connection
    _HAS_AGENT_TOOLS = True
except ImportError:
    _HAS_AGENT_TOOLS = False
