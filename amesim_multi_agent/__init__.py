"""
Amesim 6-Agent 自动化仿真系统

基于 OpenAI Agents SDK 构建，采用程序化编排实现从自然语言需求到
Amesim 模型构建的完整闭环。

Agent 架构 (6-Agent流水线):
    OrchestratorAgent  (总管: 需求分解、拓扑规划、错误路由)
        ↓
    SelectorAgent      (元件选择: 数据库查询、子模型推荐、不编造)
        ↓
    BridgeAgent        (元件连接: 存在性审查、端口规划、跨域桥接)
        ↓
    ParameterAgent     (参数设置: 查询参数范围、物理推断赋值)
        ↓
    BuilderAgent       (模型生成: 脚本生成、auto/manual模式)
        ↓
    DiagnosisAgent     (错误诊断: 输出解析、错误分类、路由建议)

容错机制:
    - 程序化编排，不依赖LLM handoff
    - 3次重试阈值，同一Agent连续失败触发总管重新规划
    - errors必须处理，warnings仅报告

用法:
    uv run python -m amesim_multi_agent.orchestrator_runner "构建一个质量-弹簧-阻尼系统"
    或 (兼容旧入口)
    uv run python -m amesim_multi_agent.runner "构建一个质量-弹簧-阻尼系统"
"""


def _lazy_import():
    """延迟导入，避免 python -m 运行时的模块顺序 Warning。"""
    from .agents.orchestrator import create_orchestrator_agent
    from .agents.selector import create_selector_agent
    from .agents.bridge import create_bridge_agent
    from .agents.parameter import create_parameter_agent
    from .agents.builder import create_builder_agent
    from .agents.diagnosis import create_diagnosis_agent
    from .orchestrator_runner import main as run
    return (
        create_orchestrator_agent, create_selector_agent,
        create_bridge_agent, create_parameter_agent,
        create_builder_agent, create_diagnosis_agent, run,
    )


def __getattr__(name):
    agent_names = (
        "create_orchestrator_agent", "create_selector_agent",
        "create_bridge_agent", "create_parameter_agent",
        "create_builder_agent", "create_diagnosis_agent", "run",
    )
    if name in agent_names:
        result = _lazy_import()
        mapping = dict(zip(agent_names, result))
        return mapping[name]
    raise AttributeError(f"module 'amesim_multi_agent' has no attribute '{name}'")
