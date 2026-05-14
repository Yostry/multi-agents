"""Test script for the new Selector architecture pipeline.
Run: .venv\Scripts\activate && python test_new_selector.py
"""
import sys, asyncio, json
sys.path.insert(0, ".")

from amesim_multi_agent.selector_integration import run_new_selector_pipeline
from amesim_multi_agent.orchestrator_runner import PipelineState, create_kimi_run_config
from amesim_multi_agent.pipeline_context import PipelineContext
from amesim_multi_agent.protocols import TopologyPlan

async def test_hydraulic_system():
    """Test the new Selector pipeline with a simple hydraulic system."""
    state = PipelineState(user_request="Simple hydraulic system with pump and valve")
    run_config = create_kimi_run_config()
    ctx = PipelineContext("Test_Hydraulic_Selector")

    # Mock a topology plan
    from amesim_multi_agent.protocols import ComponentRequirement
    state.topology_plan = TopologyPlan(
        summary="Simple hydraulic circuit: pump -> valve -> tank",
        subsystems=[],
        component_requirements=[
            ComponentRequirement(
                index=1, keywords=["hydraulic pump", "fixed displacement"],
                functional_description="Fixed displacement hydraulic pump",
                suggested_library="libhydr", physical_domain="hydraulic",
                port_requirements=[{"quantity": "flow_rate", "direction": "output", "count": 1}]
            ),
            ComponentRequirement(
                index=2, keywords=["pressure control valve"],
                functional_description="Pressure reducing valve",
                suggested_library="libhydr", physical_domain="hydraulic",
                port_requirements=[{"quantity": "pressure", "direction": "input", "count": 2}]
            ),
        ],
        connectivity_description="Pump -> Valve"
    )

    print("=" * 60)
    print("  TESTING NEW SELECTOR PIPELINE")
    print("=" * 60)

    next_stage = await run_new_selector_pipeline(state, run_config, ctx, verbose=True)

    print(f"\nPipeline result: {next_stage}")
    if state.component_selection:
        print(f"Selected {len(state.component_selection.selected)} components:")
        for s in state.component_selection.selected:
            print(f"  - {s.alias}: {s.icon_key} (submodel: {s.recommended_submodel}, confidence: {s.confidence})")
        if state.component_selection.failures:
            print(f"  Failures: {len(state.component_selection.failures)}")

    # Verify validation
    from amesim_multi_agent.tools.validator_tools import validate_all_selections
    errors = validate_all_selections(state.component_selection.selected)
    if errors:
        print(f"\nVALIDATION ERRORS: {len(errors)}")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\nVALIDATION: All components passed!")

if __name__ == "__main__":
    asyncio.run(test_hydraulic_system())
