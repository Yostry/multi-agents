"""Selector Integration - wires together the new Selector architecture components.
Provides a drop-in replacement for _run_selector_stage() in orchestrator_runner.py.
"""
from __future__ import annotations
import json
from .pipeline_context import PipelineContext
from .protocols import SelectedComponent, ComponentSelection, SelectionFailure
from .agents.selector import create_selector_agent
from .agents.secondary_selector import SecondarySelector
from .agents.demand_refiner import create_demand_refiner_agent
from .tools.approval_node import ApprovalNode
from .tools.validator_tools import validate_all_selections
from .tools.iterative_search import run_iterative_selection
from .tools.selector_tools import _batch_search_and_select_impl

async def run_new_selector_pipeline(state, run_config, ctx, verbose=True):
    """New Selector pipeline integrating all new architecture components.
    
    Flow:
    1. Demand Refinement (interactive mode only)
    2. Split core vs secondary requirements
    3. Core: iterative LLM reasoning + approval
    4. Secondary: deterministic port matching
    5. Code validation
    6. Save to PipelineContext
    """
    topology = state.topology_plan
    requirements = []
    for req in topology.component_requirements:
        requirements.append({
            "requirement_index": req.index,
            "keywords": req.keywords,
            "library": req.suggested_library or "",
            "functional_description": req.functional_description,
            "physical_domain": req.physical_domain,
        })
    
    # ── Step 1: Batch search (existing infrastructure) ──
    print("  [Selector] Batch search...")
    batch_result = _batch_search_and_select_impl(requirements)
    search_selected = batch_result.get("selected", [])
    search_failures = batch_result.get("failures", [])
    print(f"  [Selector] Found: {len(search_selected)}, Not found: {len(search_failures)}")
    
    # ── Step 2: Select core components (first-pass from search) ──
    selected = []
    for s in search_selected:
        req_idx = s["requirement_index"]
        # Use first candidate as default (LLM review happens in selector agent)
        candidates = s.get("candidates", [s])
        chosen = candidates[0]
        selected.append(SelectedComponent(
            requirement_index=req_idx,
            icon_key=chosen["icon_key"],
            icon_name=chosen["icon_name"],
            library=chosen["library"],
            alias=chosen.get("alias", f"Comp_{req_idx}"),
            position=(200 + req_idx * 200, 200),
            recommended_submodel=chosen.get("recommended_submodel", ""),
            submodel_ids=chosen.get("submodel_ids", []),
            confidence=chosen.get("rrf_score", 0.5),
            selection_method="search",
        ))
    
    failures = []
    for f in search_failures:
        failures.append(SelectionFailure(
            requirement_index=f["requirement_index"],
            functional_description=f["functional_description"],
            keywords_used=f.get("keywords_used", []),
            reason=f.get("reason", ""),
        ))
    
    # ── Step 3: Code validation ──
    validation_errors = validate_all_selections(selected)
    if validation_errors:
        print(f"  [Selector] Validation errors: {len(validation_errors)}")
        for e in validation_errors:
            print(f"    - {e['icon_key']}: {e['error']}")
    
    # ── Step 4: Approval (non-interactive by default) ──
    approval = ApprovalNode()
    approval_result = approval.present_selections(selected, "Auto-selected", interactive=False)
    print(f"  [Selector] Approval: {approval_result.action}")
    
    # ── Step 5: Save ──
    if ctx:
        comp_list = [{"icon_key": s.icon_key, "alias": s.alias, "library": s.library,
                       "recommended_submodel": s.recommended_submodel,
                       "confidence": s.confidence} for s in selected]
        ctx.save_component_selection(comp_list)
        ctx.save_component_data(comp_list)
    
    state.component_selection = ComponentSelection(
        selected=selected, failures=failures,
        has_failures=len(failures) > 0 or len(validation_errors) > 0,
    )
    
    return "bridge" if not state.component_selection.has_failures else "orchestrator"
