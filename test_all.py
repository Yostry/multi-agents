"""
Comprehensive test suite for tri_agent modules.
Run: uv run python test_all.py
"""
import sys, traceback

tests_run = 0
tests_pass = 0
tests_fail = 0

def test(name):
    global tests_run
    tests_run += 1
    print(f'  [{tests_run}] {name}...', end=' ', flush=True)

def ok():
    global tests_pass
    tests_pass += 1
    print('PASS')

def bad(e):
    global tests_fail
    tests_fail += 1
    print(f'FAIL: {e}')

# === Module 1: topology_graph ===
test('import topology_graph')
try:
    from amesim_multi_agent.tri_agent.topology_graph import (
        TopologyGraph, TopologyNode, TopologyEdge, Subsystem,
        ComponentEntry, ConnectionEntry, TorsionBarModel
    )
    ok()
except Exception as e:
    bad(e)

test('TopologyNode creation')
try:
    n = TopologyNode(node_id='n1', label='Test_Tank', role='thermal_storage',
                     domain='two_phase_flow', is_core=True, topological_role='storage')
    assert n.is_core == True
    assert n.to_dict()['node_id'] == 'n1'
    ok()
except Exception as e:
    bad(e)

test('TopologyNode from_dict')
try:
    n2 = TopologyNode.from_dict(n.to_dict())
    assert n2.label == 'Test_Tank'
    ok()
except Exception as e:
    bad(e)

test('TopologyEdge creation')
try:
    e = TopologyEdge(edge_id='e1', source_node_id='n1', target_node_id='n2',
                     flow_type='mass_flow', port_match='hflow->hflow')
    assert e.is_cross_domain == False
    ok()
except Exception as e:
    bad(e)

test('TopologyGraph core methods')
try:
    g = TopologyGraph(model_name='Test', user_request='test', physical_domains=['two_phase_flow'])
    n1 = TopologyNode(node_id='n1', label='Core', is_core=True, domain='two_phase_flow')
    n2 = TopologyNode(node_id='n2', label='Secondary', is_core=False, domain='two_phase_flow')
    g.nodes = [n1, n2]
    g.edges = [TopologyEdge(edge_id='e1', source_node_id='n1', target_node_id='n2')]
    assert len(g.core_nodes) == 1
    assert len(g.non_core_nodes) == 1
    assert g.complexity == 'basic'
    assert g.component_count == 2
    assert g.get_node('n1') is not None
    assert g.get_node('n99') is None
    assert len(g.get_edges_from('n1')) == 1
    assert len(g.get_edges_to('n2')) == 1
    assert g.is_orphan('n3') == True
    assert g.is_orphan('n1') == False
    ok()
except Exception as e:
    bad(e)

test('TopologyGraph JSON roundtrip')
try:
    j = g.to_json()
    g2 = TopologyGraph.from_json(j)
    assert g2.model_name == 'Test'
    assert len(g2.nodes) == 2
    assert len(g2.edges) == 1
    ok()
except Exception as e:
    bad(e)

test('TopologyGraph Mermaid')
try:
    m = g.to_mermaid()
    assert 'graph LR' in m
    assert 'n1' in m
    ok()
except Exception as e:
    bad(e)

test('TopologyGraph summary')
try:
    s = g.summary()
    assert 'Test' in s
    assert '1 core' in s
    ok()
except Exception as e:
    bad(e)

test('Subsystem creation')
try:
    sub = Subsystem(subsys_id='sub1', name='TestSub', domain='thermal')
    assert sub.to_dict()['subsys_id'] == 'sub1'
    ok()
except Exception as e:
    bad(e)

test('ComponentEntry + ConnectionEntry')
try:
    c = ComponentEntry(icon_name='pump01', alias='Pump', position=(200, 200),
                       submodel='PU001', submodel_path='$AME/libhydr/submodels')
    conn = ConnectionEntry(from_component='A', from_port=0, to_component='B', to_port=1, type='line')
    assert c.icon_name == 'pump01'
    assert conn.from_port == 0
    ok()
except Exception as e:
    bad(e)

test('TorsionBarModel complete flow')
try:
    tb = TorsionBarModel(model_name='HydSys')
    tb.components = [ComponentEntry(icon_name='pump01', alias='Pump', position=(200,200), submodel='PU001')]
    tb.connections = [ConnectionEntry(from_component='Pump', from_port=0, to_component='Tank', to_port=0)]
    tb.non_default_params = {'flow@Pump': '10.0'}
    assert tb.is_complete()
    d = tb.to_dict()
    assert d['circuit']['components'][0]['icon_name'] == 'pump01'
    assert len(d['circuit']['connections']) == 1
    assert d['non_default_params']['flow@Pump'] == '10.0'
    j2 = tb.to_json()
    tb2 = TorsionBarModel.from_json(j2)
    assert tb2.model_name == 'HydSys'
    assert len(tb2.components) == 1
    assert len(tb2.connections) == 1
    assert tb2.non_default_params['flow@Pump'] == '10.0'
    assert 'HydSys' in tb.summary()
    ok()
except Exception as e:
    bad(e)

test('TopologyNode defaults')
try:
    n_def = TopologyNode()
    assert n_def.extracted_attrs == {}
    assert n_def.constraints == []
    assert n_def.unresolved == []
    ok()
except Exception as e:
    bad(e)

# === Module 2: experience_memory ===
test('ExperienceMemory init')
try:
    from amesim_multi_agent.tri_agent.experience_memory import ExperienceMemory, _row_to_success_case, _row_to_error_pattern
    mem = ExperienceMemory(':memory:')
    mem.initialize()
    ok()
except Exception as e:
    bad(e)

test('ExperienceMemory record + query')
try:
    cid = mem.record_success({
        'model_name': 'TestModel', 'user_request': 'build test system',
        'topology_graph_json': '{}', 'torsionbar_json': '{}',
        'component_count': 3, 'physical_domains': ['hydraulic'],
        'total_llm_calls': 2, 'tags': ['basic'],
    })
    s = mem.stats()
    assert s['success_cases'] == 1
    cases = mem.query_success_by_domain('hydraulic', top_k=5)
    assert len(cases) >= 1
    assert cases[0].model_name == 'TestModel'
    ok()
except Exception as e:
    bad(e)

test('ExperienceMemory error patterns')
try:
    mem.upsert_error_pattern('CAUSALITY_CONFLICT', 'connector', 'R-C violation at valve', 'Add C element between')
    mem.upsert_error_pattern('CAUSALITY_CONFLICT', 'connector', 'R-C violation at valve', 'Add C element between')
    patterns = mem.query_error_patterns(stage='connector')
    assert len(patterns) == 1
    assert patterns[0].occurrence_count == 2
    mem.upsert_error_pattern('PORT_MISMATCH', 'orchestrator', 'port tag mismatch', 'Use bridge')
    assert len(mem.query_error_patterns(stage='orchestrator')) == 1
    ok()
except Exception as e:
    bad(e)

test('ExperienceMemory prompt versions')
try:
    mem.record_prompt_version('orchestrator', 'v1.0', 'Original prompt', 0.85)
    mem.record_prompt_version('orchestrator', 'v1.1', 'Improved prompt', 0.92)
    active = mem.get_active_prompt('orchestrator')
    assert active['version'] == 'v1.1'
    assert active['performance_score'] == 0.92
    assert mem.get_active_prompt('nonexistent') is None
    ok()
except Exception as e:
    bad(e)

test('ExperienceMemory self-play + full stats')
try:
    scid = mem.record_self_play_case({
        'source_case_id': 1, 'mutation_type': 'domain_swap',
        'mutated_request': 'build pneumatic system',
        'expected_topology_json': {}, 'actual_topology_json': {},
        'result_status': 'success', 'metrics_json': {'accuracy': 0.9},
    })
    assert scid > 0
    s = mem.stats()
    assert s['success_cases'] == 1
    assert s['error_patterns'] == 2
    assert s['prompt_versions'] == 2
    assert s['self_play_cases'] == 1
    mem.close()
    ok()
except Exception as e:
    bad(e)

# === Module 3: Prompts ===
test('import prompts')
try:
    from amesim_multi_agent.tri_agent.prompts import (
        ORCHESTRATOR_SYSTEM_PROMPT, CONNECTOR_PHASE1_SYSTEM_PROMPT,
        CONNECTOR_PHASE2_SYSTEM_PROMPT, PARAMETER_RUNNER_SYSTEM_PROMPT,
    )
    assert len(ORCHESTRATOR_SYSTEM_PROMPT) > 500
    assert len(CONNECTOR_PHASE1_SYSTEM_PROMPT) > 500
    assert len(CONNECTOR_PHASE2_SYSTEM_PROMPT) > 500
    assert len(PARAMETER_RUNNER_SYSTEM_PROMPT) > 500
    ok()
except Exception as e:
    bad(e)

# === Module 4: Agent factories ===
test('agent factories + creation')
try:
    from amesim_multi_agent.tri_agent.agents import (
        create_tri_orchestrator_agent, create_orchestrator_with_tools,
        create_connector_phase1_agent, create_connector_phase2_agent,
        create_connector_phase1_with_tools, create_connector_phase2_with_tools,
        create_parameter_runner_agent, create_parameter_runner_with_tools,
    )
    o = create_tri_orchestrator_agent()
    assert o.name == 'Orchestrator'
    c1 = create_connector_phase1_agent()
    assert c1.name == 'Connector_Phase1'
    c2 = create_connector_phase2_agent()
    assert c2.name == 'Connector_Phase2'
    pr = create_parameter_runner_agent()
    assert pr.name == 'ParameterRunner'
    ok()
except Exception as e:
    bad(e)

test('Agent with tools factory signatures')
try:
    # Create mock functions
    def mock_search(): pass
    def mock_exp(): pass
    def mock_kb(): pass
    def mock_ask(): pass
    o2 = create_orchestrator_with_tools(mock_search, mock_exp, mock_kb, mock_ask)
    assert len(o2.tools) == 4

    def bsr(): pass
    def sc(): pass
    def gcd(): pass
    def rs(): pass
    c1w = create_connector_phase1_with_tools(bsr, sc, gcd, rs)
    assert len(c1w.tools) == 4

    def bqp(): pass
    def vac(): pass
    def ddp(): pass
    def fpc(): pass
    def fbp(): pass
    def fjc(): pass
    def vcp(): pass
    def fdtc(): pass
    c2w = create_connector_phase2_with_tools(bqp, vac, ddp, fpc, fbp, fjc, vcp, fdtc, gcd)
    assert len(c2w.tools) == 9

    def barm(): pass
    def rbs(): pass
    def de(): pass
    def ce(): pass
    def pbo(): pass
    def qsp(): pass
    def bqp2(): pass
    prw = create_parameter_runner_with_tools(bqp2, qsp, barm, rbs, de, ce, pbo)
    assert len(prw.tools) == 7
    ok()
except Exception as e:
    bad(e)

# === Module 5: Pipeline ===
test('pipeline imports + state')
try:
    from amesim_multi_agent.tri_agent.pipeline import (
        TriAgentPipeline, PipelineState,
        MAX_VALIDATION_ITERATIONS, MAX_EXPANSION_LAYERS,
    )
    ps = PipelineState(user_request='test', skip_review=True)
    assert ps.topology_graph is None
    assert ps.is_topology_ready == False
    ps.topology_graph = TopologyGraph(model_name='T', nodes=[TopologyNode(node_id='n1')])
    assert ps.is_topology_ready == True
    assert MAX_VALIDATION_ITERATIONS == 3
    assert MAX_EXPANSION_LAYERS == 4
    ok()
except Exception as e:
    bad(e)

test('Pipeline init + json extraction')
try:
    pl = TriAgentPipeline(skip_review=True, verbose=False, experience_db_path=':memory:')
    assert pl.state.skip_review == True
    assert pl.memory is not None
    pl.memory.close()

    # Test JSON extraction
    result = TriAgentPipeline._extract_json('{"a": 1}')
    assert result == {'a': 1}
    result2 = TriAgentPipeline._extract_json('```json\n{"b": 2}\n```')
    assert result2 == {'b': 2}
    result3 = TriAgentPipeline._extract_json('prefix {"c": 3} suffix')
    assert result3 == {'c': 3}

    # Test invalid JSON
    try:
        TriAgentPipeline._extract_json('not json')
        assert False, 'should have raised'
    except ValueError:
        pass
    ok()
except Exception as e:
    bad(e)

test('Pipeline validation logic')
try:
    pl = TriAgentPipeline(skip_review=True, verbose=False)
    pl.state.topology_graph = TopologyGraph(
        model_name='Test', nodes=[
            TopologyNode(node_id='n1', label='Core_Tank', is_core=True, domain='two_phase_flow'),
            TopologyNode(node_id='n2', label='Secondary_Valve', is_core=False, domain='two_phase_flow'),
        ]
    )
    pl.state.torsionbar_model = TorsionBarModel(model_name='Test')
    pl.state.torsionbar_model.components = [
        ComponentEntry(icon_name='tpf_chamber_heat', alias='Core_Tank', position=(200,200), submodel='TPFHECH000', layer=0),
        ComponentEntry(icon_name='tpf_orifice', alias='Secondary_Valve', position=(400,200), submodel='TPFGR00', layer=1),
    ]
    pl.state.torsionbar_model.connections = [
        ConnectionEntry(from_component='Core_Tank', from_port=2, to_component='Secondary_Valve', to_port=0),
    ]
    issues = pl._validate_model()
    # Core_Tank matches via alias, Secondary_Valve has connection
    assert len(issues) == 0, f'expected 0 issues, got {issues}'

    # Test: isolated non-core component
    pl.state.torsionbar_model.components.append(
        ComponentEntry(icon_name='extra', alias='Orphan_Comp', position=(600,200), layer=1)
    )
    issues2 = pl._validate_model()
    assert len(issues2) == 1
    assert issues2[0]['type'] == 'ISOLATED_COMPONENT'

    ok()
except Exception as e:
    bad(e)

# === Module 6: self_play_engine ===
test('Self-play engine + mutations')
try:
    from amesim_multi_agent.tri_agent.self_play_engine import SelfPlayEngine, MUTATION_STRATEGIES
    assert len(MUTATION_STRATEGIES) == 6
    assert 'domain_swap' in MUTATION_STRATEGIES
    assert 'combined' in MUTATION_STRATEGIES

    engine = SelfPlayEngine(None)
    original = 'Build a hydraulic pump system with 10 bar pressure and 100 L/min flow'
    for strategy in ['domain_swap', 'scale_change', 'topology_add', 'topology_remove', 'requirement_rewrite', 'combined']:
        mutated = engine._mutate_request(original, strategy)
        assert isinstance(mutated, str) and len(mutated) > 0, f'{strategy} returned empty'
    ok()
except Exception as e:
    bad(e)

# === Module 7: role_vocabulary ===
test('role vocab classifier')
try:
    from amesim_multi_agent.tri_agent.role_vocabulary.build_vocab import (
        _classify_role, _classify_domain
    )
    tests_role = [
        ('constant volume chamber with heat exchange', 'tpf_chamber_heat', 'storage'),
        ('generic restriction', 'tpf_orifice', 'transfer'),
        ('pressure sensor', 'tpf_sensor', 'sensor'),
        ('mass with friction and endstops', 'mass2port', 'storage'),
        ('constant signal source', 'constant', 'control'),
        ('hydraulic pump', 'pump01', 'actuator'),
        ('heat exchanger radiator cooling', 'heat_exchanger', 'transfer'),
    ]
    for label, icon, expected in tests_role:
        r = _classify_role(label, '', icon)
        assert r == expected, f'{icon}: expected {expected}, got {r}'
    ok()
except Exception as e:
    bad(e)

test('domain classifier')
try:
    tests_domain = [
        ('libtpf', 'two_phase_flow'), ('libmec', 'mechanical_1d'),
        ('libsig', 'signal'), ('libth', 'thermal'), ('libhydr', 'hydraulic'),
        ('libpn', 'pneumatic'), ('libeb', 'electric'), ('libthh', 'thermal_hydraulic'),
        ('libcs', 'cooling'), ('libunknown', ''),
    ]
    for lib, expected in tests_domain:
        assert _classify_domain(lib) == expected, f'{lib}: expected {expected}, got {_classify_domain(lib)}'
    ok()
except Exception as e:
    bad(e)

# === Summary ===
print(f'\n{"="*50}')
print(f'  Results: {tests_pass}/{tests_run} passed, {tests_fail} failed')
if tests_fail == 0:
    print('  ALL TESTS PASSED')
else:
    print(f'  {tests_fail} FAILURES FOUND')
print(f'{"="*50}')
sys.exit(0 if tests_fail == 0 else 1)
