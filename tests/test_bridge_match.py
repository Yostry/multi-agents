"""TDD tests for batch port matching (RED phase).

Tests for the refactored Bridge agent port matching system that replaces
the current scoring system (same_domain +10, io_match +5) with a batch LLM
review that reads port norm data directly.

ALL tests are xfail — the feature does not exist yet.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "amesim_multi_agent")
)

from amesim_multi_agent.agents.bridge_core import BridgePlanResult

import pytest


# ============================================================
# Shared Fixtures
# ============================================================


def _make_port(port_index, port_tag, variables):
    """Helper to create a port dict with canonical structure."""
    return {
        "port_index": port_index,
        "port_tag": port_tag,
        "variables": variables,
    }


def _make_var(norm, io, title="", units=""):
    """Helper to create a variable dict."""
    return {"norm": norm, "io": io, "title": title, "units": units}


# --- Component fixture builders ---


@pytest.fixture
def mechanical_components():
    """Two mechanical-translation components with compatible pressure/force ports."""
    return [
        {
            "alias": "Mass",
            "icon_key": "libmec:mass_friction_endstops",
            "icon_name": "mass_friction_endstops",
            "library": "libmec",
            "submodel_id": "MAS001",
            "ports": [
                _make_port(
                    0,
                    "mec_trans",
                    [
                        _make_var("force", "1", "force at port 1", "N"),
                        _make_var("velocity", "2", "velocity at port 1", "m/s"),
                        _make_var("displacement", "1", "displacement at port 1", "m"),
                    ],
                ),
                _make_port(
                    1,
                    "mec_trans",
                    [
                        _make_var("force", "2", "force at port 2", "N"),
                        _make_var("velocity", "1", "velocity at port 2", "m/s"),
                        _make_var("displacement", "2", "displacement at port 2", "m"),
                    ],
                ),
            ],
        },
        {
            "alias": "Spring",
            "icon_key": "libmec:spring01",
            "icon_name": "spring01",
            "library": "libmec",
            "submodel_id": "SPR001",
            "ports": [
                _make_port(
                    0,
                    "mec_trans",
                    [
                        _make_var("force", "2", "force at port 1", "N"),
                        _make_var("velocity", "2", "velocity at port 1", "m/s"),
                        _make_var("displacement", "2", "displacement at port 1", "m"),
                    ],
                ),
                _make_port(
                    1,
                    "mec_trans",
                    [
                        _make_var("force", "1", "force at port 2", "N"),
                        _make_var("velocity", "1", "velocity at port 2", "m/s"),
                        _make_var("displacement", "1", "displacement at port 2", "m"),
                    ],
                ),
            ],
        },
    ]


@pytest.fixture
def cross_domain_components():
    """Thermal output to two_phase_flow input — needs a bridge."""
    return [
        {
            "alias": "HeatSource",
            "icon_key": "libth:heat_source",
            "icon_name": "heat_source",
            "library": "libth",
            "submodel_id": "THS001",
            "ports": [
                _make_port(
                    0,
                    "thermal",
                    [
                        _make_var("temperature", "1", "temperature", "K"),
                        _make_var("heat_flow_rate", "1", "heat flow rate", "W"),
                    ],
                ),
            ],
        },
        {
            "alias": "Tank",
            "icon_key": "libtpf:tank_two_phase",
            "icon_name": "tank_two_phase",
            "library": "libtpf",
            "submodel_id": "TPF001",
            "ports": [
                _make_port(
                    0,
                    "two_phase_flow",
                    [
                        _make_var("pressure", "2", "pressure at port 1", "barA"),
                        _make_var("temperature", "2", "temperature at port 1", "K"),
                        _make_var("mass_flow_rate", "2", "mass flow rate at port 1", "kg/s"),
                    ],
                ),
            ],
        },
    ]


@pytest.fixture
def multi_port_pump_and_valve():
    """Pump (3 ports) + Valve (4 ports) — no port should be used twice."""
    return [
        {
            "alias": "Pump",
            "icon_key": "libhydr:pump01",
            "icon_name": "pump01",
            "library": "libhydr",
            "submodel_id": "PU001",
            "ports": [
                _make_port(0, "hydraulic", [
                    _make_var("pressure", "2", "pressure at suction", "bar"),
                    _make_var("flow_rate", "1", "flow rate at suction", "L/min"),
                ]),
                _make_port(1, "hydraulic", [
                    _make_var("pressure", "1", "pressure at discharge", "bar"),
                    _make_var("flow_rate", "2", "flow rate at discharge", "L/min"),
                ]),
                _make_port(2, "hydraulic", [
                    _make_var("pressure", "1", "pressure at drain", "bar"),
                    _make_var("flow_rate", "2", "flow rate at drain", "L/min"),
                ]),
            ],
        },
        {
            "alias": "Valve",
            "icon_key": "libhydr:valve4port",
            "icon_name": "valve4port",
            "library": "libhydr",
            "submodel_id": "VAL001",
            "ports": [
                _make_port(0, "hydraulic", [
                    _make_var("pressure", "1", "pressure at P", "bar"),
                    _make_var("flow_rate", "2", "flow rate at P", "L/min"),
                ]),
                _make_port(1, "hydraulic", [
                    _make_var("pressure", "2", "pressure at T", "bar"),
                    _make_var("flow_rate", "1", "flow rate at T", "L/min"),
                ]),
                _make_port(2, "hydraulic", [
                    _make_var("pressure", "1", "pressure at A", "bar"),
                    _make_var("flow_rate", "2", "flow rate at A", "L/min"),
                ]),
                _make_port(3, "hydraulic", [
                    _make_var("pressure", "2", "pressure at B", "bar"),
                    _make_var("flow_rate", "1", "flow rate at B", "L/min"),
                ]),
            ],
        },
    ]


@pytest.fixture
def branch_topology_components():
    """One pump output branching to two valves — needs junction/splitter."""
    return [
        {
            "alias": "Pump",
            "icon_key": "libhydr:pump01",
            "icon_name": "pump01",
            "library": "libhydr",
            "submodel_id": "PU001",
            "ports": [
                _make_port(0, "hydraulic", [
                    _make_var("pressure", "2", "pressure at suction", "bar"),
                    _make_var("flow_rate", "1", "flow rate at suction", "L/min"),
                ]),
                _make_port(1, "hydraulic", [
                    _make_var("pressure", "1", "pressure at discharge", "bar"),
                    _make_var("flow_rate", "2", "flow rate at discharge", "L/min"),
                ]),
            ],
        },
        {
            "alias": "ValveA",
            "icon_key": "libhydr:valve01",
            "icon_name": "valve01",
            "library": "libhydr",
            "submodel_id": "VAL02",
            "ports": [
                _make_port(0, "hydraulic", [
                    _make_var("pressure", "1", "pressure at P", "bar"),
                    _make_var("flow_rate", "2", "flow rate at P", "L/min"),
                ]),
                _make_port(1, "hydraulic", [
                    _make_var("pressure", "2", "pressure at T", "bar"),
                    _make_var("flow_rate", "1", "flow rate at T", "L/min"),
                ]),
            ],
        },
        {
            "alias": "ValveB",
            "icon_key": "libhydr:valve01",
            "icon_name": "valve01",
            "library": "libhydr",
            "submodel_id": "VAL03",
            "ports": [
                _make_port(0, "hydraulic", [
                    _make_var("pressure", "1", "pressure at P", "bar"),
                    _make_var("flow_rate", "2", "flow rate at P", "L/min"),
                ]),
                _make_port(1, "hydraulic", [
                    _make_var("pressure", "2", "pressure at T", "bar"),
                    _make_var("flow_rate", "1", "flow rate at T", "L/min"),
                ]),
            ],
        },
    ]


@pytest.fixture
def component_with_missing_norm():
    """Component whose port has no variables — should degrade gracefully."""
    return [
        {
            "alias": "MysteryComponent",
            "icon_key": "libsig:unknown_signal",
            "icon_name": "unknown_signal",
            "library": "libsig",
            "submodel_id": "SIG999",
            "ports": [
                _make_port(0, "signal", []),
                _make_port(1, "signal", []),
            ],
        },
        {
            "alias": "NormalSensor",
            "icon_key": "libsig:sensor01",
            "icon_name": "sensor01",
            "library": "libsig",
            "submodel_id": "SEN001",
            "ports": [
                _make_port(0, "signal", [
                    _make_var("signal", "1", "output signal", "null"),
                ]),
            ],
        },
    ]


# ============================================================
# Helper to extract port indices from connections
# ============================================================


def _collect_used_ports(connections):
    """Return {(alias, port_index), ...} for all connections."""
    used = set()
    for conn in connections:
        used.add((conn["from_alias"], conn["from_port"]))
        used.add((conn["to_alias"], conn["to_port"]))
    return used


# ============================================================
# Test 1: Same-domain direct match
# ============================================================


@pytest.mark.xfail(reason="Awaiting batch review prompt implementation")
def test_same_domain_direct_match(mechanical_components):
    """Two mec_trans components: one output (io=1) pressure port matches
    the other's input (io=2) pressure port. Should identify the correct
    port indices without guessing."""
    from amesim_multi_agent.agents.bridge_core import batch_match_ports

    topology_pairs = [("Mass", "Spring")]

    result = batch_match_ports(mechanical_components, topology_pairs)

    # Result should contain at least one connection
    assert result is not None
    connections = result.get("connections", [])
    assert len(connections) >= 1

    # The connection must reference the correct port indices
    conn = connections[0]
    assert conn["from_alias"] == "Mass"
    assert conn["to_alias"] == "Spring"
    assert conn["type"] == "line"

    # Verify domain match: both ports should have same port_tag
    assert conn.get("port_type_from") == "mec_trans"
    assert conn.get("port_type_to") == "mec_trans"

    # The from_port should be Mass-port-0 (io=1 / output),
    # the to_port should be Spring-port-1 (io=1 / output)
    # Wait — Spring port 0 is io=2 (input). A proper match is
    # Mass-port-0 (output force) → Spring-port-0 (input force).
    assert conn["from_port"] == 0
    assert conn["to_port"] == 0
    assert conn.get("line_alias", "").startswith("wire_")


# ============================================================
# Test 2: Cross-domain bridge needed
# ============================================================


@pytest.mark.xfail(reason="Awaiting batch review prompt implementation")
def test_cross_domain_bridge_needed(cross_domain_components):
    """Thermal output port (port_tag=thermal) should connect to
    two_phase_flow input port — different port_tags mean a bridge
    component must be inserted."""
    from amesim_multi_agent.agents.bridge_core import batch_match_ports

    topology_pairs = [("HeatSource", "Tank")]

    result = batch_match_ports(cross_domain_components, topology_pairs)

    assert result is not None
    bridge_components = result.get("bridge_components", [])
    connections = result.get("connections", [])

    # A bridge component must be proposed
    assert len(bridge_components) >= 1, (
        "Expected at least 1 bridge component for thermal→two_phase_flow"
    )

    bridge = bridge_components[0]
    assert "icon_key" in bridge
    assert "alias" in bridge

    # Connections should link through the bridge: from→bridge + bridge→to
    assert len(connections) >= 2

    # The two source components' port_tags must differ (proving bridge is needed)
    heat_port_tag = "thermal"
    tank_port_tag = "two_phase_flow"
    assert heat_port_tag != tank_port_tag


# ============================================================
# Test 3: Multi-port no duplicate usage
# ============================================================


@pytest.mark.xfail(reason="Awaiting batch review prompt implementation")
def test_multi_port_no_duplicate(multi_port_pump_and_valve):
    """A pump with 3 ports and a valve with 4 ports — no port_index
    should appear in more than one connection."""
    from amesim_multi_agent.agents.bridge_core import batch_match_ports

    topology_pairs = [("Pump", "Valve")]

    result = batch_match_ports(multi_port_pump_and_valve, topology_pairs)
    assert result is not None
    connections = result.get("connections", [])

    used = _collect_used_ports(connections)

    # Every (alias, port_index) pair must appear at most once
    assert len(used) == 2 * len(connections), (
        f"Duplicate port usage detected: {used} from {connections}"
    )

    # All from/to aliases must be among known components
    known = {"Pump", "Valve"}
    for conn in connections:
        assert conn["from_alias"] in known
        assert conn["to_alias"] in known


# ============================================================
# Test 4: Branch topology needs junction
# ============================================================


@pytest.mark.xfail(reason="Awaiting batch review prompt implementation")
def test_branch_topology_junction(branch_topology_components):
    """One pump output connecting to two valves — needs a junction
    or splitter component to avoid connecting one output to
    two inputs directly."""
    from amesim_multi_agent.agents.bridge_core import batch_match_ports

    # Pump's discharge → both ValveA and ValveB
    topology_pairs = [("Pump", "ValveA"), ("Pump", "ValveB")]

    result = batch_match_ports(branch_topology_components, topology_pairs)
    assert result is not None

    connections = result.get("connections", [])
    bridge_components = result.get("bridge_components", [])

    # Should propose at least one junction/splitter
    assert len(bridge_components) >= 1, (
        "Expected junction/splitter for branch topology"
    )

    junction = bridge_components[0]
    # Junction alias should appear in the connections
    junction_alias = junction.get("alias", "")
    assert junction_alias, "Junction must have an alias"

    junction_seen = any(
        conn["from_alias"] == junction_alias or conn["to_alias"] == junction_alias
        for conn in connections
    )
    assert junction_seen, (
        f"Junction '{junction_alias}' not referenced in any connection"
    )

    # The pump's discharge port (index 1) should only appear once
    pump_discharge_count = sum(
        1 for c in connections
        if c["from_alias"] == "Pump" and c["from_port"] == 1
    )
    assert pump_discharge_count <= 1, (
        "Pump discharge port used more than once without junction"
    )


# ============================================================
# Test 5: Missing norm — graceful degradation
# ============================================================


@pytest.mark.xfail(reason="Awaiting batch review prompt implementation")
def test_missing_norm_graceful_degradation(component_with_missing_norm):
    """A component port has no variables data. The matcher must not crash
    and should produce a result (possibly with a validation issue)."""
    from amesim_multi_agent.agents.bridge_core import batch_match_ports

    topology_pairs = [("MysteryComponent", "NormalSensor")]

    # Must not raise
    result = batch_match_ports(component_with_missing_norm, topology_pairs)

    assert result is not None

    # Should still return a valid structure even if matching is incomplete
    assert "connections" in result
    assert isinstance(result["connections"], list)

    # If a validation_issues field exists, it may flag the missing data
    if "validation_issues" in result:
        assert isinstance(result["validation_issues"], list)


# ============================================================
# Test 6: Empty input — no crash
# ============================================================


@pytest.mark.xfail(reason="Awaiting batch review prompt implementation")
def test_empty_input_no_crash():
    """Empty component_ports and topology_pairs must return gracefully."""
    from amesim_multi_agent.agents.bridge_core import batch_match_ports

    result = batch_match_ports([], [])

    assert result is not None
    assert isinstance(result.get("connections", []), list)
    assert len(result.get("connections", [])) == 0

    # Bridge components should also be empty
    assert isinstance(result.get("bridge_components", []), list)
    assert len(result.get("bridge_components", [])) == 0
