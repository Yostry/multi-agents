"""Regression tests for bridge_core functions that must not change during refactoring.

These functions are the stable core of the Bridge agent. plan_single_connection()
and format_connection_review_prompt() will be removed/replaced, but the functions
tested here MUST remain behaviorally identical.
"""

import os
import sys

# Add multi_agent to sys.path so that amesim_multi_agent is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amesim_multi_agent.agents.bridge_core import (
    detect_dangling_ports_detailed,
    parse_topology_pairs,
    verify_components_exist,
)


# ============================================================================
# parse_topology_pairs()  —  Topology text parsing
# ============================================================================


class TestParseTopologyPairsArrowFormat:
    """Single and chained arrow connectors: → -> -- —"""

    def test_single_arrow(self):
        pairs = parse_topology_pairs("Pump → Valve", ["Pump", "Valve"])
        assert pairs == [("Pump", "Valve")]

    def test_chained_arrows_three(self):
        pairs = parse_topology_pairs("A → B → C", ["A", "B", "C"])
        assert pairs == [("A", "B"), ("B", "C")]

    def test_chained_arrows_four(self):
        pairs = parse_topology_pairs(
            "Pump → Valve → Tank → Nozzle",
            ["Pump", "Valve", "Tank", "Nozzle"],
        )
        assert pairs == [("Pump", "Valve"), ("Valve", "Tank"), ("Tank", "Nozzle")]

    def test_dash_arrow(self):
        pairs = parse_topology_pairs("A -> B", ["A", "B"])
        assert pairs == [("A", "B")]

    def test_double_dash(self):
        pairs = parse_topology_pairs("A -- B", ["A", "B"])
        assert pairs == [("A", "B")]

    def test_em_dash(self):
        pairs = parse_topology_pairs("A — B", ["A", "B"])
        assert pairs == [("A", "B")]


class TestParseTopologyPairsSemicolonSeparator:
    """Multi-segment topologies joined by ; or ；"""

    def test_two_segments_semicolon(self):
        pairs = parse_topology_pairs("A → B; C → D", ["A", "B", "C", "D"])
        assert pairs == [("A", "B"), ("C", "D")]

    def test_two_segments_chinese_semicolon(self):
        pairs = parse_topology_pairs("A → B；C → D", ["A", "B", "C", "D"])
        assert pairs == [("A", "B"), ("C", "D")]

    def test_three_segments(self):
        pairs = parse_topology_pairs(
            "A → B; B → C; C → D",
            ["A", "B", "C", "D"],
        )
        assert pairs == [("A", "B"), ("B", "C"), ("C", "D")]


class TestParseTopologyPairsChineseConnectors:
    """Chinese connector verbs: 连接到, 流入, 输出到, etc.

    NOTE: The regex for the second component requires at least 2 \\w chars
    (\\w matches Unicode letters in Python 3).  Single-character aliases
    like "A"/"B" will NOT produce pairs via Chinese connectors — the
    regex simply does not match single-letter targets.
    """

    def test_lian_jie_dao(self):
        pairs = parse_topology_pairs("Pump连接到Valve", ["Pump", "Valve"])
        assert pairs == [("Pump", "Valve")]

    def test_liu_ru(self):
        pairs = parse_topology_pairs("Pump流入Valve", ["Pump", "Valve"])
        assert pairs == [("Pump", "Valve")]

    def test_shu_chu_dao(self):
        pairs = parse_topology_pairs("Pump输出到Valve", ["Pump", "Valve"])
        assert pairs == [("Pump", "Valve")]

    def test_with_port_qualifier(self):
        # The trailing port qualifier on the target component is captured as
        # part of the name by the regex; fuzzy matching resolves it.
        pairs = parse_topology_pairs("Pump的出口连接到Valve的入口", ["Pump", "Valve"])
        assert pairs == [("Pump", "Valve")]

    def test_single_char_aliases_not_matched(self):
        # Known limitation: single-char second component doesn't match
        # the 2-char minimum in the regex capture group.
        pairs = parse_topology_pairs("A连接到B", ["A", "B"])
        assert pairs == []


class TestParseTopologyPairsCommaFallback:
    """Comma-delimited sequence used when no arrow/Chinese connector matches."""

    def test_comma_sequence(self):
        pairs = parse_topology_pairs("Pump, Valve, Tank", ["Pump", "Valve", "Tank"])
        assert pairs == [("Pump", "Valve"), ("Valve", "Tank")]

    def test_comma_with_spaces(self):
        pairs = parse_topology_pairs(
            "  Pump  ,  Valve , Tank  ",
            ["Pump", "Valve", "Tank"],
        )
        assert pairs == [("Pump", "Valve"), ("Valve", "Tank")]


class TestParseTopologyPairsFuzzyMatching:
    """Fuzzy alias resolution: substring containment in either direction."""

    def test_topology_has_longer_name(self):
        # Topology text contains "Pump_Outlet"; known alias list has "Pump"
        pairs = parse_topology_pairs("Pump_Outlet → Valve", ["Pump", "Valve"])
        assert pairs == [("Pump", "Valve")]

    def test_alias_has_longer_name(self):
        # Topology text says "Pump"; known alias is "Pump_Outlet"
        pairs = parse_topology_pairs("Pump → Valve", ["Pump_Outlet", "Valve"])
        assert pairs == [("Pump_Outlet", "Valve")]

    def test_fuzzy_both_directions(self):
        pairs = parse_topology_pairs(
            "Relief_Valve_Dual → Tank_Reservoir",
            ["Relief_Valve", "Tank"],
        )
        assert pairs == [("Relief_Valve", "Tank")]

    def test_case_insensitive(self):
        pairs = parse_topology_pairs("pump → valve", ["Pump", "Valve"])
        assert pairs == [("Pump", "Valve")]


class TestParseTopologyPairsEdgeCases:
    """Empty input, no matches, self-loops, duplicates."""

    def test_empty_string(self):
        pairs = parse_topology_pairs("", ["A", "B"])
        assert pairs == []

    def test_whitespace_only(self):
        pairs = parse_topology_pairs("   ", ["A", "B"])
        assert pairs == []

    def test_no_matching_aliases(self):
        pairs = parse_topology_pairs("X → Y", ["A", "B"])
        assert pairs == []

    def test_self_loop_skipped(self):
        # "A → A" should be skipped (from_alias == to_alias)
        pairs = parse_topology_pairs("A → A", ["A"])
        assert pairs == []

    def test_chain_with_self_loop_in_middle(self):
        pairs = parse_topology_pairs("A → B → B → C", ["A", "B", "C"])
        # A→B valid, B→B skipped (self-loop), B→C valid
        assert pairs == [("A", "B"), ("B", "C")]

    def test_bidirectional_chain_a_b_a(self):
        # "A → B → A" — both edges are distinct, neither is self-loop
        pairs = parse_topology_pairs("A → B → A", ["A", "B"])
        assert pairs == [("A", "B"), ("B", "A")]

    def test_duplicate_pair_not_added_twice(self):
        pairs = parse_topology_pairs("A → B; A → B", ["A", "B"])
        assert pairs == [("A", "B")]

    def test_empty_alias_list(self):
        pairs = parse_topology_pairs("A → B", [])
        assert pairs == []

    def test_alias_with_empty_string(self):
        pairs = parse_topology_pairs("A → B", ["A", "B", ""])
        assert pairs == [("A", "B")]

    def test_mixed_real_and_fake_names(self):
        # Only the names matching aliases produce pairs
        pairs = parse_topology_pairs("Pump → Ghost → Valve", ["Pump", "Valve"])
        assert pairs == [("Pump", "Valve")]


# ============================================================================
# verify_components_exist()  —  Component existence against real index
# ============================================================================


class TestVerifyComponentsExist:
    """Tests that read from the real component_index.json in the knowledge base."""

    REAL_ICON_KEY = "libac:ac_accumulator"

    def test_all_exist(self):
        result = verify_components_exist([
            {"alias": "Accum", "icon_key": self.REAL_ICON_KEY},
        ])
        assert result["all_exist"] is True
        assert len(result["verified"]) == 1
        assert result["verified"][0]["exists"] is True
        assert result["verified"][0]["alias"] == "Accum"
        assert result["missing"] == []

    def test_some_missing(self):
        result = verify_components_exist([
            {"alias": "GoodComp", "icon_key": self.REAL_ICON_KEY},
            {"alias": "BadComp", "icon_key": "nosuchlib:nosuch_component_xyz"},
        ])
        assert result["all_exist"] is False
        assert len(result["verified"]) == 1
        assert len(result["missing"]) == 1
        assert result["missing"][0]["alias"] == "BadComp"
        assert result["missing"][0]["exists"] is False

    def test_empty_list(self):
        result = verify_components_exist([])
        assert result["all_exist"] is True
        assert result["verified"] == []
        assert result["missing"] == []

    def test_returned_library_field(self):
        result = verify_components_exist([
            {"alias": "Test", "icon_key": self.REAL_ICON_KEY},
        ])
        assert result["verified"][0]["library"] == "libac"


# ============================================================================
# detect_dangling_ports_detailed()  —  Dangling port detection
# ============================================================================


class TestDetectDanglingPortsDetailed:
    """Tests for the pure-code dangling port detector."""

    # --- helpers ---

    @staticmethod
    def _make_port(index, tag, variables):
        return {"port_index": index, "port_tag": tag, "variables": variables}

    @staticmethod
    def _make_conn(from_a, from_p, to_a, to_p):
        return {
            "from_alias": from_a,
            "from_port": from_p,
            "to_alias": to_a,
            "to_port": to_p,
        }

    @staticmethod
    def _make_info(icon_key="lib:comp", library="lib", submodel_id="SM01"):
        return {"icon_key": icon_key, "library": library, "submodel_id": submodel_id}

    # --- tests ---

    def test_all_ports_connected_no_dangling(self):
        connections = [
            self._make_conn("Source", 1, "Pump", 1),
            self._make_conn("Pump", 2, "Valve", 1),
            self._make_conn("Valve", 2, "Tank", 1),
        ]
        component_ports = {
            "Source": [self._make_port(1, "out", [{"io": "1", "norm": "Q"}])],
            "Pump": [self._make_port(1, "in", [{"io": "2", "norm": "P"}]), self._make_port(2, "out", [{"io": "1", "norm": "Q"}])],
            "Valve": [self._make_port(1, "in", [{"io": "2", "norm": "P"}]), self._make_port(2, "out", [{"io": "1", "norm": "Q"}])],
            "Tank": [self._make_port(1, "in", [{"io": "2", "norm": "P"}])],
        }
        component_info = {
            "Source": self._make_info("src:lib"),
            "Pump": self._make_info("pump:hyd_pump"),
            "Valve": self._make_info("valve:relief"),
            "Tank": self._make_info("tank:reservoir"),
        }
        dangling = detect_dangling_ports_detailed(
            connections, component_ports, component_info,
        )
        assert dangling == []

    def test_some_ports_unconnected(self):
        connections = [
            self._make_conn("Pump", 1, "Valve", 1),
        ]
        component_ports = {
            "Pump": [self._make_port(1, "port1", [{"io": "1", "norm": "Q"}])],
            "Valve": [
                self._make_port(1, "in", [{"io": "2", "norm": "P"}]),
                self._make_port(2, "out", [{"io": "1", "norm": "Q"}]),
            ],
        }
        component_info = {
            "Pump": self._make_info("p:lib"),
            "Valve": self._make_info("v:lib"),
        }
        dangling = detect_dangling_ports_detailed(
            connections, component_ports, component_info,
        )
        assert len(dangling) == 1
        assert dangling[0]["alias"] == "Valve"
        assert dangling[0]["port_index"] == 2

    def test_empty_connections_all_ports_dangling(self):
        connections = []
        component_ports = {
            "Comp": [self._make_port(1, "p1", [{"io": "1", "norm": "x"}])],
        }
        component_info = {"Comp": self._make_info()}
        dangling = detect_dangling_ports_detailed(
            connections, component_ports, component_info,
        )
        assert len(dangling) == 1
        assert dangling[0]["alias"] == "Comp"
        assert dangling[0]["port_index"] == 1

    def test_empty_component_ports(self):
        connections = [self._make_conn("A", 1, "B", 1)]
        component_ports = {}
        component_info = {}
        dangling = detect_dangling_ports_detailed(
            connections, component_ports, component_info,
        )
        assert dangling == []

    def test_io_summary_output(self):
        connections = []
        component_ports = {
            "Comp": [self._make_port(1, "out", [{"io": "1", "norm": "x"}])],
        }
        component_info = {"Comp": self._make_info()}
        dangling = detect_dangling_ports_detailed(
            connections, component_ports, component_info,
        )
        assert len(dangling) == 1
        assert dangling[0]["io_summary"] == "output"

    def test_io_summary_input(self):
        connections = []
        component_ports = {
            "Comp": [self._make_port(1, "in", [{"io": "2", "norm": "x"}])],
        }
        component_info = {"Comp": self._make_info()}
        dangling = detect_dangling_ports_detailed(
            connections, component_ports, component_info,
        )
        assert len(dangling) == 1
        assert dangling[0]["io_summary"] == "input"

    def test_io_summary_bidirectional_mixed_io(self):
        connections = []
        component_ports = {
            "Comp": [
                self._make_port(1, "bidir", [
                    {"io": "1", "norm": "x"},
                    {"io": "2", "norm": "y"},
                ]),
            ],
        }
        component_info = {"Comp": self._make_info()}
        dangling = detect_dangling_ports_detailed(
            connections, component_ports, component_info,
        )
        assert len(dangling) == 1
        assert dangling[0]["io_summary"] == "bidirectional"

    def test_io_summary_unknown(self):
        # "unknown" is expected … but io_values = set() triggers
        # all(v=="1" for v in set()) == True, so the actual result is "output".
        # This documents the real (possibly unintentional) behaviour.
        connections = []
        component_ports = {
            "Comp": [self._make_port(1, "unk", [{"io": "", "norm": "x"}])],
        }
        component_info = {"Comp": self._make_info()}
        dangling = detect_dangling_ports_detailed(
            connections, component_ports, component_info,
        )
        assert len(dangling) == 1
        assert dangling[0]["io_summary"] == "output"

    def test_io_summary_with_io_3_not_dangling_when_connected(self):
        # io="3" is bidirectional in Amesim; when port IS connected → NOT dangling
        connections = [
            self._make_conn("Src", 1, "Comp", 1),
        ]
        component_ports = {
            "Src": [self._make_port(1, "out", [{"io": "1", "norm": "Q"}])],
            "Comp": [self._make_port(1, "bidir", [{"io": "3", "norm": "P"}])],
        }
        component_info = {
            "Src": self._make_info("s:lib"),
            "Comp": self._make_info("c:lib"),
        }
        dangling = detect_dangling_ports_detailed(
            connections, component_ports, component_info,
        )
        assert dangling == []

    def test_io_summary_with_io_3_unconnected(self):
        # io="3" is filtered out (not in ("1","2")), leaving empty io_values.
        # all() on empty set returns True → "output".
        connections = []
        component_ports = {
            "Comp": [self._make_port(1, "bidir", [{"io": "3", "norm": "P"}])],
        }
        component_info = {"Comp": self._make_info("c:lib")}
        dangling = detect_dangling_ports_detailed(
            connections, component_ports, component_info,
        )
        assert len(dangling) == 1
        assert dangling[0]["io_summary"] == "output"

    def test_result_contains_all_expected_fields(self):
        connections = []
        component_ports = {
            "Comp": [self._make_port(7, "tag7", [{"io": "1", "norm": "N1"}, {"io": "", "norm": "N2"}])],
        }
        component_info = {
            "Comp": self._make_info("x:y", "x-lib", "SM42"),
        }
        dangling = detect_dangling_ports_detailed(
            connections, component_ports, component_info,
        )
        assert len(dangling) == 1
        d = dangling[0]
        assert d["alias"] == "Comp"
        assert d["port_index"] == 7
        assert d["port_tag"] == "tag7"
        assert d["io_summary"] == "output"
        assert d["icon_key"] == "x:y"
        assert d["library"] == "x-lib"
        assert d["submodel_id"] == "SM42"
        assert d["variable_norms"] == ["N1", "N2"]

    def test_multiple_components_mixed_connected(self):
        connections = [
            self._make_conn("A", 1, "B", 1),
        ]
        component_ports = {
            "A": [
                self._make_port(1, "p1", [{"io": "1", "norm": "a1"}]),
                self._make_port(2, "p2", [{"io": "2", "norm": "a2"}]),
            ],
            "B": [
                self._make_port(1, "p1", [{"io": "2", "norm": "b1"}]),
                self._make_port(2, "p2", [{"io": "1", "norm": "b2"}]),
            ],
            "C": [
                self._make_port(1, "p1", [{"io": "2", "norm": "c1"}]),
            ],
        }
        component_info = {
            "A": self._make_info(),
            "B": self._make_info(),
            "C": self._make_info(),
        }
        dangling = detect_dangling_ports_detailed(
            connections, component_ports, component_info,
        )
        assert len(dangling) == 3
        aliases = {d["alias"] for d in dangling}
        assert aliases == {"A", "B", "C"}
        a_dangling = [d for d in dangling if d["alias"] == "A"]
        assert len(a_dangling) == 1
        assert a_dangling[0]["port_index"] == 2
        assert a_dangling[0]["io_summary"] == "input"
        b_dangling = [d for d in dangling if d["alias"] == "B"]
        assert len(b_dangling) == 1
        assert b_dangling[0]["port_index"] == 2
        assert b_dangling[0]["io_summary"] == "output"
        c_dangling = [d for d in dangling if d["alias"] == "C"]
        assert len(c_dangling) == 1
        assert c_dangling[0]["port_index"] == 1
        assert c_dangling[0]["io_summary"] == "input"
