"""Secondary Component Selector - deterministic port-matching selection. No LLM."""
from __future__ import annotations
import json, os
from amesim_builder.unified_search import get_searcher

class SecondarySelector:
    def __init__(self):
        self.searcher = get_searcher()
        self._port_physics = None
        self._connection_rules = None
    
    @property
    def port_physics(self):
        if self._port_physics is None:
            pp = os.path.join(os.path.dirname(__file__), "..", "..", "..", "knowledge_base", "json", "components", "port_physics.json")
            self._port_physics = json.load(open(pp, "r", encoding="utf-8")) if os.path.exists(pp) else {}
        return self._port_physics
    
    @property
    def connection_rules(self):
        if self._connection_rules is None:
            cr = os.path.join(os.path.dirname(__file__), "..", "..", "..", "knowledge_base", "json", "components", "connection_rules.json")
            self._connection_rules = json.load(open(cr, "r", encoding="utf-8")) if os.path.exists(cr) else {"rules":[],"cross_domain_bridges":[]}
        return self._connection_rules
    
    def find_compatible(self, domain, quantity, direction, top_k=5, library=None):
        results = self.searcher.search(quantity, top_k=top_k * 3, library=library)
        scored = []
        for r in results:
            ik = f"{r['library']}:{r['icon_name']}"
            pp_data = self.port_physics.get(ik, {})
            score = r.get("rrf_score", 0)
            for port in pp_data.get("ports", []):
                if port.get("domain") == domain and port.get("physical_quantity") == quantity:
                    score += 0.02 if port.get("direction") != direction else 0.005
            scored.append((score, r))
        scored.sort(key=lambda x: -x[0])
        return [s[1] for s in scored[:top_k]]
    
    def match_for_unconnected_port(self, domain, quantity, direction, library=None):
        needed_dir = "input" if direction == "output" else "output"
        return self.find_compatible(domain, quantity, needed_dir, top_k=5, library=library)
