"""Code validation layer - deterministic submodel compat + port checks."""
from __future__ import annotations
import json, os
from ..protocols import SelectedComponent

def _load_compat_matrix():
    p = os.path.join(os.path.dirname(__file__), "..", "..", "..", "knowledge_base", "json", "components", "submodel_compat.json")
    return json.load(open(p, "r", encoding="utf-8")) if os.path.exists(p) else {}

def _load_port_physics():
    p = os.path.join(os.path.dirname(__file__), "..", "..", "..", "knowledge_base", "json", "components", "port_physics.json")
    return json.load(open(p, "r", encoding="utf-8")) if os.path.exists(p) else {}

def validate_submodel_compat(icon_key, submodel_id):
    matrix = _load_compat_matrix()
    if not matrix: return True
    valid = matrix.get(icon_key, [])
    if not valid: return True
    return submodel_id in valid

def validate_port_count(icon_key, min_ports):
    pp = _load_port_physics()
    ports = pp.get(icon_key, {}).get("ports", [])
    return len(ports) >= min_ports

def validate_all_selections(selected):
    errors = []
    for s in selected:
        if s.recommended_submodel and s.icon_key:
            if not validate_submodel_compat(s.icon_key, s.recommended_submodel):
                errors.append({"requirement_index": s.requirement_index, "icon_key": s.icon_key, "error": f"Submodel {s.recommended_submodel} not compatible", "severity": "error"})
    return errors
