"""Demand Refinement Agent - clarifies user requirements before component selection."""
from __future__ import annotations
from agents import Agent
from ..kimi_provider import KIMI_MODEL_NAME

QUESTION_TEMPLATES = [
    {"domain": "hydraulic", "questions": [
        {"question": "Hydraulic pump flow rate range?", "options": ["A) <10 L/min", "B) 10-100 L/min", "C) >100 L/min"], "key": "flow_rate_range"},
        {"question": "System maximum pressure?", "options": ["A) <10 bar", "B) 10-100 bar", "C) 100-350 bar", "D) >350 bar"], "key": "max_pressure"},
        {"question": "Control valve type?", "options": ["A) Directional", "B) Pressure control", "C) Flow control", "D) Proportional/servo"], "key": "valve_type"},
    ]},
    {"domain": "mechanical", "questions": [
        {"question": "Motion type?", "options": ["A) Linear", "B) Rotary", "C) Both"], "key": "motion_type"},
        {"question": "Load range?", "options": ["A) <100 N", "B) 100-1000 N", "C) 1000-10000 N", "D) >10000 N"], "key": "load_range"},
    ]},
    {"domain": "thermal", "questions": [
        {"question": "Heat transfer mode?", "options": ["A) Conduction", "B) Convection", "C) Radiation", "D) Mixed"], "key": "heat_mode"},
        {"question": "Temperature range?", "options": ["A) <0 C", "B) 0-100 C", "C) 100-500 C", "D) >500 C"], "key": "temp_range"},
    ]},
    {"domain": "pneumatic", "questions": [
        {"question": "Pressure source?", "options": ["A) Compressor", "B) Bottled gas", "C) Vacuum pump"], "key": "pressure_source"},
    ]},
]

DEMAND_REFINER_SYSTEM_PROMPT = """\
You are a Demand Refinement Agent for Amesim simulation modeling.
Analyze the user's modeling request and topology plan from the Orchestrator.
For each component requirement, generate 2-3 multiple-choice questions to clarify specific needs.

Output JSON:
{"refined_requirements":[{"requirement_index":1,"original_description":"hydraulic pump","questions":[{"question":"Flow rate?","options":["A) <10","B) 10-100"],"key":"flow_rate"}],"refined_constraints":{}}]}
"""

def create_demand_refiner_agent() -> Agent:
    return Agent(name="DemandRefiner", instructions=DEMAND_REFINER_SYSTEM_PROMPT, model=KIMI_MODEL_NAME, tools=[])
