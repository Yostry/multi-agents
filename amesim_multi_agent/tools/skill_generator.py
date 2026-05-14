"""Skill auto-generation - creates reusable skills from successful modeling sessions."""
from __future__ import annotations
import json, os
from datetime import datetime
from ..protocols import GeneratedSkill

class SkillGenerator:
    """Generates skills from successful modeling sessions."""
    
    def __init__(self, skills_dir=None):
        self.skills_dir = skills_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "knowledge_base", "skills"
        )
        os.makedirs(self.skills_dir, exist_ok=True)
    
    def generate(self, model_name, component_selections, topology, reasoning_chain=None):
        """Generate a skill from a successful modeling session."""
        skill = GeneratedSkill(
            model_name=model_name,
            created_at=datetime.now().isoformat(),
            component_selections=component_selections,
            topology_description=str(topology),
            reasoning_chain=reasoning_chain or [],
        )
        model_dir = os.path.join(self.skills_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save machine-readable
        with open(os.path.join(model_dir, "skill.json"), "w", encoding="utf-8") as f:
            json.dump(self._to_dict(skill), f, ensure_ascii=False, indent=2)
        
        # Save human-readable
        md = self._to_markdown(skill)
        with open(os.path.join(model_dir, "skill.md"), "w", encoding="utf-8") as f:
            f.write(md)
        
        # Update index
        self._update_index(skill)
        return skill
    
    def _to_dict(self, skill):
        return {
            "model_name": skill.model_name,
            "created_at": skill.created_at,
            "component_selections": skill.component_selections,
            "topology_description": skill.topology_description,
            "reasoning_chain": skill.reasoning_chain,
        }
    
    def _to_markdown(self, skill):
        lines = [f"# Skill: {skill.model_name}", "", f"Created: {skill.created_at}", ""]
        lines.append("## Topology")
        lines.append(str(skill.topology_description))
        lines.append("")
        lines.append("## Components")
        for c in skill.component_selections:
            lines.append(f"- {c.get('alias', c.get('icon_key','?'))}")
        return "\n".join(lines)
    
    def _update_index(self, skill):
        idx_path = os.path.join(self.skills_dir, "index.json")
        index = {}
        if os.path.exists(idx_path):
            index = json.load(open(idx_path, "r", encoding="utf-8"))
        index[skill.model_name] = {"created_at": skill.created_at, "component_count": len(skill.component_selections)}
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    
    def search(self, query):
        """Search skills by keyword."""
        idx_path = os.path.join(self.skills_dir, "index.json")
        if not os.path.exists(idx_path):
            return []
        index = json.load(open(idx_path, "r", encoding="utf-8"))
        return [k for k in index if query.lower() in k.lower()]
    
    def load(self, model_name):
        """Load a skill by model name."""
        path = os.path.join(self.skills_dir, model_name, "skill.json")
        if os.path.exists(path):
            return json.load(open(path, "r", encoding="utf-8"))
        return None
