"""Pipeline Context - manages intermediate files for the modeling pipeline."""
from __future__ import annotations
import json, os, shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

_MODELS_ROOT = Path(__file__).resolve().parent.parent.parent / "ame_models"

@dataclass
class PipelineContext:
    model_name: str
    model_dir: Path = field(init=False)

    def __post_init__(self):
        self.model_dir = _MODELS_ROOT / self.model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save_topology_plan(self, plan_dict): return self._write_json("_topology_plan.json", plan_dict)
    def load_topology_plan(self): return self._read_json("_topology_plan.json")

    def save_component_selection(self, components): return self._write_json("_component_selection.json", {"model_name":self.model_name,"components":components,"component_count":len(components)})
    def load_component_selection(self): return self._read_json("_component_selection.json")

    def save_component_data(self, data): return self._write_json("_component_data.json", {"model_name":self.model_name,"components":data,"component_count":len(data)})
    def load_component_data(self): return self._read_json("_component_data.json")

    def save_refined_requirements(self, data): return self._write_json("_refined_requirements.json", {"model_name":self.model_name,"refined_requirements":data})
    def load_refined_requirements(self): return self._read_json("_refined_requirements.json")

    def save_approval_log(self, data): return self._write_json("_approval_log.json", {"model_name":self.model_name,"approvals":data})
    def load_approval_log(self): return self._read_json("_approval_log.json")

    def save_secondary_selections(self, data): return self._write_json("_secondary_selection.json", {"model_name":self.model_name,"components":data,"component_count":len(data)})
    def load_secondary_selections(self): return self._read_json("_secondary_selection.json")

    def save_connection_plan(self, plan_dict): return self._write_json("_connection_plan.json", plan_dict)
    def load_connection_plan(self): return self._read_json("_connection_plan.json")

    def save_parameter_data(self, data): return self._write_json("_parameter_data.json", data)
    def load_parameter_data(self): return self._read_json("_parameter_data.json")

    def save_error_summary(self, data): return self._write_json("_error_summary.json", data)
    def load_error_summary(self): return self._read_json("_error_summary.json")

    def save_extracted_json(self, data): return self._write_json(f"{self.model_name}_extracted.json", data)
    def load_extracted_json(self): return self._read_json(f"{self.model_name}_extracted.json")

    def cleanup_intermediate_files(self):
        deleted = []
        for p in self.model_dir.glob("_*"):
            if p.is_file(): p.unlink(); deleted.append(p.name)
        return deleted

    def cleanup_prompt(self):
        choice = input("Delete intermediate files? [Y/N/C]: ").strip().upper()
        if choice in ("Y", "C"):
            deleted = self.cleanup_intermediate_files()
            if deleted:
                print(f"  Cleaned {len(deleted)} files: {', '.join(deleted)}")
            else:
                print("  No intermediate files to clean")
            return True
        else:
            print("  Intermediate files kept")
            return False

    def _write_json(self, filename, data):
        path = self.model_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def _read_json(self, filename):
        path = self.model_dir / filename
        if path.exists():
            return json.load(open(path, "r", encoding="utf-8"))
        return None

def ensure_model_dir(model_name):
    model_dir = _MODELS_ROOT / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir
