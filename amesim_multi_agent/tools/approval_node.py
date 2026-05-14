"""Human Approval Node - presents component selections for user confirmation."""
from __future__ import annotations
from dataclasses import dataclass
from ..protocols import ApprovalResponse, SelectedComponent

@dataclass
class ApprovalNode:
    confidence_threshold: float = 0.8

    def format_selections(self, selections, reasoning=""):
        lines = ["=" * 60, "  COMPONENT SELECTION - REVIEW", "=" * 60]
        if reasoning: lines.append(f"`nReasoning: {reasoning[:500]}")
        for i, s in enumerate(selections, 1):
            conf = s.confidence or 0
            flag = "WARNING: LOW CONFIDENCE" if conf < self.confidence_threshold else "OK"
            lines.append(f"`n--- Selection {i}: {s.alias} [{flag}] ---")
            lines.append(f"  Component: {s.icon_key}")
            lines.append(f"  Submodel:  {s.recommended_submodel}")
            lines.append(f"  Confidence: {conf:.0%}")
        lines.append("`n[Options: (A)pprove all | (R)eject | (M)anual override]")
        return "`n".join(lines)

    def present_selections(self, selections, reasoning="", interactive=False):
        if interactive:
            print(self.format_selections(selections, reasoning))
            response = input("Your choice [A/R/M]: ").strip().upper()
            if response == "A": return ApprovalResponse(requirement_index=0, action="approved")
            elif response == "M": return ApprovalResponse(requirement_index=0, action="manual")
            else: return ApprovalResponse(requirement_index=0, action="reject")
        all_high = all((s.confidence or 0) >= self.confidence_threshold for s in selections)
        return ApprovalResponse(requirement_index=0, action="approved" if all_high else "reject")
