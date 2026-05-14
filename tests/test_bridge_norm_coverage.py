"""Verify port norm data completeness in knowledge base.

Refactoring Bridge to use batch LLM review requires norm data
in variables[norm] for ports. This test ensures key libraries
have adequate coverage (>80% of ports with norm data).
"""
from __future__ import annotations

import json
import os
import sys
import random
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "amesim_multi_agent")
)

import pytest

# ---- Constants ----
TARGET_LIBRARIES = ["libmec", "libhydr", "libth", "libsig", "libtpf"]
MAX_SAMPLES_PER_LIBRARY = 30
COVERAGE_THRESHOLD = 0.80


def _load_component_index():
    """Load component_index.json from knowledge base."""
    comp_idx_path = (
        Path(__file__).resolve().parent.parent.parent
        / "knowledge_base"
        / "json"
        / "index"
        / "component_index.json"
    )
    if not comp_idx_path.exists():
        pytest.skip(f"Component index not found: {comp_idx_path}")
    with comp_idx_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_searcher():
    """Lazy-load the unified searcher singleton."""
    try:
        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(__file__), "..", ".."
            ),
        )
        from amesim_builder.unified_search import get_searcher
        return get_searcher()
    except ImportError:
        pytest.skip("amesim_builder.unified_search not available")


def _count_port_norms(ports: list[dict]) -> tuple[int, int]:
    """Count ports with and without non-empty norm data.

    Returns (ports_with_norm, total_ports).
    """
    total = 0
    with_norm = 0

    for port in ports:
        total += 1
        variables = port.get("variables", [])
        if not variables:
            continue
        for var in variables:
            norm_val = var.get("norm", "")
            if norm_val and str(norm_val).strip():
                with_norm += 1
                break

    return with_norm, total


def test_port_norm_coverage():
    """Test key libraries have adequate norm data (>80% ports with norms)."""
    random.seed(42)

    comp_idx = _load_component_index()
    searcher = _get_searcher()

    # Group components by library
    by_library: dict[str, list[str]] = {}
    for key, entry in comp_idx.items():
        lib = entry.get("library", "")
        if not lib:
            continue
        by_library.setdefault(lib, []).append(key)

    results: dict[str, dict] = {}
    total_ports = 0
    total_with_norm = 0

    print("\n=== Port Norm Coverage Report ===\n")

    for lib_name in TARGET_LIBRARIES:
        icon_keys = by_library.get(lib_name, [])
        if not icon_keys:
            print(f"  {lib_name}: NOT FOUND in index")
            results[lib_name] = {
                "sampled": 0,
                "total_ports": 0,
                "with_norm": 0,
                "pct": None,
            }
            continue

        # Sample up to MAX_SAMPLES_PER_LIBRARY
        sample_size = min(MAX_SAMPLES_PER_LIBRARY, len(icon_keys))
        sampled = random.sample(icon_keys, sample_size)

        lib_total = 0
        lib_with_norm = 0
        skipped = 0

        for icon_key in sampled:
            doc = searcher.get_document_by_iconkey(icon_key)
            if doc is None:
                skipped += 1
                continue
            ports = doc.get("ports", [])
            if not ports:
                skipped += 1
                continue
            w, t = _count_port_norms(ports)
            lib_with_norm += w
            lib_total += t

        pct = (lib_with_norm / lib_total * 100) if lib_total > 0 else None
        pct_str = f"{pct:.1f}%" if pct is not None else "N/A (no ports)"

        print(
            f"  {lib_name}: {lib_with_norm}/{lib_total} ports with norm "
            f"({pct_str}) — sampled {sample_size} components, skipped {skipped}"
        )

        results[lib_name] = {
            "sampled": sample_size,
            "skipped": skipped,
            "total_ports": lib_total,
            "with_norm": lib_with_norm,
            "pct": pct,
        }
        total_ports += lib_total
        total_with_norm += lib_with_norm

    overall_pct = (
        (total_with_norm / total_ports * 100) if total_ports > 0 else 0.0
    )
    print(f"\n  OVERALL: {total_with_norm}/{total_ports} ports with norm "
          f"({overall_pct:.1f}%)")
    print(f"  Threshold: {COVERAGE_THRESHOLD * 100:.0f}%\n")

    assert (
        total_ports > 0
    ), "No ports found across all sampled libraries — check searcher/DB"
    assert overall_pct >= COVERAGE_THRESHOLD * 100, (
        f"Overall norm coverage {overall_pct:.1f}% below threshold "
        f"{COVERAGE_THRESHOLD * 100:.0f}%"
    )
