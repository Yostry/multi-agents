"""Tests for build_model_direct() deterministic behavior.

These are TDD RED tests — they should FAIL initially because the integration
with _run_builder_stage is not yet complete (pending Task 4).
"""
import ast
import json
import tempfile
from pathlib import Path

import pytest

from multi_agent.amesim_multi_agent.agents.builder import build_model_direct
from multi_agent.amesim_multi_agent.tools.builder_tools import repair_build_script


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def minimal_components():
    """Minimal component list for testing."""
    return [
        {
            "icon_name": "mass_friction_endstops",
            "alias": "Mass",
            "position": [100, 50],
            "submodel": "MAS000",
            "library": "libmec",
            "rotations": 0,
        },
        {
            "icon_name": "spring",
            "alias": "Spring",
            "position": [200, 50],
            "submodel": "SPR000",
            "library": "libmec",
            "rotations": 0,
        },
    ]


@pytest.fixture
def minimal_connections():
    """Minimal connection list for testing."""
    return [
        {
            "from_alias": "Mass",
            "from_port": 1,
            "to_alias": "Spring",
            "to_port": 0,
            "type": "direct",
        }
    ]


@pytest.fixture
def temp_model_dir():
    """Temporary directory for model output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Test Cases (all xfail - integration pending Task 4)
# =============================================================================
@pytest.mark.xfail(reason="Integration with _run_builder_stage pending Task 4")
def test_deterministic_output(minimal_components, minimal_connections):
    """Same components+connections twice produces identical script content.

    Verifies that build_model_direct() is deterministic - calling it twice
    with the same inputs produces byte-for-byte identical output scripts.
    """
    model_name = "TestDeterministic"

    # First call
    result1 = build_model_direct(
        model_name=model_name,
        components=minimal_components,
        connections=minimal_connections,
        mode="manual",
    )
    result1_data = json.loads(result1)
    script_path1 = result1_data["script_path"]

    # Second call with identical inputs
    result2 = build_model_direct(
        model_name=model_name,
        components=minimal_components,
        connections=minimal_connections,
        mode="manual",
    )
    result2_data = json.loads(result2)
    script_path2 = result2_data["script_path"]

    # Read both scripts
    with open(script_path1, "r", encoding="utf-8") as f:
        content1 = f.read()
    with open(script_path2, "r", encoding="utf-8") as f:
        content2 = f.read()

    # Scripts should be identical
    assert content1 == content2, (
        f"Scripts differ on second call with identical inputs.\n"
        f"First: {script_path1}\n"
        f"Second: {script_path2}"
    )


@pytest.mark.xfail(reason="Integration with _run_builder_stage pending Task 4")
def test_script_syntax_valid(minimal_components, minimal_connections):
    """Generated .py file parses successfully with ast.parse().

    Verifies that the generated build script is valid Python syntax.
    """
    model_name = "TestSyntax"

    result = build_model_direct(
        model_name=model_name,
        components=minimal_components,
        connections=minimal_connections,
        mode="manual",
    )
    result_data = json.loads(result)
    script_path = result_data["script_path"]

    # Script must be parseable
    with open(script_path, "r", encoding="utf-8") as f:
        script_content = f.read()

    # Should not raise SyntaxError
    try:
        ast.parse(script_content)
    except SyntaxError as e:
        pytest.fail(f"Generated script has invalid syntax: {e}")


@pytest.mark.xfail(reason="Integration with _run_builder_stage pending Task 4")
def test_api_calls_present(minimal_components, minimal_connections):
    """Script contains expected ame_apy API calls.

    Verifies that the generated script contains the core Amesim API calls:
    - add_component (for each component)
    - add_line (for each connection)
    - set_parameter (for parameter assignment)
    """
    model_name = "TestAPICalls"
    params = {"mass@Mass": "10"}

    result = build_model_direct(
        model_name=model_name,
        components=minimal_components,
        connections=minimal_connections,
        parameters=params,
        mode="manual",
    )
    result_data = json.loads(result)
    script_path = result_data["script_path"]

    with open(script_path, "r", encoding="utf-8") as f:
        script_content = f.read()

    # Check for expected API calls
    assert "add_component" in script_content, "Script missing add_component call"
    assert "add_line" in script_content, "Script missing add_line call"
    assert "set_parameter" in script_content, "Script missing set_parameter call"


@pytest.mark.xfail(reason="Integration with _run_builder_stage pending Task 4")
def test_manual_mode_build_only(minimal_components, minimal_connections):
    """mode='manual' returns status='build_only' without executing AMEPython.

    Verifies that when mode='manual', the function returns immediately after
    generating the script, without attempting to run AMEPython.exe.
    """
    model_name = "TestManualMode"

    result = build_model_direct(
        model_name=model_name,
        components=minimal_components,
        connections=minimal_connections,
        mode="manual",
    )
    result_data = json.loads(result)

    assert result_data["status"] == "build_only", (
        f"Expected status='build_only' for manual mode, got '{result_data['status']}'"
    )
    assert result_data["mode"] == "manual"
    assert "script_path" in result_data
    # stderr should be empty for manual mode (no execution attempted)
    assert result_data["stderr"] == ""


@pytest.mark.xfail(reason="Integration with _run_builder_stage pending Task 4")
def test_empty_components_handled(minimal_connections):
    """components=[] does not crash and produces valid script.

    Verifies that an empty component list is handled gracefully without
    raising exceptions.
    """
    model_name = "TestEmptyComponents"

    # Should not raise
    result = build_model_direct(
        model_name=model_name,
        components=[],
        connections=minimal_connections,
        mode="manual",
    )
    result_data = json.loads(result)

    # Should return success
    assert result_data["status"] == "build_only"
    assert "script_path" in result_data

    # Script should still be valid Python
    script_path = result_data["script_path"]
    with open(script_path, "r", encoding="utf-8") as f:
        script_content = f.read()
    ast.parse(script_content)  # Should not raise


@pytest.mark.xfail(reason="Integration with _run_builder_stage pending Task 4")
def test_alias_mapping_correct(minimal_components, minimal_connections):
    """Component alias correctly maps to add_component calls in script.

    Verifies that each component's alias appears correctly in the generated
    script's add_component calls, maintaining proper mapping.
    """
    model_name = "TestAliasMapping"

    result = build_model_direct(
        model_name=model_name,
        components=minimal_components,
        connections=minimal_connections,
        mode="manual",
    )
    result_data = json.loads(result)
    script_path = result_data["script_path"]

    with open(script_path, "r", encoding="utf-8") as f:
        script_content = f.read()

    # Each alias should appear in an add_component context
    for comp in minimal_components:
        alias = comp["alias"]
        # The alias should appear in the script
        assert alias in script_content, f"Alias '{alias}' not found in script"

        # Should appear as the alias argument to add_component
        # Looking for pattern like: add_component(..., alias="Mass", ...)
        assert f'alias="{alias}"' in script_content, (
            f"Alias '{alias}' not properly used in add_component call"
        )


# =============================================================================
# Repair Workflow Tests (xfail - repair function integration pending Task 4)
# =============================================================================
@pytest.mark.xfail(reason="Integration with _run_builder_stage pending Task 4")
def test_repair_reads_existing_script():
    """repair_build_script reads a script file and returns its content.

    Verifies that repair_build_script correctly reads an existing .py file
    and returns its content in the script_content field.
    """
    script_content = '''# Test build script
import ame_apy
model = ame_apy.create_model("TestModel")
ame_apy.add_component(model, "mass_friction_endstops", alias="Mass")
'''

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script_content)
        script_path = f.name

    try:
        result = repair_build_script(
            script_path=script_path,
            error_info="Some build error",
            fixes="Fix the parameter name",
        )
        result_data = json.loads(result)

        assert result_data["script_content"] == script_content, (
            "repair_build_script did not return the expected script content"
        )
    finally:
        Path(script_path).unlink()


@pytest.mark.xfail(reason="Integration with _run_builder_stage pending Task 4")
def test_repair_output_format_valid():
    """repair_build_script returns JSON with required keys.

    Verifies that the returned JSON contains all required keys:
    - script_content: str
    - error_info: str
    - applied_fixes: str
    """
    script_content = "# Simple test script\npass\n"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script_content)
        script_path = f.name

    try:
        result = repair_build_script(
            script_path=script_path,
            error_info="Build failed with SyntaxError",
            fixes="Add missing import",
        )
        result_data = json.loads(result)

        # Check all required keys are present
        assert "script_content" in result_data, "Missing 'script_content' key"
        assert "error_info" in result_data, "Missing 'error_info' key"
        assert "applied_fixes" in result_data, "Missing 'applied_fixes' key"

        # Check types
        assert isinstance(result_data["script_content"], str)
        assert isinstance(result_data["error_info"], str)
        assert isinstance(result_data["applied_fixes"], str)

        # Check error_info and applied_fixes echo the inputs
        assert result_data["error_info"] == "Build failed with SyntaxError"
        assert result_data["applied_fixes"] == "Add missing import"
    finally:
        Path(script_path).unlink()


@pytest.mark.xfail(reason="Integration with _run_builder_stage pending Task 4")
def test_repair_preserves_unchanged_lines():
    """repair_build_script returns unchanged content when no fixes needed.

    Verifies that when repair is called on a valid script, the returned
    script_content is identical to the original — only targeted lines
    should be modified by the repair workflow, not the entire file.
    """
    original_script = '''# Amesim build script
import ame_apy

def build_model():
    model = ame_apy.create_model("TestModel")
    ame_apy.add_component(model, "mass_friction_endstops", alias="Mass")
    ame_apy.add_component(model, "spring", alias="Spring")
    ame_apy.add_line(model, "Mass", 1, "Spring", 0)
    return model
'''

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(original_script)
        script_path = f.name

    try:
        result = repair_build_script(
            script_path=script_path,
            error_info="",
            fixes="",
        )
        result_data = json.loads(result)

        # Script content should be unchanged when no fixes needed
        assert result_data["script_content"] == original_script, (
            "repair_build_script altered script content when no fixes were specified"
        )
    finally:
        Path(script_path).unlink()