"""Actual selected-artifact calls; native checks require --scalar-engine native-serial."""

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import invar_python as inv
import jsonschema
import pytest

from conftest import child_timeout
from scalar_cases import SOURCE, SOURCE_PATH, SUCCESS, TRAPS
from test_scalar_acceptance import SCHEMA, oracle_trap


EXPORT_NAMES = tuple(dict.fromkeys(case.name for case in (*SUCCESS, *TRAPS)))


@pytest.fixture(scope="module")
def scalar_module(scalar_engine):
    return inv.compile_source(SOURCE, path=SOURCE_PATH, engine=scalar_engine, exports=EXPORT_NAMES)


def rejected(error_class, code, callback):
    with pytest.raises(error_class) as caught:
        callback()
    document = caught.value.diagnostics
    jsonschema.validate(document, SCHEMA)
    assert json.loads(caught.value.diagnostic_json) == document
    assert document["diagnostics"][0]["code"] == code
    assert "result" not in document
    return document


@pytest.mark.parametrize("case", SUCCESS, ids=lambda case: case.label)
def test_scalar_artifact_results_match_source_oracle(scalar_module, case):
    oracle = inv.compile_source(case.reference_source(), path=SOURCE_PATH)
    assert oracle.run_main() == (case.expected if case.result_type == "i32" else 1)
    assert scalar_module.call(case.name, *case.args) == case.expected


@pytest.mark.parametrize("case", TRAPS, ids=lambda case: case.label)
def test_scalar_artifact_original_diagnostic_matches_oracle(scalar_module, case):
    expected = oracle_trap(case)
    actual = rejected(inv.TrapError, case.trap, lambda: scalar_module.call(case.name, *case.args))
    assert actual["trap"] == expected["trap"] == {"category": case.trap, "span": case.trap_span()}
    assert actual["diagnostics"][0] == expected["diagnostics"][0]


def test_scalar_artifact_success_trap_success_same_handle(scalar_module):
    identity = scalar_module.artifact_id
    for case in TRAPS:
        assert scalar_module.call("signed_add", -2147483648, 2147483647) == -1
        rejected(inv.TrapError, case.trap, lambda: scalar_module.call(case.name, *case.args))
        assert scalar_module.call("zero") == 4294967295
        assert scalar_module.artifact_id == identity


def test_scalar_artifact_inventory_and_order(scalar_module, scalar_engine):
    descriptors = scalar_module.exports()
    assert [item["name"] for item in descriptors] == list(EXPORT_NAMES)
    assert re.fullmatch("[0-9a-f]{64}", scalar_module.artifact_id)
    assert scalar_module.engine == scalar_engine
    assert len({item["symbol"] for item in descriptors}) == len(descriptors)
    for descriptor in descriptors:
        assert descriptor["artifact_id"] == scalar_module.artifact_id
        assert descriptor["engine"] == scalar_engine
        assert descriptor["exported"] and descriptor["callable"]
        assert descriptor["result"] in ("i32", "u32")
        assert all(item["type"] in ("i32", "u32") and item["mode"] == "value"
                   for item in descriptor["parameters"])
    assert all(item["exported"] and item["callable"]
               for item in scalar_module.functions()["source_functions"])


def test_scalar_artifact_explicit_selection(scalar_engine):
    module = inv.compile_source(SOURCE, engine=scalar_engine, exports=["zero", "signed_add"])
    assert [item["name"] for item in module.exports()] == ["zero", "signed_add"]
    functions = {item["name"]: item for item in module.functions()["source_functions"]}
    assert functions["zero"]["callable"] and functions["signed_add"]["callable"]
    assert not functions["signed_div"]["callable"]
    assert module.call("zero") == 4294967295
    rejected(inv.EngineError, "PY_EXPORT_UNAVAILABLE", lambda: module.call("signed_div", 1, 0))
    rejected(inv.ValidationError, "NAME_UNKNOWN", lambda: module.call("absent"))
    rejected(inv.ValidationError, "PY_ARGUMENT_COUNT", lambda: module.call("signed_add", 1))


@pytest.mark.parametrize("name,invalid", [
    ("signed_add", True), ("signed_add", 1.0), ("signed_add", "1"),
    ("signed_add", -2147483649), ("signed_add", 2147483648), ("signed_add", 1 << 100),
    ("unsigned_add", True), ("unsigned_add", 1.0), ("unsigned_add", "1"),
    ("unsigned_add", -1), ("unsigned_add", 4294967296), ("unsigned_add", 1 << 100),
])
def test_scalar_artifact_invalid_values_and_recovery(scalar_module, name, invalid):
    rejected(inv.ValidationError, "TYPE_MISMATCH", lambda: scalar_module.call(name, 1, invalid))
    assert scalar_module.call("zero") == 4294967295


def test_scalar_artifact_callback_and_subclass_rejection(scalar_module):
    class Index:
        called = False

        def __index__(self):
            self.called = True
            return 1

    class Integer(int):
        pass

    index = Index()
    for value in (index, Integer(1)):
        rejected(inv.ValidationError, "TYPE_MISMATCH", lambda: scalar_module.call("signed_add", value, 0))
        assert scalar_module.call("zero") == 4294967295
    assert not index.called


@pytest.mark.parametrize("fuel", [0, -1, True, 1.0, "1", None, 1 << 100, 10_000_001])
def test_scalar_artifact_fuel_validation(scalar_module, fuel):
    rejected(inv.ResourceError, "INVALID_FUEL", lambda: scalar_module.call("zero", fuel=fuel))
    assert scalar_module.call("zero") == 4294967295


def test_scalar_artifact_fuel_failure_resets_invocation(scalar_module):
    actual = rejected(inv.ResourceError, "EVALUATION_LIMIT",
                      lambda: scalar_module.call("triangular", 100000, fuel=3))
    assert actual["fuel_limit"] == actual["steps_used"] == 3
    assert scalar_module.call("zero") == 4294967295


def test_scalar_artifact_call_depth_failure_resets_invocation(scalar_engine):
    source = "fn recurse(n: u32) -> u32 { if n == 0u32 { return 0u32; } return recurse(n - 1u32); }"
    module = inv.compile_source(source, engine=scalar_engine, exports=["recurse"])
    assert module.call("recurse", 1) == 0
    rejected(inv.ResourceError, "CALL_DEPTH_LIMIT", lambda: module.call("recurse", 100))
    assert module.call("recurse", 1) == 0


@pytest.mark.parametrize("exports", ["zero", b"zero", {"zero"}, [1], [True]])
def test_scalar_selection_requires_exact_sequence_and_strings(exports):
    rejected(inv.ValidationError, "TYPE_MISMATCH", lambda: inv.compile_source(SOURCE, exports=exports))


def test_scalar_selection_does_not_execute_python_callbacks():
    class Selection:
        called = False

        def __iter__(self):
            self.called = True
            yield "zero"

    class Names(list):
        called = False

        def __iter__(self):
            self.called = True
            return super().__iter__()

    for selection in (Selection(), Names(["zero"])):
        rejected(inv.ValidationError, "TYPE_MISMATCH", lambda: inv.compile_source(SOURCE, exports=selection))
        assert not selection.called


@pytest.mark.parametrize("exports", [
    pytest.param(["zero"] * 4097, id="too-many-names"),
    pytest.param(["x" * 1_048_577], id="too-much-name-text"),
])
def test_scalar_selection_has_bounded_metadata(exports):
    rejected(inv.ResourceError, "RESOURCE_LIMIT", lambda: inv.compile_source(SOURCE, exports=exports))


def test_scalar_two_artifacts_and_original_source_snapshot(scalar_engine, tmp_path):
    path = tmp_path / "source.inv"
    source = "// Original UTF-8: é.\nfn shift(x: i32) -> i32 { return x + 1; }"
    path.write_text(source)
    first = inv.compile_source(source, path=str(path), engine=scalar_engine, exports=["shift"])
    second = inv.compile_source("fn shift(x: i32) -> i32 { return x + 2; }",
                                path="second.inv", engine=scalar_engine, exports=["shift"])
    assert first.artifact_id != second.artifact_id
    path.write_text("fn shift(x: i32) -> i32 { return x; }")
    span_start = len(source[:source.index("+")].encode("utf-8"))
    for _ in range(3):
        assert first.call("shift", 41) == second.call("shift", 40) == 42
        document = rejected(inv.TrapError, "INTEGER_OVERFLOW", lambda: first.call("shift", 2147483647))
        assert document["trap"]["span"] == {"start": span_start, "end": span_start + 1}
        assert document["diagnostics"][0]["path"] == str(path)
        assert second.call("shift", 0) == 2
        assert first.call("shift", 0) == 1


def test_scalar_artifact_real_python_process_survives_all_traps(scalar_engine):
    program = textwrap.dedent("""
        import json, pathlib, sys
        import invar_python as inv
        sys.path.insert(0, sys.argv[2])
        from scalar_cases import SOURCE, SOURCE_PATH, SUCCESS, TRAPS
        names = tuple(dict.fromkeys(case.name for case in (*SUCCESS, *TRAPS)))
        module = inv.compile_source(SOURCE, path=SOURCE_PATH, engine=sys.argv[1], exports=names)
        traps = 0
        for case in TRAPS:
            assert module.call("zero") == 4294967295
            try:
                module.call(case.name, *case.args)
            except inv.TrapError as error:
                assert error.diagnostics["trap"] == {"category": case.trap, "span": case.trap_span()}
                traps += 1
            else:
                raise AssertionError("missing source trap")
            assert module.call("signed_add", -2147483648, 2147483647) == -1
        print(json.dumps({"engine": sys.argv[1], "traps": traps, "recovered": True}))
    """)
    result = subprocess.run([sys.executable, "-c", program, scalar_engine,
                             str(Path(__file__).resolve().parent)],
                            capture_output=True, text=True, timeout=child_timeout(120))
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert json.loads(result.stdout) == {"engine": scalar_engine, "traps": 14, "recovered": True}
