"""Executed source oracle preparation; compiled-export checks await frozen API."""

import json
import pathlib

import invar_python as inv
import jsonschema
import pytest

from scalar_cases import SOURCE, SOURCE_PATH, SUCCESS, TRAPS


ROOT = pathlib.Path(__file__).resolve().parents[3]
SCHEMA = json.loads((ROOT / "schema/invar-diagnostics-v3.schema.json").read_text())


def oracle_trap(case):
    module = inv.compile_source(case.reference_source(), path=SOURCE_PATH)
    with pytest.raises(inv.TrapError) as caught:
        module.run_main()
    document = caught.value.diagnostics
    jsonschema.validate(document, SCHEMA)
    assert json.loads(caught.value.diagnostic_json) == document
    assert document["trap"] == {"category": case.trap, "span": case.trap_span()}
    first = document["diagnostics"][0]
    assert first["code"] == case.trap
    assert first["path"] == SOURCE_PATH
    assert {key: first["span"][key] for key in ("start", "end")} == case.trap_span()
    assert "result" not in document
    return document


@pytest.mark.parametrize("case", SUCCESS, ids=lambda case: case.label)
def test_scalar_source_oracle_success(case):
    module = inv.compile_source(case.reference_source(), path=SOURCE_PATH)
    assert module.run_main() == (case.expected if case.result_type == "i32" else 1)


@pytest.mark.parametrize("case", TRAPS, ids=lambda case: case.label)
def test_scalar_source_oracle_original_trap(case):
    oracle_trap(case)


def test_scalar_source_oracle_success_trap_success_survives():
    good = inv.compile_source(SUCCESS[0].reference_source(), path=SOURCE_PATH)
    assert good.run_main() == -1
    for case in TRAPS:
        oracle_trap(case)
        assert good.run_main() == -1


def test_scalar_source_inventory_without_entrypoint():
    functions = inv.compile_source(SOURCE, path=SOURCE_PATH).functions()
    assert functions["c_imports"] == []
    actual = {function["name"]: function for function in functions["source_functions"]}
    assert set(actual) == {case.name for case in (*SUCCESS, *TRAPS)}
    for case in (*SUCCESS, *TRAPS):
        function = actual[case.name]
        assert tuple(parameter["type"] for parameter in function["parameters"]) == case.types
        assert all(parameter["mode"] == "value" for parameter in function["parameters"])
        assert function["result"] == case.result_type


@pytest.mark.parametrize("name,valid,invalid", [
    ("signed_add", 1, True),
    ("signed_add", 1, 1.0),
    ("signed_add", 1, "1"),
    ("signed_add", 1, -2147483649),
    ("signed_add", 1, 2147483648),
    ("signed_add", 1, 1 << 100),
    ("unsigned_add", 1, True),
    ("unsigned_add", 1, 1.0),
    ("unsigned_add", 1, "1"),
    ("unsigned_add", 1, -1),
    ("unsigned_add", 1, 4294967296),
    ("unsigned_add", 1, 1 << 100),
])
def test_source_call_scalar_validation_precedes_execution(name, valid, invalid):
    module = inv.compile_source(SOURCE, path=SOURCE_PATH)
    with pytest.raises(inv.ValidationError) as caught:
        module.call(name, valid, invalid)
    jsonschema.validate(caught.value.diagnostics, SCHEMA)
    assert caught.value.diagnostics["diagnostics"][0]["code"] == "TYPE_MISMATCH"


def test_source_call_does_not_run_scalar_callbacks():
    module = inv.compile_source(SOURCE, path=SOURCE_PATH)

    class Index:
        called = False

        def __index__(self):
            self.called = True
            return 1

    class Integer(int):
        pass

    index = Index()
    for value in (index, Integer(1)):
        for name in ("signed_add", "unsigned_add"):
            with pytest.raises(inv.ValidationError) as caught:
                module.call(name, 1, value)
            jsonschema.validate(caught.value.diagnostics, SCHEMA)
            assert caught.value.diagnostics["diagnostics"][0]["code"] == "TYPE_MISMATCH"
    assert not index.called
