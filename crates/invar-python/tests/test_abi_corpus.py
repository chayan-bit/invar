"""L1-06 reference equivalence on the corpus subset expressible through the ABI.

Every checked `.inv` program under tests/fixtures, bench and examples is
scanned. A function is expressible when each parameter is an exact i32/u32
value or a read/mut Array<i32|u32|f32> and its result is i32/u32 (byte strings
cross only through C imports, which have their own suite). Expressible
functions are exported natively; those whose bodies the native export
fragment refuses are counted, never silently dropped. Each admitted export is
called on deterministic inputs with native-serial and with the reference
engine, and the outcomes and mut arrays must be identical.
Set INVAR_CORPUS_SUMMARY to a path to record the subset counts as JSON.
"""
import json
import os
import pathlib

import invar_python as inv
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCALARS = {"i32", "u32"}
ARRAYS = {"Array<i32>": np.int32, "Array<u32>": np.uint32, "Array<f32>": np.float32}
FUEL = 200_000


def expressible(function):
    if function["result"] not in SCALARS:
        return False
    for param in function["parameters"]:
        if param["mode"] == "value" and param["type"] in SCALARS:
            continue
        if param["mode"] in ("read", "mut") and param["type"] in ARRAYS:
            continue
        return False
    return True


def inputs(function, seed):
    values = []
    for param in function["parameters"]:
        if param["type"] in SCALARS:
            values.append(seed)
        else:
            values.append(np.array([3, 1, 2, seed], ARRAYS[param["type"]]))
    return values


def outcome(module, name, values):
    try:
        result = ("ok", module.call(name, *values, fuel=FUEL))
    except inv.InvarError as error:
        document = error.diagnostics
        result = ("error", type(error).__name__, document["diagnostics"][0]["code"], document.get("trap"))
    return result, [v.tobytes() for v in values if isinstance(v, np.ndarray)]


def corpus():
    for base in ("tests/fixtures", "bench", "tools/bench", "examples"):
        yield from sorted((ROOT / base).rglob("*.inv"))


def test_native_exports_equal_the_reference_on_the_abi_corpus_subset(scalar_engine):
    if scalar_engine != "native-serial":
        pytest.skip("reference-equivalence of native exports requires --scalar-engine native-serial")
    counts = {"files": 0, "checked_files": 0, "functions": 0, "expressible": 0,
              "admitted": 0, "refused_by_native_fragment": 0, "compared_calls": 0,
              "f32_expressible": 0, "f32_admitted": 0, "f32_compared_calls": 0}
    refused = {}
    for path in corpus():
        counts["files"] += 1
        source = path.read_bytes()
        relative = str(path.relative_to(ROOT))
        try:
            module = inv.compile_source(source, path=relative)
        except inv.InvarError:
            continue
        counts["checked_files"] += 1
        functions = module.functions()["source_functions"]
        counts["functions"] += len(functions)
        for function in filter(expressible, functions):
            counts["expressible"] += 1
            has_f32 = any(p["type"] == "Array<f32>" for p in function["parameters"])
            counts["f32_expressible"] += int(has_f32)
            name = function["name"]
            try:
                native = inv.compile_source(source, path=relative, engine="native-serial", exports=[name])
            except inv.EngineError as error:
                assert error.diagnostics["diagnostics"][0]["code"] in ("NATIVE_ABI", "NATIVE_BACKEND")
                counts["refused_by_native_fragment"] += 1
                refused[f"{relative}::{name}"] = error.diagnostics["diagnostics"][0]["message"]
                continue
            counts["admitted"] += 1
            counts["f32_admitted"] += int(has_f32)
            reference = inv.compile_source(source, path=relative, engine="reference", exports=[name])
            for seed in (0, 1, 2):
                expected = outcome(reference, name, inputs(function, seed))
                actual = outcome(native, name, inputs(function, seed))
                assert actual == expected, f"{relative}::{name} seed {seed}"
                counts["compared_calls"] += 1
                counts["f32_compared_calls"] += int(has_f32)
    summary = {"counts": counts, "refused": refused}
    if os.environ.get("INVAR_CORPUS_SUMMARY"):
        pathlib.Path(os.environ["INVAR_CORPUS_SUMMARY"]).write_text(json.dumps(summary, indent=1, sort_keys=True))
    print("L1-06 ABI corpus subset:", json.dumps(counts, sort_keys=True))
    assert counts["admitted"] > 0 and counts["compared_calls"] == 3 * counts["admitted"]

    assert counts["f32_admitted"] > 0
    assert counts["f32_compared_calls"] == 3 * counts["f32_admitted"]
