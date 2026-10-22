"""HA-1.2 array exports through CheckedModule.call (L1-06).

Run with --scalar-engine reference and --scalar-engine native-serial. The
native-serial engine runs at the NumPy address (zero-copy) and restores mut
arrays on failure; the reference engine is the copying oracle. Every refusal
is typed, carries schema-valid AI-01 JSON and leaves inputs byte-identical.
"""
import gc
import json

import invar_python as inv
import jsonschema
import numpy as np
import pytest

from test_scalar_acceptance import SCHEMA

SOURCE = """\
fn summed(xs: read Array<i32>) -> i32 { var acc: i32 = 0; for i in 0u32..len(xs) { acc = acc + xs[i]; } return acc; }
fn scale(xs: read Array<u32>, out: mut Array<u32>, k: u32) -> u32 { for i in 0u32..len(xs) { out[i] = xs[i] * k; } return len(out); }
fn negate(xs: read Array<i32>, out: mut Array<i32>) -> u32 { for i in 0u32..len(xs) { out[i] = 0 - xs[i]; } return len(xs); }
fn poke(xs: mut Array<i32>, i: u32, v: i32) -> i32 { xs[0u32] = v; xs[i] = v + 1; return xs[0u32]; }
fn count(xs: read Array<i32>) -> u32 { return len(xs); }
fn pair(a: read Array<i32>, b: read Array<i32>) -> i32 { return a[0u32] + b[0u32]; }
fn bump(xs: mut Array<u32>) -> u32 { for i in 0u32..len(xs) { xs[i] = xs[i] + 1u32; } return len(xs); }
"""
NAMES = ["summed", "scale", "negate", "poke", "count", "pair", "bump"]


@pytest.mark.parametrize("names", [("first", "second", "plain", "peer"), ("peer", "second", "plain", "first")])
def test_reordered_same_signature_exports_call_named_symbol(scalar_engine, names):
    module = inv.compile_source(
        "fn first(xs: read Array<i32>) -> i32 { return xs[0u32] + 100; } "
        "fn second(xs: read Array<i32>) -> i32 { return xs[0u32] + 101; } "
        "fn plain() -> i32 { return 102; } fn peer() -> i32 { return 103; }",
        engine=scalar_engine, exports=list(names), path="reordered.inv",
    )
    descriptors = module.exports()
    assert [(item["name"], item["symbol"]) for item in descriptors] == [
        (name, f"inv_export_{ordinal}") for ordinal, name in enumerate(names)
    ]
    assert all(item["artifact_id"] == module.artifact_id for item in descriptors)
    source = np.zeros(1, np.int32)
    for name, expected in [("first", 100), ("second", 101)]:
        assert module.call(name, source) == expected
    assert module.call("plain") == 102
    assert module.call("peer") == 103


@pytest.fixture(scope="module")
def arrays(scalar_engine):
    return inv.compile_source(SOURCE, path="native_arrays.inv", engine=scalar_engine, exports=NAMES)


@pytest.fixture(scope="module")
def oracle():
    return inv.compile_source(SOURCE, path="native_arrays.inv", engine="reference", exports=NAMES)


def rejected(error_class, code, callback, *arrays):
    before = [a.tobytes() for a in arrays]
    with pytest.raises(error_class) as caught:
        callback()
    document = caught.value.diagnostics
    jsonschema.validate(document, SCHEMA)
    assert json.loads(caught.value.diagnostic_json) == document
    assert document["diagnostics"][0]["code"] == code
    assert [a.tobytes() for a in arrays] == before, "inputs must be byte-identical"
    return document


def test_array_descriptors_report_modes_and_the_copy_path(arrays, scalar_engine):
    scale = next(d for d in arrays.exports() if d["name"] == "scale")
    assert [(p["type"], p["mode"]) for p in scale["parameters"]] == [
        ("Array<u32>", "read"), ("Array<u32>", "mut"), ("u32", "value")]
    zero_copy = scalar_engine == "native-serial"
    assert all(p["zero_copy"] is zero_copy for p in scale["parameters"][:2])
    caps = inv.capabilities()
    assert caps["numeric_array_abi"] is False, "the public C ABI is unchanged"
    assert caps["array_exports"] == ["read Array<i32>", "read Array<u32>", "mut Array<i32>", "mut Array<u32>", "read Array<f32>", "mut Array<f32>",
                                     "read Array<f64>", "mut Array<f64>", "read Array<Bool>", "mut Array<Bool>"]
    assert caps["gil_released"] is False


@pytest.mark.parametrize("dtype,name", [(np.int32, "negate"), (np.uint32, "scale")])
@pytest.mark.parametrize("size", [0, 1, 7, 1_000_000])
def test_i32_u32_arrays_round_trip_with_caller_outputs(arrays, oracle, dtype, name, size):
    info = np.iinfo(dtype)
    source = np.linspace(info.min // 4, info.max // 4, size).astype(dtype)
    out = np.zeros(size, dtype)
    expected = np.zeros(size, dtype)
    args = (source, out, 3) if name == "scale" else (source, out)
    oracle_args = (source, expected, 3) if name == "scale" else (source, expected)
    assert arrays.call(name, *args, fuel=10_000_000) == size
    assert oracle.call(name, *oracle_args, fuel=10_000_000) == size
    assert out.tobytes() == expected.tobytes()
    if name == "scale":
        assert np.array_equal(out, (source.astype(np.uint64) * 3 % 2**32).astype(np.uint32))
    else:
        assert np.array_equal(out, -source)


def test_zero_copy_same_buffer_address_on_both_sides(arrays, scalar_engine):
    source = np.arange(1, 6, dtype=np.uint32)
    out = np.zeros(5, np.uint32)
    addresses = (source.ctypes.data, out.ctypes.data)
    observed = arrays.call("scale", source, out, 10, observe=True)
    assert observed["result"] == 5 and observed["engine"] == scalar_engine
    assert out.tolist() == [10, 20, 30, 40, 50]
    assert (out.ctypes.data, out.flags.owndata, out.base) == (addresses[1], True, None)
    if scalar_engine == "native-serial":
        # The private records handed to native code carry the NumPy buffers' own
        # addresses: native code read and wrote the caller's memory in place.
        assert observed["copying"] is False
        assert observed["record_addresses"] == list(addresses)
    else:
        assert observed["copying"] is True and observed["record_addresses"] == []


def test_empty_arrays_never_dereference_and_report_null(arrays, scalar_engine):
    empty = np.zeros(0, np.int32)
    address = empty.ctypes.data
    observed = arrays.call("count", empty, observe=True)
    assert observed["result"] == 0
    if scalar_engine == "native-serial":
        assert observed["record_addresses"] == [0], "an empty record carries NULL"
    assert empty.ctypes.data == address
    rejected(inv.TrapError, "INDEX_OUT_OF_BOUNDS", lambda: arrays.call("poke", empty, 0, 1), empty)


def test_trap_restores_mut_arrays_bit_exactly(arrays, oracle):
    for module in (arrays, oracle):
        xs = np.array([10, 20, 30], np.int32)
        document = rejected(inv.TrapError, "INDEX_OUT_OF_BOUNDS", lambda: module.call("poke", xs, 3, -4), xs)
        start, end = document["trap"]["span"]["start"], document["trap"]["span"]["end"]
        assert SOURCE.encode()[start:end] == b"xs[i]"
        assert module.call("poke", xs, 2, -4) == -4
        assert xs.tolist() == [-4, 20, -3]


def test_fuel_exhaustion_restores_mut_arrays(arrays, oracle):
    for fuel in range(1, 30):
        results = []
        for module in (arrays, oracle):
            xs = np.array([1, 2, 3, 4], np.uint32)
            try:
                results.append(("ok", module.call("bump", xs, fuel=fuel), xs.tolist()))
            except inv.ResourceError as error:
                assert xs.tolist() == [1, 2, 3, 4], "no partial output"
                results.append(("limit", error.diagnostics["diagnostics"][0]["code"], xs.tolist()))
        assert results[0] == results[1], f"fuel {fuel}"


def test_exact_read_aliases_are_admitted(arrays):
    shared = np.array([4, 5], np.int32)
    assert arrays.call("pair", shared, shared) == 8


@pytest.mark.parametrize("bad,code", [
    (np.zeros(3, np.int64), "PY_ARRAY_DTYPE"),
    (np.zeros(3, np.uint32), "PY_ARRAY_DTYPE"),
    (np.zeros((3, 1), np.int32), "PY_ARRAY_RANK"),
    (np.zeros(6, np.int32)[::2], "PY_ARRAY_CONTIGUITY"),
    (np.frombuffer(bytearray(17), np.int32, 4, offset=1), "PY_ARRAY_ALIGNMENT"),
    (np.zeros(6, np.int32)[1:], "PY_ARRAY_OWNERSHIP"),
    (np.zeros(1_000_001, np.int32), "PY_BUFFER_LIMIT"),
])
def test_invalid_input_arrays_are_refused_before_execution(arrays, bad, code):
    rejected(inv.ValidationError, code, lambda: arrays.call("summed", bad), bad)


def test_read_only_outputs_and_forbidden_aliases_are_refused(arrays):
    source = np.arange(4, dtype=np.int32)
    frozen = np.zeros(4, np.int32)
    frozen.flags.writeable = False
    rejected(inv.ValidationError, "PY_ARRAY_WRITABILITY", lambda: arrays.call("negate", source, frozen), source, frozen)
    rejected(inv.ValidationError, "PY_BUFFER_ALIAS", lambda: arrays.call("negate", source, source), source)


def test_unknown_names_arity_and_scalars_are_typed(arrays):
    xs = np.zeros(2, np.int32)
    rejected(inv.ValidationError, "NAME_UNKNOWN", lambda: arrays.call("absent", xs), xs)
    rejected(inv.ValidationError, "PY_ARGUMENT_COUNT", lambda: arrays.call("poke", xs, 0), xs)
    rejected(inv.ValidationError, "TYPE_MISMATCH", lambda: arrays.call("poke", xs, -1, 0), xs)
    rejected(inv.ValidationError, "TYPE_MISMATCH", lambda: arrays.call("poke", xs, 0, 2**31), xs)
    rejected(inv.ValidationError, "PY_ARRAY_DTYPE", lambda: arrays.call("summed", [1, 2]))


@pytest.mark.parametrize("source", [
    "fn f(xs: take Array<i32>) -> u32 { return len(xs); }",
    "fn f(xs: read Array<i32>) -> Array<i32> { var b = filled_i32(len(xs), 0); return b; }",
])
def test_take_and_returned_arrays_are_not_exported(source, scalar_engine):
    rejected(inv.EngineError, "NATIVE_ABI",
             lambda: inv.compile_source(source, engine=scalar_engine, exports=["f"]))


def test_f32_and_returned_arrays_use_the_copying_reference_path():
    module = inv.compile_source(
        "fn fcopy(xs: read Array<f32>, out: mut Array<f32>) -> u32 { for i in 0u32..len(xs) { out[i] = xs[i]; } return len(xs); }\n"
        "fn fresh(ys: read Array<f32>) -> Array<f32> { var b = filled_f32(len(ys), 0.0); for i in 0u32..len(ys) { b[i] = ys[i]; } return b; }",
        path="f32.inv")
    for size in (0, 1, 1000):
        xs = np.linspace(-1, 1, size, dtype=np.float32)
        out = np.full(size, np.nan, np.float32)
        observed = module.call_reference("fcopy", xs, out, fuel=1_000_000)
        assert observed["copying"] is True and observed["result"] == size
        assert out.tobytes() == xs.tobytes()
        fresh = module.call_reference("fresh", xs, fuel=1_000_000)["result"]
        assert fresh.dtype == np.float32 and fresh.tobytes() == xs.tobytes()


def test_arrays_survive_the_call_and_can_be_resized_afterwards(arrays):
    xs = np.zeros(3, np.uint32)
    assert arrays.call("bump", xs) == 3
    xs.resize(5, refcheck=False)
    del xs
    gc.collect()
    ys = np.zeros(2, np.uint32)
    assert arrays.call("bump", ys) == 2 and ys.tolist() == [1, 1]
