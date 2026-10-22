"""HA-1.4 native f64 and Bool arrays: exact bits, rollback and the copying reference oracle (#237)."""
import invar_python as inv
import numpy as np
import pytest
from test_native_arrays import rejected

SOURCE = """
fn fcopy(xs: read Array<f64>, out: mut Array<f64>) -> u32 {
    for i in 0u32..len(xs) { out[i] = xs[i]; } return len(out);
}
fn half(xs: read Array<f64>, out: mut Array<f64>) -> u32 {
    for i in 0u32..len(xs) { out[i] = xs[i] * 0.5; } return len(out);
}
fn poke(xs: mut Array<f64>, i: u32) -> u32 { xs[0u32] = 1.0; xs[i] = -0.0; return len(xs); }
fn depth(n: u32) -> u32 { return depth(n + 1u32); }
fn fail(xs: mut Array<f64>) -> u32 { xs[0u32] = 2.0; return depth(0u32); }
fn positive(xs: read Array<f64>, out: mut Array<Bool>) -> u32 {
    for i in 0u32..len(xs) { out[i] = xs[i] > 0.0; } return len(out);
}
fn count(flags: read Array<Bool>) -> u32 {
    var n: u32 = 0u32; for i in 0u32..len(flags) { if flags[i] { n = n + 1u32; } } return n;
}
fn invert(flags: mut Array<Bool>) -> u32 {
    for i in 0u32..len(flags) { flags[i] = not flags[i]; } return len(flags);
}
fn mark_fail(flags: mut Array<Bool>) -> u32 { flags[0u32] = true; return depth(0u32); }
fn pair(a: read Array<f64>, b: read Array<f64>) -> u32 { if a[0u32] == b[0u32] { return 1u32; } return 0u32; }
"""
BITS = [0x7FF0000000000001, 0xFFF8000000012345, 0x7FF8000000054321, 0x8000000000000000, 0,
        0x7FF0000000000000, 0xFFF0000000000000, 1, 2, 0x000FFFFFFFFFFFFF, 0x0010000000000000, 0x8010000000000000]
NAMES = ["fcopy", "half", "poke", "fail", "positive", "count", "invert", "mark_fail", "pair"]


def doubles(bits=BITS):
    # Copy creates an owning float64 array without converting payloads.
    return np.array(bits, np.uint64).view(np.float64).copy()


@pytest.fixture(scope="module")
def mod(scalar_engine):
    return inv.compile_source(SOURCE, path="native_f64.inv", engine=scalar_engine, exports=NAMES)


@pytest.fixture(scope="module")
def oracle():
    return inv.compile_source(SOURCE, path="native_f64.inv", engine="reference", exports=NAMES)


@pytest.mark.parametrize("size", [0, 1, len(BITS), 1_000_000])
def test_f64_round_trip_same_address_payloads_empty_and_maximum(mod, oracle, scalar_engine, size):
    xs = doubles(np.resize(np.array(BITS, np.uint64), size))
    out = np.zeros(size, np.float64)
    expected = np.zeros(size, np.float64)
    addresses = [xs.ctypes.data, out.ctypes.data] if size else [0, 0]
    receipt = mod.call("fcopy", xs, out, fuel=10_000_000, observe=True)
    assert oracle.call("fcopy", xs, expected, fuel=10_000_000) == size
    assert receipt["result"] == size and receipt["engine"] == scalar_engine
    assert out.tobytes() == expected.tobytes() == xs.tobytes()
    assert receipt["copying"] is (scalar_engine == "reference")
    assert receipt["record_addresses"] == (addresses if scalar_engine == "native-serial" else [])
    descriptor = next(d for d in mod.exports() if d["name"] == "fcopy")
    assert [(p["type"], p["mode"]) for p in descriptor["parameters"]] == [("Array<f64>", "read"), ("Array<f64>", "mut")]
    assert all(p["zero_copy"] is (scalar_engine == "native-serial") for p in descriptor["parameters"])


def test_f64_ieee_arithmetic_does_not_flush_subnormals(mod, oracle):
    xs = doubles(BITS[3:])
    out, expected = np.zeros(xs.size, np.float64), np.zeros(xs.size, np.float64)
    assert mod.call("half", xs, out) == xs.size
    oracle.call("half", xs, expected)
    assert out.tobytes() == expected.tobytes()
    # Ties round to even: 2^-1074 / 2 is zero, the largest subnormal and the smallest normal halve to 2^-1023.
    assert out.view(np.uint64).tolist() == [0x8000000000000000, 0, 0x7FF0000000000000, 0xFFF0000000000000, 0, 1,
                                            0x0008000000000000, 0x0008000000000000, 0x8008000000000000]


def test_f64_all_fuel_prefixes_restore_and_equal_reference(mod, oracle):
    outcomes = set()
    for fuel in range(1, 150):
        results = []
        for module in (mod, oracle):
            xs, out = doubles(), doubles(list(reversed(BITS)))
            before = out.tobytes()
            try:
                result = module.call("fcopy", xs, out, fuel=fuel)
                assert out.tobytes() == xs.tobytes()
                results.append(("ok", result, out.tobytes()))
            except inv.ResourceError as error:
                assert out.tobytes() == before
                results.append(("limit", error.diagnostics, out.tobytes()))
        assert results[0] == results[1], f"fuel={fuel}"
        outcomes.add(results[0][0])
    assert outcomes == {"ok", "limit"}


@pytest.mark.parametrize("name,index,error_class,code", [
    ("poke", 99, inv.TrapError, "INDEX_OUT_OF_BOUNDS"),
    ("fail", None, inv.ResourceError, "CALL_DEPTH_LIMIT"),
])
def test_f64_trap_and_tool_failure_restore_payloads(mod, oracle, name, index, error_class, code):
    documents = []
    for module in (mod, oracle):
        xs = doubles()
        args = (xs,) if index is None else (xs, index)
        documents.append(rejected(error_class, code, lambda: module.call(name, *args, fuel=10000), xs))
    assert documents[0] == documents[1]


def test_bool_arrays_round_trip_and_match_reference(mod, oracle, scalar_engine):
    xs = np.array([1.0, -1.0, 0.0, 2.5, -0.0], np.float64)
    out, expected = np.zeros(xs.size, np.bool_), np.zeros(xs.size, np.bool_)
    receipt = mod.call("positive", xs, out, observe=True)
    oracle.call("positive", xs, expected)
    assert out.tolist() == expected.tolist() == [True, False, False, True, False]
    assert receipt["copying"] is (scalar_engine == "reference")
    flags = np.array([True, False, True, True], np.bool_)
    assert mod.call("count", flags) == oracle.call("count", flags) == 3
    assert mod.call("invert", flags) == 4
    assert flags.tolist() == [False, True, False, False]
    empty = np.zeros(0, np.bool_)
    assert mod.call("count", empty) == 0 and mod.call("invert", empty) == 0
    descriptor = next(d for d in mod.exports() if d["name"] == "positive")
    assert [(p["type"], p["mode"]) for p in descriptor["parameters"]] == [("Array<f64>", "read"), ("Array<Bool>", "mut")]


def test_bool_failure_restores_the_pre_call_bytes(mod, oracle):
    documents = []
    for module in (mod, oracle):
        flags = np.array([False, True, False], np.bool_)
        documents.append(rejected(inv.ResourceError, "CALL_DEPTH_LIMIT",
                                  lambda: module.call("mark_fail", flags, fuel=10000), flags))
    assert documents[0] == documents[1]


@pytest.mark.parametrize("byte", [2, 3, 128, 255])
def test_bool_bytes_other_than_zero_or_one_are_refused(mod, byte):
    flags = np.array([False, True, False])
    flags.view(np.uint8)[2] = byte
    rejected(inv.ValidationError, "PY_ARRAY_BOOL_VALUE", lambda: mod.call("count", flags), flags)
    out = np.array([False, True])
    out.view(np.uint8)[1] = byte
    rejected(inv.ValidationError, "PY_ARRAY_BOOL_VALUE", lambda: mod.call("invert", out), out)


@pytest.mark.parametrize("bad,code", [
    (np.zeros(3, np.float32), "PY_ARRAY_DTYPE"),
    (np.zeros(3, np.int64), "PY_ARRAY_DTYPE"),
    (np.zeros((3, 1), np.float64), "PY_ARRAY_RANK"),
    (np.zeros(6, np.float64)[::2], "PY_ARRAY_CONTIGUITY"),
    (np.frombuffer(bytearray(33), np.float64, 4, offset=1), "PY_ARRAY_ALIGNMENT"),
    (np.frombuffer(bytearray(36), np.float64, 4, offset=4), "PY_ARRAY_ALIGNMENT"),
    (np.zeros(6, np.float64)[1:], "PY_ARRAY_OWNERSHIP"),
    (np.zeros(1_000_001, np.float64), "PY_BUFFER_LIMIT"),
])
def test_f64_bad_buffers_refused_before_writes(mod, bad, code):
    out = doubles()
    rejected(inv.ValidationError, code, lambda: mod.call("fcopy", bad, out), bad, out)


def test_f64_mut_alias_readonly_and_empty_trap_are_typed(mod):
    xs = doubles()
    frozen = doubles()
    frozen.flags.writeable = False
    rejected(inv.ValidationError, "PY_ARRAY_WRITABILITY", lambda: mod.call("fcopy", xs, frozen), xs, frozen)
    rejected(inv.ValidationError, "PY_BUFFER_ALIAS", lambda: mod.call("fcopy", xs, xs), xs)
    rejected(inv.TrapError, "INDEX_OUT_OF_BOUNDS", lambda: mod.call("poke", np.zeros(0, np.float64), 0))
    shared = np.array([1.0], np.float64)
    assert mod.call("pair", shared, shared) == 1


def test_scalar_f64_and_returned_arrays_remain_refused(scalar_engine):
    for source in ["fn f(x: f64) -> u32 { return 0u32; }",
                   "fn f(xs: read Array<f64>) -> f64 { return xs[0u32]; }",
                   "fn f(xs: read Array<Bool>) -> Array<Bool> { var b: Array<Bool> = [true]; return b; }",
                   "fn f(xs: take Array<f64>) -> u32 { return len(xs); }"]:
        rejected(inv.EngineError, "NATIVE_ABI", lambda: inv.compile_source(source, engine=scalar_engine, exports=["f"]))
