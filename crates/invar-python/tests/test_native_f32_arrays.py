"""HA-1.3 native float32 arrays, exact bits and the copying reference oracle."""
import invar_python as inv
import numpy as np
import pytest
from test_native_arrays import rejected

SOURCE = """
fn fcopy(xs: read Array<f32>, out: mut Array<f32>) -> u32 {
    for i in 0u32..len(xs) { out[i] = xs[i]; } return len(out);
}
fn half(xs: read Array<f32>, out: mut Array<f32>) -> u32 {
    for i in 0u32..len(xs) { out[i] = xs[i] * 0.5; } return len(out);
}
fn poke(xs: mut Array<f32>, i: u32) -> u32 { xs[0u32] = 1.0; xs[i] = -0.0; return len(xs); }
fn depth(n: u32) -> u32 { return depth(n + 1u32); }
fn fail(xs: mut Array<f32>) -> u32 { xs[0u32] = 2.0; return depth(0u32); }
fn pair(a: read Array<f32>, b: read Array<f32>) -> u32 { if a[0u32] == b[0u32] { return 1u32; } return 0u32; }
"""
BITS = [0x7f800001, 0xffc12345, 0x7fc54321, 0x80000000, 0,
        0x7f800000, 0xff800000, 1, 2, 0x007fffff, 0x00800000, 0x80800000]
NAMES = ["fcopy", "half", "poke", "fail", "pair"]


def floats(bits=BITS):
    # Copy creates an owning float32 array without converting payloads.
    return np.array(bits, np.uint32).view(np.float32).copy()


@pytest.fixture(scope="module")
def f32(scalar_engine):
    return inv.compile_source(SOURCE, path="native_f32.inv", engine=scalar_engine, exports=NAMES)


@pytest.fixture(scope="module")
def f32_oracle():
    return inv.compile_source(SOURCE, path="native_f32.inv", engine="reference", exports=NAMES)


@pytest.mark.parametrize("size", [0, 1, len(BITS), 1_000_000])
def test_f32_round_trip_same_address_payloads_empty_and_maximum(f32, f32_oracle, scalar_engine, size):
    xs = floats(np.resize(np.array(BITS, np.uint32), size))
    out = np.zeros(size, np.float32)
    expected = np.zeros(size, np.float32)
    addresses = [xs.ctypes.data, out.ctypes.data] if size else [0, 0]
    receipt = f32.call("fcopy", xs, out, fuel=10_000_000, observe=True)
    assert f32_oracle.call("fcopy", xs, expected, fuel=10_000_000) == size
    assert receipt["result"] == size and receipt["engine"] == scalar_engine
    assert out.tobytes() == expected.tobytes() == xs.tobytes()
    assert receipt["copying"] is (scalar_engine == "reference")
    assert receipt["record_addresses"] == (addresses if scalar_engine == "native-serial" else [])
    assert out.flags.owndata and out.base is None
    descriptor = next(d for d in f32.exports() if d["name"] == "fcopy")
    assert [(p["type"], p["mode"]) for p in descriptor["parameters"]] == [("Array<f32>", "read"), ("Array<f32>", "mut")]
    assert all(p["zero_copy"] is (scalar_engine == "native-serial") for p in descriptor["parameters"])


def test_f32_ieee_arithmetic_does_not_flush_subnormals(f32, f32_oracle):
    xs = floats(BITS[3:])
    out, expected = np.zeros(xs.size, np.float32), np.zeros(xs.size, np.float32)
    assert f32.call("half", xs, out) == xs.size
    f32_oracle.call("half", xs, expected)
    assert out.tobytes() == expected.tobytes()
    assert out.view(np.uint32).tolist() == [0x80000000, 0, 0x7f800000, 0xff800000, 0, 1, 0x00400000, 0x00400000, 0x80400000]


def test_f32_all_fuel_prefixes_restore_and_equal_reference(f32, f32_oracle):
    outcomes = set()
    for fuel in range(1, 150):
        results = []
        for module in (f32, f32_oracle):
            xs, out = floats(), floats(list(reversed(BITS)))
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
def test_f32_trap_and_tool_failure_restore_payloads(f32, f32_oracle, name, index, error_class, code):
    documents = []
    for module in (f32, f32_oracle):
        xs = floats()
        args = (xs,) if index is None else (xs, index)
        documents.append(rejected(error_class, code, lambda: module.call(name, *args, fuel=10000), xs))
    assert documents[0] == documents[1]


@pytest.mark.parametrize("bad,code", [
    (np.zeros(3, np.float64), "PY_ARRAY_DTYPE"),
    (np.zeros(3, np.float16), "PY_ARRAY_DTYPE"),
    (np.zeros(3, np.uint32), "PY_ARRAY_DTYPE"),
    (np.zeros((3, 1), np.float32), "PY_ARRAY_RANK"),
    (np.zeros(6, np.float32)[::2], "PY_ARRAY_CONTIGUITY"),
    (np.frombuffer(bytearray(17), np.float32, 4, offset=1), "PY_ARRAY_ALIGNMENT"),
    (np.zeros(6, np.float32)[1:], "PY_ARRAY_OWNERSHIP"),
    (np.zeros(1_000_001, np.float32), "PY_BUFFER_LIMIT"),
])
def test_f32_bad_buffers_refused_before_writes(f32, bad, code):
    out = floats()
    rejected(inv.ValidationError, code, lambda: f32.call("fcopy", bad, out), bad, out)


def test_f32_mut_alias_readonly_and_empty_trap_are_typed(f32):
    xs = floats()
    frozen = floats()
    frozen.flags.writeable = False
    rejected(inv.ValidationError, "PY_ARRAY_WRITABILITY", lambda: f32.call("fcopy", xs, frozen), xs, frozen)
    rejected(inv.ValidationError, "PY_BUFFER_ALIAS", lambda: f32.call("fcopy", xs, xs), xs)
    rejected(inv.TrapError, "INDEX_OUT_OF_BOUNDS", lambda: f32.call("poke", np.zeros(0, np.float32), 0))
    shared = np.array([1.0], np.float32)
    assert f32.call("pair", shared, shared) == 1


def test_f32_take_and_returned_arrays_remain_refused(scalar_engine):
    for source in ["fn f(xs: take Array<f32>) -> u32 { return len(xs); }",
                   "fn f(xs: read Array<f32>) -> Array<f32> { return filled_f32(len(xs), 0.0); }"]:
        rejected(inv.EngineError, "NATIVE_ABI", lambda: inv.compile_source(source, engine=scalar_engine, exports=["f"]))
