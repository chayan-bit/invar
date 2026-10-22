"""HA-1.1 explicit reference-engine array entry point: CheckedModule.call_reference.

Every refusal is checked for its exception class and code, steps_used == 0 and
byte-identical NumPy inputs. Non-normal outcomes keep mut inputs unchanged; a
returned Result::Err commits like any normal return. The native `call` path
keeps refusing array functions.
"""
import hashlib
import os
import sys
import threading
import time

import numpy as np
import pytest

import invar_python as ip

SOURCE = b"""\
fn store2(xs: mut Array<i32>, i: u32, v: i32) -> i32 { xs[0u32] = xs[0u32] + 1; xs[i] = v; return xs[i]; }
fn div_after(xs: mut Array<i32>, d: i32) -> i32 { xs[0u32] = 1; return xs[1u32] / d; }
fn over(xs: mut Array<i32>) -> i32 { xs[0u32] = 2147483647; xs[1u32] = xs[0u32] + 1; return 0; }
fn both(xs: mut Array<i32>, ys: read Array<i32>) -> i32 { xs[0u32] = ys[0u32]; return 0; }
fn twice(xs: mut Array<i32>, ys: mut Array<i32>) -> i32 { xs[0u32] = 1; ys[0u32] = 2; return 0; }
fn reads(a: read Array<i32>, b: read Array<i32>) -> i32 { return a[0u32] + b[0u32]; }
fn settle(xs: mut Array<i32>, k: i32) -> Result<i32, u32> { xs[0u32] = k; if k < 0 { return Result::Err(7u32); } return Result::Ok(k); }
fn flag(xs: mut Array<u32>, on: Bool) -> Result<Bool, i32> { xs[0u32] = 1u32; if on { return Result::Ok(true); } return Result::Err(0 - 3); }
fn fcopy(xs: mut Array<f32>, ys: read Array<f32>, i: u32) -> f32 { xs[0u32] = ys[0u32]; xs[i] = ys[1u32]; return xs[0u32]; }
fn fresh(ys: read Array<f32>) -> Array<f32> { var b = filled_f32(len(ys), 0.0); for i in 0u32..len(ys) { b[i] = ys[i]; } return b; }
fn scale(xs: mut Array<f32>, k: f32) -> f32 { xs[0u32] = xs[0u32] * k; return xs[0u32]; }
fn eat(xs: take Array<i32>) -> u32 { return len(xs); }
fn grow_after(xs: mut Array<i32>, ys: read Array<i32>) -> u32 { xs[0u32] = 7; let t = filled_u32(1u32, 0u32); return len(t) + len(ys); }
fn spin(xs: mut Array<u32>, n: u32) -> u32 { for i in 0u32..n { xs[0u32] = xs[0u32] + 1u32; } return xs[0u32]; }
fn plain(a: i32, b: i32) -> i32 { return a + b; }
fn near_f(a: read Array<f32>, b: read Array<f32>) -> u32 { let t = filled_u32(1u32, 0u32); return len(a) + len(b) + len(t); }
fn near_i(a: read Array<i32>, b: read Array<i32>) -> u32 { let t = filled_u32(1u32, 0u32); return len(a) + len(b) + len(t); }
fn three_f(a: read Array<f32>, b: read Array<f32>, c: read Array<f32>) -> u32 { return len(a) + len(b) + len(c); }
fn three_i(a: read Array<i32>, b: read Array<i32>, c: read Array<i32>) -> u32 { return len(a) + len(b) + len(c); }
fn fmulti(a: mut Array<f32>, b: mut Array<f32>, c: mut Array<i32>, r: read Array<f32>, i: u32) -> f32 { a[0u32] = r[0u32]; a[1u32] = 0.0 * (0.0 - 1.0); b[0u32] = r[1u32]; c[0u32] = 9; b[i] = r[0u32]; return b[0u32]; }
"""


@pytest.fixture()
def module():
    return ip.compile_source(SOURCE, path="reference_arrays.inv")


def code(error):
    return error.diagnostics["diagnostics"][0]["code"]


def refused(module, cls, expected, name, *args, **kwargs):
    arrays = [a for a in args if isinstance(a, np.ndarray)]
    before = [a.tobytes() for a in arrays]
    with pytest.raises(cls) as raised:
        module.call_reference(name, *args, **kwargs)
    assert code(raised.value) == expected
    assert raised.value.steps_used == 0, "refused before execution"
    assert [a.tobytes() for a in arrays] == before, "byte-identical inputs"
    return raised.value


def test_loaded_extension_is_this_build():
    # Old installed extensions lack call_reference; HA1PY_EXPECT_SO/SHA256 pin
    # the exact file when set by the runner.
    assert hasattr(ip.CheckedModule, "call_reference")
    expected = os.environ.get("HA1PY_EXPECT_SO")
    if expected:
        assert os.path.realpath(ip.__file__) == os.path.realpath(expected)
        digest = hashlib.sha256(open(ip.__file__, "rb").read()).hexdigest()
        assert digest == os.environ["HA1PY_EXPECT_SHA256"]


# --- pre-execution refusals: zero steps, unchanged buffers -----------------

@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.bool_, np.uint8])
def test_wrong_dtype(module, dtype):
    refused(module, ip.ValidationError, "PY_ARRAY_DTYPE", "store2", np.zeros(3, dtype), 0, 1)


def test_non_ndarray_and_subclass(module):
    refused(module, ip.ValidationError, "PY_ARRAY_DTYPE", "store2", [1, 2, 3], 0, 1)
    sub = np.zeros(3, np.int32).view(np.matrix)
    refused(module, ip.ValidationError, "PY_ARRAY_DTYPE", "store2", sub, 0, 1)


def test_rank(module):
    refused(module, ip.ValidationError, "PY_ARRAY_RANK", "store2", np.zeros((2, 2), np.int32), 0, 1)


def test_non_contiguous_view(module):
    a = np.arange(8, dtype=np.int32)
    refused(module, ip.ValidationError, "PY_ARRAY_CONTIGUITY", "store2", a[::2], 0, 1)


def test_misaligned_view(module):
    raw = np.frombuffer(bytearray(4 * 4 + 1), dtype=np.int32, offset=1)
    refused(module, ip.ValidationError, "PY_ARRAY_ALIGNMENT", "reads", raw, raw)


def test_read_only_array_as_mut_and_as_read(module):
    a = np.array([1, 2, 3], np.int32)
    a.setflags(write=False)
    refused(module, ip.ValidationError, "PY_ARRAY_WRITABILITY", "store2", a, 0, 1)
    assert module.call_reference("reads", a, a)["result"] == 2


def test_overlapping_views_are_not_owners(module):
    a = np.arange(10, dtype=np.int32)
    refused(module, ip.ValidationError, "PY_ARRAY_OWNERSHIP", "both", a[1:5], a[3:9])


def test_same_array_mut_and_read_or_twice_mut(module):
    a = np.arange(3, dtype=np.int32)
    refused(module, ip.ValidationError, "PY_BUFFER_ALIAS", "both", a, a)
    refused(module, ip.ValidationError, "PY_BUFFER_ALIAS", "twice", a, a)


def test_oversized_array(module):
    refused(module, ip.ValidationError, "PY_BUFFER_LIMIT", "store2", np.zeros(1_000_001, np.int32), 0, 1)


def test_take_and_f32_scalar_and_scalar_types(module):
    refused(module, ip.ValidationError, "PY_TAKE_UNAVAILABLE", "eat", np.zeros(2, np.int32))
    refused(module, ip.EngineError, "PY_ABI_TYPE_UNAVAILABLE", "scale", np.ones(1, np.float32), 2.0)
    refused(module, ip.ValidationError, "TYPE_MISMATCH", "store2", np.zeros(3, np.int32), True, 1)
    refused(module, ip.ValidationError, "TYPE_MISMATCH", "flag", np.zeros(1, np.uint32), 1)
    refused(module, ip.ValidationError, "TYPE_MISMATCH", "plain", 1, np.int32(2))


def test_name_count_and_fuel(module):
    refused(module, ip.ValidationError, "NAME_UNKNOWN", "missing")
    refused(module, ip.ValidationError, "PY_ARGUMENT_COUNT", "store2", np.zeros(3, np.int32), 0)
    a = np.zeros(3, np.int32)
    with pytest.raises(ip.ResourceError) as raised:
        module.call_reference("store2", a, 0, 1, fuel=0)
    assert code(raised.value) == "INVALID_FUEL"
    assert a.tolist() == [0, 0, 0]


# --- normal returns -----------------------------------------------------------

def test_commit_scalar_result_and_provenance(module):
    a = np.array([1, 2, 3], np.int32)
    doc = module.call_reference("store2", a, 2, -9)
    assert (doc["engine"], doc["copying"], doc["result"], doc["committed"]) == ("reference", True, -9, [0])
    assert doc["steps"] > 0 and doc["cleanup"] == []
    assert a.tolist() == [2, 2, -9]
    doc = module.call_reference("plain", 2, 3)
    assert {k: v for k, v in doc.items() if k != "steps"} == {
        "engine": "reference", "copying": True, "result": 5, "cleanup": [], "committed": []}
    assert doc["steps"] > 0


def test_result_ok_and_err_are_returned_and_commit(module):
    b = np.array([3, 4], np.int32)
    assert module.call_reference("settle", b, 5)["result"] == {"variant": "Ok", "value": 5}
    assert b.tolist() == [5, 4]
    doc = module.call_reference("settle", b, -2)
    assert doc["result"] == {"variant": "Err", "value": 7}
    assert b.tolist() == [-2, 4], "Err is a normal, committing return"
    u = np.zeros(1, np.uint32)
    assert module.call_reference("flag", u, True)["result"] == {"variant": "Ok", "value": True}
    u[0] = 0
    assert module.call_reference("flag", u, False)["result"] == {"variant": "Err", "value": -3}
    assert u.tolist() == [1]


def test_same_array_twice_read_is_one_owner(module):
    a = np.array([4, 5], np.int32)
    assert module.call_reference("reads", a, a)["result"] == 8


def test_f32_bits_are_exact_in_and_out(module):
    xs = np.array([0x7F800001, 0x80000000], np.uint32).view(np.float32).copy()
    ys = np.array([0x7FA00001, 0xFFC12345], np.uint32).view(np.float32).copy()
    doc = module.call_reference("fcopy", xs, ys, 1)
    assert xs.view(np.uint32).tolist() == [0x7FA00001, 0xFFC12345]
    assert doc["result"]["bits"] == 0x7FA00001
    out = module.call_reference("fresh", ys)["result"]
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert out.view(np.uint32).tolist() == [0x7FA00001, 0xFFC12345]
    assert not np.shares_memory(out, ys)


# --- non-normal outcomes: exceptions, provenance, unchanged buffers ---------

def test_trap_keeps_bytes_and_span(module):
    a = np.array([5, 6], np.int32)
    with pytest.raises(ip.TrapError) as raised:
        module.call_reference("div_after", a, 0)
    error = raised.value
    assert code(error) == "DIVISION_BY_ZERO"
    span = error.diagnostics["trap"]["span"]
    assert SOURCE[span["start"]:span["end"]] == b"/"
    assert error.diagnostics["diagnostics"][0]["path"] == "reference_arrays.inv"
    assert error.steps_used > 0
    assert a.tolist() == [5, 6]
    with pytest.raises(ip.TrapError) as raised:
        module.call_reference("over", a)
    assert code(raised.value) == "INTEGER_OVERFLOW"
    assert a.tolist() == [5, 6]
    with pytest.raises(ip.TrapError):
        module.call_reference("fcopy", np.ones(2, np.float32), np.ones(2, np.float32), 5)


def test_fuel_keeps_bytes_and_callee_temporaries_follow_the_language_limit(module):
    a = np.array([1, 2, 3], np.int32)
    full = module.call_reference("store2", a.copy(), 2, -9)["steps"]
    for fuel in range(1, full):
        with pytest.raises(ip.ResourceError) as raised:
            module.call_reference("store2", a, 2, -9, fuel=fuel)
        assert raised.value.steps_used == fuel
        assert a.tolist() == [1, 2, 3]
    # The callee temporary is the core's allocation, bounded by the language
    # limit (raised separately), not by the host-array bound: this call now runs.
    big = np.ones(1_000_000, np.int32)
    other = np.ones(1_000_000, np.int32)
    assert module.call_reference("grow_after", big, other)["result"] == 1_000_001
    assert big[0] == 7 and other[0] == 1
    # A refusal at the host live bound keeps every byte; see
    # test_f32_live_refusal_before_execution_matches_the_core.


def test_repeated_calls_release_guards(module):
    a = np.array([0, 6], np.int32)
    for k in range(60):
        if k % 3 == 0:
            with pytest.raises(ip.TrapError):
                module.call_reference("div_after", a, 0)
        elif k % 3 == 1:
            refused(module, ip.ValidationError, "PY_BUFFER_ALIAS", "both", a, a)
        else:
            module.call_reference("div_after", a, 2)
    assert a.tolist() == [1, 6]
    b = np.zeros(1, np.uint32)
    for _ in range(20):
        module.call_reference("spin", b, 5)
    assert b.tolist() == [100]


# --- GIL retention and the native path ---------------------------------------

def test_gil_is_held_for_the_whole_call(module):
    counter = [0]
    stop = threading.Event()

    def run():
        while not stop.is_set():
            counter[0] += 1

    interval = sys.getswitchinterval()
    sys.setswitchinterval(5.0)
    thread = threading.Thread(target=run)
    xs = np.zeros(1, np.uint32)
    try:
        thread.start()
        time.sleep(0.05)  # releases the GIL: the thread runs
        start = counter[0]
        time.sleep(0.05)
        assert counter[0] > start, "control: a GIL release lets the thread run"
        start, began = counter[0], time.perf_counter()
        doc = module.call_reference("spin", xs, 100_000)
        elapsed, after = time.perf_counter() - began, counter[0]
        assert doc["result"] == 100_000 and elapsed < 4.0
        assert after == start, "no other Python thread ran during the call"
    finally:
        stop.set()
        sys.setswitchinterval(interval)
        thread.join()
    assert ip.capabilities()["gil_released"] is False
    assert ip.capabilities()["reference_array_entry"] == "copying"


def test_native_call_path_still_refuses_arrays(module):
    with pytest.raises(ip.EngineError) as raised:
        module.call("store2", np.zeros(3, np.int32), 0, 1)
    assert code(raised.value) == "PY_EXPORT_UNAVAILABLE"


# --- f32 bridges: aliases, live cap, multiple mut buffers --------------------

def test_f32_exact_alias_counts_once_toward_the_host_live_bound(module):
    f = np.zeros(1_000_000, np.float32)
    i = np.zeros(1_000_000, np.int32)
    g, j = f.copy(), i.copy()
    # The callee temporary is the core's allocation, bounded by the language
    # limit (raised separately), so distinct arrays plus a temporary run.
    assert module.call_reference("near_f", f, g)["result"] == 2_000_001
    assert module.call_reference("near_i", i, j)["result"] == 2_000_001
    # An exact alias counts once toward the 2,000,000-element host union.
    assert module.call_reference("three_f", f, f, g)["result"] == 3_000_000
    assert module.call_reference("three_i", i, i, j)["result"] == 3_000_000


def test_f32_live_refusal_before_execution_matches_the_core(module):
    fs = [np.zeros(1_000_000, np.float32) for _ in range(3)]
    is_ = [np.zeros(1_000_000, np.int32) for _ in range(3)]
    f_error = refused(module, ip.ResourceError, "ARRAY_SIZE_LIMIT", "three_f", *fs)
    i_error = refused(module, ip.ResourceError, "ARRAY_SIZE_LIMIT", "three_i", *is_)
    assert f_error.diagnostics["diagnostics"][0]["message"] == i_error.diagnostics["diagnostics"][0]["message"]


def test_multiple_f32_mut_buffers_commit_exact_bits_or_stay_unchanged(module):
    a = np.array([0x3F800000, 0x40000000], np.uint32).view(np.float32).copy()
    b = np.array([0x7F800001, 0x80000000, 0x00000001], np.uint32).view(np.float32).copy()
    c = np.array([5, 6], np.int32)
    r = np.array([0x7FA00001, 0xFFC12345], np.uint32).view(np.float32).copy()
    before = [x.tobytes() for x in (a, b, c, r)]
    with pytest.raises(ip.TrapError):
        module.call_reference("fmulti", a, b, c, r, 3)
    assert [x.tobytes() for x in (a, b, c, r)] == before, "trap after writes: all unchanged"
    doc = module.call_reference("fmulti", a, b, c, r, 2)
    assert doc["committed"] == [0, 1, 2]
    assert a.view(np.uint32).tolist() == [0x7FA00001, 0x80000000]
    assert b.view(np.uint32).tolist() == [0xFFC12345, 0x80000000, 0x7FA00001]
    assert c.tolist() == [9, 6] and r.tobytes() == before[3]
    assert doc["result"]["bits"] == 0xFFC12345
