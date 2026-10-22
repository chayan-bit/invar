"""The v0.2 types f64, Array<f64> and Array<Bool> at the Python boundary (#237, HA-1.4).

`call_reference` takes an exact built-in `float` for f64, NumPy `float64` arrays and NumPy `bool_` arrays (every byte
0 or 1), read or mut, and returns f64 as a `float` and arrays as new NumPy arrays. Every refusal is typed, raised before
execution with `steps_used == 0`, and leaves the inputs byte-identical. Exports (`exports=[...]`) still refuse the
types at artifact admission.
"""
import struct

import numpy as np
import pytest

import invar_python as ip

SOURCE = b"""\
fn twice(x: f64) -> u32 { if x > 0.0 { return 1u32; } return 0u32; }
fn half() -> f64 { return 0.5; }
fn scale(x: f64, k: f64) -> f64 { return x * k; }
fn ident(x: f64) -> f64 { return x; }
fn width(xs: read Array<f64>) -> u32 { return len(xs); }
fn three(a: read Array<f64>, b: read Array<f64>, c: read Array<f64>) -> u32 { return len(a) + len(b) + len(c); }
fn summed(xs: read Array<f64>) -> f64 { var acc: f64 = 0.0f64; for i in 0u32..len(xs) { acc = acc + xs[i]; } return acc; }
fn grid() -> Array<f64> { return filled_f64(2u32, 0.25); }
fn copy_f64(xs: read Array<f64>, out: mut Array<f64>) -> u32 { for i in 0u32..len(xs) { out[i] = xs[i]; } return len(out); }
fn bump(xs: mut Array<f64>, ys: mut Array<f64>) -> u32 { xs[0u32] = 1.0f64; ys[0u32] = 2.0f64; return len(xs) + len(ys); }
fn write_then_trap(xs: mut Array<f64>) -> u32 { xs[0u32] = 9.0f64; xs[len(xs)] = 1.0f64; return 0u32; }
fn blen(xs: read Array<Bool>) -> u32 { return len(xs); }
fn count(xs: read Array<Bool>) -> u32 { var n: u32 = 0u32; for i in 0u32..len(xs) { if xs[i] { n = n + 1u32; } } return n; }
fn flags() -> Array<Bool> { var b: Array<Bool> = [true, false]; return b; }
fn mark(xs: read Array<f64>, out: mut Array<Bool>) -> u32 { for i in 0u32..len(xs) { out[i] = xs[i] > 0.0f64; } return len(out); }
fn set_then_trap(out: mut Array<Bool>) -> u32 { out[0u32] = true; out[len(out)] = true; return 0u32; }
fn main() -> i32 { return 0; }
"""

# The core admits f64 C import parameters (Array<Bool> and f32 ones are check errors), but call_foreign has no f64
# bridge, so functions() must not advertise the import as callable.
FOREIGN_F64 = b"""\
fn twice_f(x: f64) -> Result<i32,ForeignError> { if x > 0.0 { return Result::Ok(1); } return Result::Ok(0); }
extern c fn c_f64(x: f64) -> Result<i32,ForeignError> oracle twice_f;
fn main() -> i32 { return 0; }
"""

# NaN payloads, signalling NaN, signed zero, infinities and the smallest subnormal: every bit must survive.
SPECIAL_BITS = [
    0x7FF8_0000_0000_0123,
    0xFFF0_0000_0000_0001,
    0x7FF0_0000_0000_0001,
    0x8000_0000_0000_0000,
    0x7FF0_0000_0000_0000,
    0xFFF0_0000_0000_0000,
    0x0000_0000_0000_0001,
]


@pytest.fixture()
def module():
    return ip.compile_source(SOURCE, path="v02_types.inv")


def code(error):
    return error.diagnostics["diagnostics"][0]["code"]


def bits(array):
    return np.ascontiguousarray(array).view(np.uint64).tolist()


def from_bits(values):
    # A copy: a view would not own its data, and the boundary admits only owning arrays.
    return np.array(values, dtype=np.uint64).view(np.float64).copy()


def signature(module, name):
    function = next(f for f in module.functions()["source_functions"] if f["name"] == name)
    return [p["type"] for p in function["parameters"]], function["result"]


def refused(module, cls, expected, name, *args):
    before = [a.tobytes() for a in args if isinstance(a, np.ndarray)]
    with pytest.raises(cls) as raised:
        module.call_reference(name, *args)
    assert code(raised.value) == expected
    assert raised.value.steps_used == 0, "refused before execution"
    assert [a.tobytes() for a in args if isinstance(a, np.ndarray)] == before, "byte-identical inputs"
    return raised.value


def export_refused(name):
    with pytest.raises(ip.EngineError) as raised:
        ip.compile_source(SOURCE, path="v02_types.inv", exports=[name], engine="reference")
    assert code(raised.value) == "NATIVE_ABI"


# --- f64 scalars -----------------------------------------------------------

def test_f64_signatures(module):
    assert signature(module, "twice") == (["f64"], "u32")
    assert signature(module, "half") == ([], "f64")
    assert signature(module, "width") == (["Array<f64>"], "u32")
    assert signature(module, "grid") == ([], "Array<f64>")
    assert signature(module, "count") == (["Array<Bool>"], "u32")
    assert signature(module, "flags") == ([], "Array<Bool>")


def test_f64_scalar_round_trip(module):
    out = module.call_reference("scale", 1.5, 2.0)
    assert out["result"] == 3.0 and type(out["result"]) is float
    assert module.call_reference("twice", 1.5)["result"] == 1
    assert module.call_reference("twice", -1.5)["result"] == 0
    half = module.call_reference("half")["result"]
    assert half == 0.5 and type(half) is float


@pytest.mark.parametrize("pattern", SPECIAL_BITS)
def test_f64_scalar_keeps_every_bit(module, pattern):
    value = struct.unpack("<d", struct.pack("<Q", pattern))[0]
    result = module.call_reference("ident", value)["result"]
    assert struct.unpack("<Q", struct.pack("<d", result))[0] == pattern


@pytest.mark.parametrize("value", [1, True, None, "1.0", np.float64(1.0), np.float32(1.0), [1.0], 1 + 0j])
def test_f64_scalar_requires_an_exact_float(module, value):
    refused(module, ip.ValidationError, "TYPE_MISMATCH", "scale", value, 2.0)
    refused(module, ip.ValidationError, "TYPE_MISMATCH", "scale", 2.0, value)


def test_f64_scalar_float_subclass_is_refused(module):
    class Loose(float):
        pass

    refused(module, ip.ValidationError, "TYPE_MISMATCH", "twice", Loose(1.0))


# --- Array<f64> -----------------------------------------------------------

def test_array_f64_read_and_empty(module):
    xs = np.array([0.5, 1.5, 2.0])
    before = xs.tobytes()
    assert module.call_reference("summed", xs)["result"] == 4.0
    assert xs.tobytes() == before
    assert module.call_reference("summed", np.zeros(0))["result"] == 0.0
    assert module.call_reference("width", np.zeros(0))["result"] == 0


def test_array_f64_large_bounded(module):
    assert module.call_reference("width", np.ones(1_000_000))["result"] == 1_000_000
    assert module.call_reference("summed", np.ones(50_000), fuel=10_000_000)["result"] == 50_000.0


def test_array_f64_mut_copy_is_exact(module):
    source = from_bits(SPECIAL_BITS)
    out = np.zeros(len(source))
    info = module.call_reference("copy_f64", source, out)
    assert info["result"] == len(source) and info["committed"] == [1]
    assert bits(out) == SPECIAL_BITS and bits(source) == SPECIAL_BITS


def test_array_f64_mut_is_unchanged_after_a_trap(module):
    xs = from_bits(SPECIAL_BITS)
    before = xs.tobytes()
    with pytest.raises(ip.TrapError) as raised:
        module.call_reference("write_then_trap", xs)
    assert raised.value.steps_used > 0
    assert xs.tobytes() == before, "the write before the trap is rolled back, NaN payloads included"


def test_array_f64_returned_is_a_new_numpy_array(module):
    grid = module.call_reference("grid")["result"]
    assert isinstance(grid, np.ndarray) and grid.dtype == np.float64 and grid.tolist() == [0.25, 0.25]
    assert grid.flags.owndata and grid.flags.writeable


def test_array_f64_two_mut_arrays_use_the_doubled_bridge_bound(module):
    # Two 1M mut f64 arrays bridge 16 MB: allowed because the bound scales with the 8-byte element.
    a, b = np.zeros(1_000_000), np.zeros(1_000_000)
    info = module.call_reference("bump", a, b)
    assert info["result"] == 2_000_000 and (a[0], b[0]) == (1.0, 2.0)


def test_array_f64_limits(module):
    refused(module, ip.ValidationError, "PY_BUFFER_LIMIT", "width", np.zeros(1_000_001))
    three = [np.zeros(1_000_000), np.zeros(1_000_000), np.zeros(1)]
    error = refused(module, ip.ResourceError, "ARRAY_SIZE_LIMIT", "three", *three)
    assert "live element" in str(error)


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.int32, np.uint8, np.bool_, ">f8"])
def test_array_f64_wrong_dtype(module, dtype):
    refused(module, ip.ValidationError, "PY_ARRAY_DTYPE", "width", np.zeros(3, dtype))


def test_array_f64_shape_layout_and_ownership(module):
    refused(module, ip.ValidationError, "PY_ARRAY_RANK", "width", np.zeros((2, 2)))
    refused(module, ip.ValidationError, "PY_ARRAY_CONTIGUITY", "width", np.arange(8.0)[::2])
    refused(module, ip.ValidationError, "PY_ARRAY_ALIGNMENT", "width",
            np.frombuffer(bytearray(8 * 4 + 4), dtype=np.float64, offset=4))
    refused(module, ip.ValidationError, "PY_ARRAY_OWNERSHIP", "width", np.arange(8.0)[1:5])
    read_only = np.zeros(3)
    read_only.setflags(write=False)
    refused(module, ip.ValidationError, "PY_ARRAY_WRITABILITY", "write_then_trap", read_only)
    assert module.call_reference("width", read_only)["result"] == 3


def test_array_f64_alias_of_two_mut_arrays(module):
    a = np.zeros(3)
    refused(module, ip.ValidationError, "PY_BUFFER_ALIAS", "bump", a, a)
    refused(module, ip.ValidationError, "PY_BUFFER_ALIAS", "copy_f64", a, a)


# --- Array<Bool> ----------------------------------------------------------

def test_array_bool_read(module):
    flags = np.array([True, False, True, True])
    before = flags.tobytes()
    assert module.call_reference("count", flags)["result"] == 3
    assert flags.tobytes() == before
    assert module.call_reference("count", np.zeros(0, np.bool_))["result"] == 0
    assert module.call_reference("blen", np.ones(1_000_000, np.bool_))["result"] == 1_000_000


def test_array_bool_mut_commits_and_returns(module):
    out = np.array([True, True, True])
    info = module.call_reference("mark", np.array([1.0, -1.0, 0.5]), out)
    assert info["result"] == 3 and out.tolist() == [True, False, True] and out.dtype == np.bool_
    flags = module.call_reference("flags")["result"]
    assert isinstance(flags, np.ndarray) and flags.dtype == np.bool_ and flags.tolist() == [True, False]


def test_array_bool_mut_is_unchanged_after_a_trap(module):
    out = np.array([False, True])
    with pytest.raises(ip.TrapError):
        module.call_reference("set_then_trap", out)
    assert out.tolist() == [False, True]


@pytest.mark.parametrize("dtype", [np.uint8, np.int8, np.float64, np.int32])
def test_array_bool_wrong_dtype(module, dtype):
    refused(module, ip.ValidationError, "PY_ARRAY_DTYPE", "count", np.zeros(3, dtype))


@pytest.mark.parametrize("name,mut", [("count", False), ("set_then_trap", True)])
@pytest.mark.parametrize("bad", [2, 3, 128, 255])
def test_array_bool_byte_other_than_zero_or_one_is_refused(module, name, mut, bad):
    flags = np.array([True, False, True, False])
    flags.view(np.uint8)[2] = bad
    assert flags.view(np.uint8).tolist() == [1, 0, bad, 0]
    refused(module, ip.ValidationError, "PY_ARRAY_BOOL_VALUE", name, flags)


def test_array_bool_byte_check_is_one_byte_wide_not_a_word(module):
    # The invalid byte sits at the last element of an odd-length array and in the first of another.
    for size, index in [(1, 0), (7, 6), (33, 0), (1_000_000, 999_999)]:
        flags = np.zeros(size, np.bool_)
        flags.view(np.uint8)[index] = 2
        refused(module, ip.ValidationError, "PY_ARRAY_BOOL_VALUE", "count", flags)


def test_array_bool_shape_layout_and_ownership(module):
    refused(module, ip.ValidationError, "PY_ARRAY_RANK", "count", np.zeros((2, 2), np.bool_))
    refused(module, ip.ValidationError, "PY_ARRAY_CONTIGUITY", "count", np.zeros(8, np.bool_)[::2])
    refused(module, ip.ValidationError, "PY_ARRAY_OWNERSHIP", "count", np.zeros(8, np.bool_)[1:5])
    refused(module, ip.ValidationError, "PY_BUFFER_LIMIT", "count", np.zeros(1_000_001, np.bool_))
    read_only = np.zeros(3, np.bool_)
    read_only.setflags(write=False)
    refused(module, ip.ValidationError, "PY_ARRAY_WRITABILITY", "set_then_trap", read_only)


# --- boundary helpers -----------------------------------------------------

def test_validate_array_names_float64_and_bool():
    assert ip.validate_array(np.zeros(4), "float64")["itemsize"] == 8
    assert ip.validate_array(np.zeros(4, np.bool_), "bool")["itemsize"] == 1
    flags = np.array([True, False])
    flags.view(np.uint8)[1] = 2
    with pytest.raises(ip.ValidationError) as raised:
        ip.validate_array(flags, "bool")
    assert code(raised.value) == "PY_ARRAY_BOOL_VALUE"
    with pytest.raises(ip.ValidationError) as raised:
        ip.validate_array(np.zeros(4), "bool")
    assert code(raised.value) == "PY_ARRAY_DTYPE"
    with pytest.raises(ip.ValidationError) as raised:
        ip.validate_array(np.zeros(4, np.float32), "float64")
    assert code(raised.value) == "PY_ARRAY_DTYPE"


def test_validate_buffers_checks_f64_pairs():
    a = np.zeros(4)
    ip.validate_buffers(a, np.zeros(4), "float64")
    with pytest.raises(ip.ValidationError) as raised:
        ip.validate_buffers(a, np.zeros(5), "float64")
    assert code(raised.value) == "PY_ARRAY_SHAPE"
    with pytest.raises(ip.ValidationError) as raised:
        ip.validate_buffers(a, a, "float64")
    assert code(raised.value) == "PY_BUFFER_ALIAS"


def test_capabilities_name_the_new_types():
    caps = ip.capabilities()
    assert {"float64", "bool"} <= set(caps["numpy_validation_types"])
    assert "read Array<f64>" in caps["reference_v02_types"]


# --- exports still refuse the types (native ABI is a separate step) -------

@pytest.mark.parametrize("name", ["twice", "half", "width", "grid", "count", "flags"])
def test_exports_refuse_the_v02_types(name):
    export_refused(name)


def test_f64_c_import_is_listed_but_not_callable(c_library):
    module = ip.compile_source(FOREIGN_F64, path="v02_foreign.inv")
    (imported,) = module.functions()["c_imports"]
    assert ([p["type"] for p in imported["parameters"]], imported["callable"]) == (["f64"], False)
    with pytest.raises(ip.EngineError) as raised:
        module.call_foreign("c_f64", c_library, 1.5)
    assert code(raised.value) == "PY_ABI_TYPE_UNAVAILABLE"
