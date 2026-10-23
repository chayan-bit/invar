import gc
import json
import pathlib
import threading
import time

import invar_python as inv
import jsonschema
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCHEMA = json.loads((ROOT / "schema/invar-diagnostics-v3.schema.json").read_text())

SOURCE = """
extern c fn c_add(x: i32, y: i32) -> Result<i32,ForeignError> oracle add;
extern c fn c_bytes(b: read Bytes) -> Result<u32,ForeignError> oracle sum;
extern c fn c_address(b: read Bytes, lo: u32, hi: u32) -> Result<u32,ForeignError> oracle address;
extern c fn c_text_len(s: read String) -> Result<u32,ForeignError> oracle text_len;
extern c fn c_hold(b: read Bytes) -> Result<u32,ForeignError> oracle sum;
extern c fn c_copy(b: read Bytes) -> Result<Bytes,ForeignError> oracle bytes_oracle;
extern c fn c_take(b: take Bytes) -> Result<Bytes,ForeignError> oracle transfer_oracle;
extern c fn c_count() -> Result<u32,ForeignError> oracle zero;
extern c fn c_releases() -> Result<u32,ForeignError> oracle zero;
extern c fn c_fail() -> Result<u32,ForeignError> oracle zero;
extern c fn c_missing() -> Result<u32,ForeignError> oracle zero;
extern c fn c_bad_layout() -> Result<Bytes,ForeignError> oracle empty;
extern c fn c_oversize() -> Result<Bytes,ForeignError> oracle empty;
extern c fn c_bad_text() -> Result<String,ForeignError> oracle text;
fn add(x: i32, y: i32) -> Result<i32,ForeignError> { return Result::Ok(x+y); }
fn sum(b: read Bytes) -> Result<u32,ForeignError> {
  var accumulator: u32 = 0u32;
  for i in 0u32 .. len(b) { accumulator = accumulator + b[i]; }
  return Result::Ok(accumulator);
}
fn address(b: read Bytes, lo: u32, hi: u32) -> Result<u32,ForeignError> { return Result::Ok(1u32); }
fn text_len(s: read String) -> Result<u32,ForeignError> { return Result::Ok(len(s)); }
fn bytes_oracle(b: read Bytes) -> Result<Bytes,ForeignError> {
  let copied: Bytes = bytes_slice(b, 0u32, len(b)); return Result::Ok(move copied);
}
fn transfer_oracle(b: take Bytes) -> Result<Bytes,ForeignError> { return Result::Ok(move b); }
fn zero() -> Result<u32,ForeignError> { return Result::Ok(0u32); }
fn empty() -> Result<Bytes,ForeignError> { return Result::Ok(b""); }
fn text() -> Result<String,ForeignError> { return Result::Ok(""); }
fn main() -> i32 { return 42; }
"""


@pytest.fixture(scope="module")
def module():
    return inv.compile_source(SOURCE)


def rejected(error_type, code, callback):
    with pytest.raises(error_type) as caught:
        callback()
    document = caught.value.diagnostics
    jsonschema.validate(document, SCHEMA)
    assert json.loads(caught.value.diagnostic_json) == document
    assert document["diagnostics"][0]["code"] == code
    return caught.value


def test_compile_signatures_explain_and_capabilities(module):
    assert module.run_main() == 42
    assert module.exports() == []
    functions = module.functions()
    assert any(f["name"] == "main" and not f["callable"] for f in functions["source_functions"])
    assert any(f["name"] == "c_bytes" and f["callable"] for f in functions["c_imports"])
    assert not next(f for f in functions["c_imports"] if f["name"] == "c_take")["callable"]
    jsonschema.validate(module.explain(), json.loads((ROOT / "schema/explain-v1.schema.json").read_text()))
    caps = inv.capabilities()
    assert caps["scalar_artifact_exports"] == ["i32", "u32"]
    assert caps["compiled_c_exports"] == "native-serial requires invar/native at build time"
    assert not caps["numeric_array_abi"]
    assert caps["f32_source_arithmetic"] and not caps["gil_released"]


@pytest.mark.parametrize("dtype", ["uint8", "int32", "uint32", "float32"])
@pytest.mark.parametrize("size", [0, 1, 100_000])
def test_valid_array_metadata_no_copy(dtype, size):
    array = np.arange(size, dtype=dtype)
    info = inv.validate_array(array, dtype, length=size)
    assert info["address"] == array.ctypes.data
    assert info["length"] == size
    assert info["owns_data"]
    other = np.empty_like(array)
    inv.validate_buffers(array, other, dtype)


@pytest.mark.parametrize("length", [0, 1, 100_000, 1_048_576])
def test_bytes_zero_copy_and_c_owned_output(module, c_library, length):
    array = np.ones(length, dtype=np.uint8)
    address = array.ctypes.data
    assert module.call_foreign("c_address", c_library, array, address & 0xFFFFFFFF, address >> 32) == 1
    assert module.call_foreign("c_bytes", c_library, array) == length
    releases = module.call_foreign("c_releases", c_library)
    assert module.call_foreign("c_copy", c_library, array) == b"\1" * length
    assert module.call_foreign("c_releases", c_library) == releases + 1
    assert np.all(array == 1)


def test_immutable_bytes_text_and_scalars(module, c_library):
    calls = module.call_foreign("c_count", c_library)
    assert module.call_foreign("c_bytes", c_library, b"a\x00\xff") == 352
    assert module.call_foreign("c_count", c_library) == calls + 1
    assert module.call_foreign("c_add", c_library, -2147483648, 2147483647) == -1
    assert module.call_foreign("c_text_len", c_library, "e\u0301é\0") == 6
    array = np.arange(10, dtype=np.uint8)
    array.flags.writeable = False
    assert module.call_foreign("c_bytes", c_library, array) == 45


@pytest.mark.parametrize("length", [0, 1, 256, 100_000])
def test_c_import_agrees_with_source_oracle(module, c_library, length):
    array = np.full(length, 120, dtype=np.uint8)
    expected = 120 * length
    assert module.call_foreign("c_bytes", c_library, array) == expected
    entrypoint = f'''fn main() -> i32 {{
      let data: Bytes = b"{'x' * length}";
      let answer: Result<u32,ForeignError> = c_bytes(data);
      let value: u32 = match move answer {{ Result::Ok(v) => v, Result::Err(e) => 0u32 }};
      if value == {expected}u32 {{ return 1; }} return 0;
    }}'''
    source = SOURCE.replace("fn main() -> i32 { return 42; }", entrypoint)
    assert inv.compile_source(source).run_main(fuel=10_000_000) == 1


@pytest.mark.parametrize("array,dtype,kwargs,code", [
    (np.zeros(4, dtype=np.float64), "uint32", {}, "PY_ARRAY_DTYPE"),
    (np.zeros(4, dtype=np.int32), "uint32", {}, "PY_ARRAY_DTYPE"),
    (np.zeros((2, 2), dtype=np.uint32), "uint32", {}, "PY_ARRAY_RANK"),
    (np.array(1, dtype=np.uint32), "uint32", {}, "PY_ARRAY_RANK"),
    (np.zeros(4, dtype=np.uint32), "uint32", {"length": 5}, "PY_ARRAY_SHAPE"),
    (np.arange(8, dtype=np.uint32)[::2], "uint32", {}, "PY_ARRAY_CONTIGUITY"),
    (np.arange(8, dtype=np.uint32)[::-1], "uint32", {}, "PY_ARRAY_CONTIGUITY"),
    (np.ndarray((4,), dtype=np.uint32, buffer=bytearray(17), offset=1), "uint32", {}, "PY_ARRAY_ALIGNMENT"),
    (np.frombuffer(bytearray(16), dtype=np.uint32), "uint32", {}, "PY_ARRAY_OWNERSHIP"),
    (np.zeros(1_000_001, dtype=np.uint32), "uint32", {}, "PY_BUFFER_LIMIT"),
    (np.zeros(1_048_577, dtype=np.uint8), "uint8", {}, "PY_BUFFER_LIMIT"),
    (np.zeros(4, dtype=">u4"), "uint32", {}, "PY_ARRAY_DTYPE"),
])
def test_invalid_arrays(array, dtype, kwargs, code):
    rejected(inv.ValidationError, code, lambda: inv.validate_array(array, dtype, **kwargs))


@pytest.mark.parametrize("access", ["read", "mut"])
def test_bool_array_bytes_other_than_zero_or_one_are_refused(access):
    module = inv.compile_source(
        f"fn touch(flags: {access} Array<Bool>) -> u32 {{ return len(flags); }}", path="bool.inv"
    )
    flags = np.array([True, False, True])
    flags.view(np.uint8)[2] = 2
    rejected(inv.ValidationError, "PY_ARRAY_BOOL_VALUE", lambda: module.call_reference("touch", flags))
    assert flags.view(np.uint8).tolist() == [1, 0, 2]


def test_readonly_output_alias_and_subclass_rejected():
    array = np.arange(4, dtype=np.uint32)
    output = np.empty_like(array)
    output.flags.writeable = False
    rejected(inv.ValidationError, "PY_ARRAY_WRITABILITY", lambda: inv.validate_buffers(array, output, "uint32"))
    rejected(inv.ValidationError, "PY_BUFFER_ALIAS", lambda: inv.validate_buffers(array, array, "uint32"))
    class Array(np.ndarray):
        pass
    rejected(inv.ValidationError, "PY_ARRAY_DTYPE", lambda: inv.validate_array(array.view(Array), "uint32"))


@pytest.mark.parametrize("value,code", [
    (np.zeros(4, dtype=np.uint32), "PY_ARRAY_DTYPE"),
    (np.zeros((2, 2), dtype=np.uint8), "PY_ARRAY_RANK"),
    (np.arange(8, dtype=np.uint8)[::2], "PY_ARRAY_CONTIGUITY"),
    (np.frombuffer(b"abc", dtype=np.uint8), "PY_ARRAY_OWNERSHIP"),
    pytest.param(b"x" * 1_048_577, "PY_BUFFER_LIMIT", id="oversized-bytes"),
])
def test_invalid_input_never_enters_c(module, c_library, value, code):
    count = module.call_foreign("c_count", c_library)
    rejected(inv.ValidationError, code, lambda: module.call_foreign("c_bytes", c_library, value))
    assert module.call_foreign("c_count", c_library) == count


@pytest.mark.parametrize("value", [True, -2147483649, 2147483648, 1.0, "1"])
def test_scalar_rejection_never_enters_c(module, c_library, value):
    count = module.call_foreign("c_count", c_library)
    rejected(inv.ValidationError, "TYPE_MISMATCH", lambda: module.call_foreign("c_add", c_library, value, 1))
    assert module.call_foreign("c_count", c_library) == count


def test_callbacks_are_not_invoked_with_buffer_references(module, c_library):
    array = np.ones(4, dtype=np.uint8)
    class Index:
        called = False
        def __index__(self):
            self.called = True
            array.resize((100,), refcheck=False)
            return 0
    index = Index()
    rejected(inv.ValidationError, "TYPE_MISMATCH", lambda: module.call_foreign("c_address", c_library, array, index, 0))
    assert not index.called
    assert array.shape == (4,)


def test_lifetime_and_resize_are_serialized_by_the_gil(module, c_library):
    array = np.ones(100, dtype=np.uint8)
    begin = threading.Event()
    attempts = []
    def resize():
        begin.wait()
        time.sleep(0.025)
        attempts.append(time.monotonic())
        array.resize((200,), refcheck=False)
    thread = threading.Thread(target=resize)
    thread.start()
    start = time.monotonic()
    begin.set()
    assert module.call_foreign("c_hold", c_library, array) == 100
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert attempts[0] - start >= 0.14
    # The call itself pins an otherwise temporary owner through C's delayed read.
    assert module.call_foreign("c_hold", c_library, np.ones(100, dtype=np.uint8)) == 100
    gc.collect()


def test_all_exception_classes_have_schema_valid_documents(module, c_library):
    rejected(inv.CompileError, "PARSE_EXPECTED_TOKEN", lambda: inv.compile_source("fn"))
    rejected(inv.CheckError, "NAME_UNKNOWN", lambda: inv.compile_source("fn main() -> i32 { return absent; }"))
    rejected(inv.TrapError, "DIVISION_BY_ZERO", lambda: inv.compile_source("fn main() -> i32 { return 1 / 0; }").run_main())
    rejected(inv.ResourceError, "INVALID_FUEL", lambda: module.run_main(fuel=0))
    rejected(inv.ResourceError, "SOURCE_SIZE_LIMIT", lambda: inv.compile_source(b" " * 1_048_577))
    rejected(inv.ResourceError, "EVALUATION_LIMIT", lambda: inv.compile_source("fn main() -> i32 { while true {} return 0; }").run_main(fuel=10))
    rejected(inv.EngineError, "PY_EXPORT_UNAVAILABLE", lambda: module.call("main"))
    rejected(inv.ForeignError, "PY_FOREIGN_STATUS", lambda: module.call_foreign("c_fail", c_library))
    rejected(inv.ForeignError, "PY_FOREIGN_SYMBOL", lambda: module.call_foreign("c_missing", c_library))
    rejected(inv.ForeignError, "PY_FOREIGN_LAYOUT", lambda: module.call_foreign("c_bad_layout", c_library))
    rejected(inv.ValidationError, "NAME_UNKNOWN", lambda: module.call("absent"))
    rejected(inv.ValidationError, "PY_ARGUMENT_COUNT", lambda: module.call("main", 1))
    rejected(inv.ValidationError, "PY_ARGUMENT_COUNT", lambda: module.call_foreign("c_bytes", c_library))
    rejected(inv.ValidationError, "PY_LIBRARY_PATH", lambda: module.call_foreign("c_bytes", "relative.so", b"x"))
    rejected(inv.ValidationError, "PY_TAKE_UNAVAILABLE", lambda: module.call_foreign("c_take", c_library, b"x"))


@pytest.mark.parametrize("name,code", [("c_fail", "PY_FOREIGN_STATUS"), ("c_bad_text", "INVALID_UTF8"), ("c_oversize", "PY_BUFFER_LIMIT")])
def test_c_owned_error_outputs_are_released_once(module, c_library, name, code):
    count = module.call_foreign("c_releases", c_library)
    rejected(inv.ForeignError, code, lambda: module.call_foreign(name, c_library))
    assert module.call_foreign("c_releases", c_library) == count + 1


def test_diagnostic_path_bounds_and_conversion_trap():
    rejected(inv.CompileError, "PARSE_EXPECTED_TOKEN", lambda: inv.compile_source("fn", path="x" * 5000))
    source = (ROOT / "tests/fixtures/r1_01/inexact-conversion.inv").read_text()
    with pytest.raises(inv.TrapError) as caught:
        inv.compile_source(source).run_main()
    document = caught.value.diagnostics
    assert document["schema"] == "invar-diagnostics-v3"
    assert document["trap"]["category"] == "UNIT_CONVERSION_INEXACT"
    jsonschema.validate(document, SCHEMA)


@pytest.mark.parametrize("value", [-1, 1 << 100, True, 1.5, "1", None])
def test_fuel_conversion_always_has_typed_diagnostics(module, value):
    rejected(inv.ResourceError, "INVALID_FUEL", lambda: module.run_main(fuel=value))


@pytest.mark.parametrize("value", [-1, 1 << 100, True, 1.5, "1"])
def test_shape_conversion_always_has_typed_diagnostics(value):
    array = np.zeros(1, dtype=np.uint32)
    rejected(inv.ValidationError, "PY_ARRAY_SHAPE", lambda: inv.validate_array(array, "uint32", length=value))


def test_control_conversion_does_not_invoke_python_callbacks(module):
    class Index:
        called = False
        def __index__(self):
            self.called = True
            return 1
    index = Index()
    rejected(inv.ResourceError, "INVALID_FUEL", lambda: module.run_main(fuel=index))
    array = np.zeros(1, dtype=np.uint32)
    rejected(inv.ValidationError, "PY_ARRAY_SHAPE", lambda: inv.validate_array(array, "uint32", length=index))
    assert not index.called


@pytest.mark.parametrize("engine", ["native", "unknown"])
def test_unsupported_engine_is_precise(engine):
    rejected(inv.EngineError, "PY_ENGINE_UNAVAILABLE", lambda: inv.compile_source("fn main() -> i32 { return 1; }", engine=engine))


# Hand-derived smoke corpus: not full ABI equivalence or an independent backend.
@pytest.mark.parametrize("source,expected", [
    ("fn main() -> i32 { return 6*7; }", 42),
    ("fn main() -> i32 { return -2147483648 % -1; }", 0),
    ("fn main() -> i32 { return -7 / 3; }", -2),
    ("fn main() -> i32 { if false { return 1 / 0; } return 8; }", 8),
    ("fn main() -> i32 { let b: Bytes = b\"abc\"; return 3; }", 3),
])
def test_bounded_reference_entrypoint_smoke(source, expected):
    assert inv.compile_source(source).run_main() == expected
