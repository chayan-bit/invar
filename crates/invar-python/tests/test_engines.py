"""Shared CLI dispatch, typed refusals, actual execution receipts and HA-1.2."""
import json
import os
import pathlib
import subprocess

import invar_python as inv
import jsonschema
import numpy as np
import pytest

from conftest import child_timeout

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCHEMA = json.loads((ROOT / "schema/invar-diagnostics-v1.schema.json").read_text())
DEVICE_SCHEMA = json.loads((ROOT / "schema/invar-explain-device-v1.schema.json").read_text())
EXPLAIN_SCHEMA = json.loads((ROOT / "schema/explain-v1.schema.json").read_text())
SIMPLE = "fn main() -> i32 { return 42; }"
MAP = """fn main() -> i32 {
    let xs: Array<u32> = [1u32, 2u32, 3u32];
    var out: Array<u32> = filled_u32(len(xs), 0u32);
    for i in 0u32..len(xs) { out[i] = wrapping_add(xs[i], 3u32); }
    if out[2u32] == 6u32 { return 42; } return 1;
}"""


def metal_library(cli, source, stem):
    """The compiled-ahead Metal library `invar build --target metal` writes for source (#46)."""
    program = stem.with_suffix(".inv")
    program.write_text(source)
    library = stem.with_suffix(".metallibc")
    built = subprocess.run([cli, "build", str(program), "--target", "metal", "--output", str(library)],
                           capture_output=True, text=True, timeout=child_timeout(300))
    assert built.returncode == 0, (built.stdout, built.stderr)
    assert json.loads(built.stdout)["library"]["runtime_compiler"] == "none"
    return library


def rejected(code, callback, error_class=inv.EngineError):
    with pytest.raises(error_class) as caught:
        callback()
    document = caught.value.diagnostics
    assert document == json.loads(caught.value.diagnostic_json)
    jsonschema.validate(document, SCHEMA)
    assert document["diagnostics"][0]["code"] == code
    return caught.value


@pytest.mark.parametrize("engine", ["reference", "native-serial", "cpu-parallel", "auto", "cuda", "metal"])
def test_selection_uses_cli_names(engine):
    module = inv.compile_source(SIMPLE, engine=engine)
    assert module.engine == engine
    assert module.receipt() is None


@pytest.mark.parametrize("engine", ["native", "CPU", "", "unknown"])
def test_unknown_engine_refuses(engine):
    rejected("PY_ENGINE_UNAVAILABLE", lambda: inv.compile_source(SIMPLE, engine=engine))


@pytest.mark.parametrize("engine", ["reference", "native-serial"])
def test_serial_receipt(engine):
    module = inv.compile_source(SIMPLE, engine=engine)
    if engine == "native-serial" and not inv.capabilities()["engine_features"]["native"]:
        rejected("NATIVE_UNAVAILABLE", module.run_main)
        return
    assert module.run_main() == 42
    receipt = module.run_main(observe=True)
    assert receipt == module.receipt()
    assert receipt["backends"] == [engine]
    assert receipt["requested_engine"] == engine
    assert receipt["result"] == 42


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_cpu_parallel_publishes_a_real_region(workers):
    module = inv.compile_source(MAP, engine="cpu-parallel")
    receipt = module.run_main(workers=workers, observe=True)
    assert receipt["result"] == 42
    assert receipt["backends"] == ["cpu-parallel"]
    assert receipt["choices"] and receipt["fallback_reason"] is None
    assert module.receipt() == receipt


@pytest.mark.parametrize("workers", [0, 65, -1, True, 1.0, 2**100])
def test_workers_boundary(workers):
    module = inv.compile_source(MAP, engine="cpu-parallel")
    rejected("CPU_WORKERS", lambda: module.run_main(workers=workers))


@pytest.mark.parametrize("engine", ["reference", "native-serial", "auto", "cuda", "metal"])
def test_workers_only_for_cpu(engine):
    module = inv.compile_source(MAP, engine=engine)
    rejected("CPU_WORKERS", lambda: module.run_main(workers=1))


def test_argument_refusal_keeps_the_previous_receipt():
    module = inv.compile_source(MAP, engine="reference")
    assert module.run_main() == 42
    receipt = module.run_main(observe=True)
    assert receipt == module.receipt()
    rejected("CPU_WORKERS", lambda: module.run_main(workers=1))
    assert module.receipt() == receipt


@pytest.mark.parametrize("source", [SIMPLE, MAP.replace("for i", "if false { for i").replace("if out[2u32]", "} if out[2u32]")])
def test_forced_cpu_never_silently_becomes_serial(source):
    module = inv.compile_source(source, engine="cpu-parallel")
    rejected("PY_ENGINE_UNAVAILABLE", module.run_main)
    assert module.receipt() is None


@pytest.mark.parametrize("source", [SIMPLE, MAP])
def test_auto_reports_and_explain_does_not_reexecute(source):
    module = inv.compile_source(source, engine="auto")
    assert "device_execution" not in module.explain()
    receipt = module.run_main(observe=True)
    assert receipt["result"] == 42 and receipt["requested_engine"] == "auto"
    assert set(receipt["backends"]) <= {"mir-serial", "cpu-parallel", "cuda", "metal"}
    if source == SIMPLE:
        assert receipt["backends"] == ["mir-serial"]
    explain = module.explain()
    jsonschema.validate(explain, EXPLAIN_SCHEMA)
    jsonschema.validate(explain["device_execution"], DEVICE_SCHEMA)
    assert explain["device_execution"]["choices"] == receipt["choices"]
    assert module.receipt() == receipt
    # Repeated explain reads the receipt and does not launch a kernel or change cache counts.
    receipt = module.receipt()
    assert module.explain() == module.explain()
    assert module.receipt() == receipt


@pytest.mark.parametrize("engine", ["reference", "native-serial", "cpu-parallel", "auto"])
def test_trap_and_fuel_are_typed_and_clear_receipts(engine):
    module = inv.compile_source(MAP, engine=engine, path="engines.inv")
    if engine == "native-serial" and not inv.capabilities()["engine_features"]["native"]:
        rejected("NATIVE_UNAVAILABLE", module.run_main)
        return
    assert module.run_main() == 42
    rejected("EVALUATION_LIMIT", lambda: module.run_main(fuel=1), inv.ResourceError)
    assert module.receipt() is None
    rejected("DIVISION_BY_ZERO", lambda: inv.compile_source(
        "fn main() -> i32 { return 1 / 0; }", engine=engine).run_main(), inv.TrapError)


@pytest.mark.parametrize("engine", ["cpu-parallel", "auto", "cuda", "metal"])
def test_host_exports_refuse_unsupported_engines(engine):
    rejected("NATIVE_ABI", lambda: inv.compile_source(
        "fn count(xs: read Array<u32>) -> u32 { return len(xs); }",
        engine=engine, exports=["count"]))


@pytest.mark.parametrize("engine", ["cuda", "metal"])
def test_forced_device_refusal_is_typed(engine):
    module = inv.compile_source(SIMPLE, engine=engine)
    rejected(engine.upper() + "_UNAVAILABLE", module.run_main)
    assert module.receipt() is None


@pytest.mark.parametrize("engine", ["cuda", "metal"])
def test_device_execution_or_counted_skip(engine, tmp_path):
    module = inv.compile_source(MAP, engine=engine)
    if not inv.capabilities()["engine_features"][engine]:
        if os.environ.get("INVAR_REQUIRE_DEVICE") == engine:
            pytest.fail(f"{engine}: required qualification feature was not built")
        rejected(engine.upper() + "_UNAVAILABLE", module.run_main)
        pytest.skip(f"{engine}: extension built without the platform feature")
    # Production Metal runs a compiled-ahead library; without INVAR_CLI to build one it refuses.
    kwargs = {}
    if engine == "metal" and os.environ.get("INVAR_CLI"):
        kwargs["artifact"] = metal_library(os.environ["INVAR_CLI"], MAP, tmp_path / "map")
    try:
        receipt = module.run_main(observe=True, **kwargs)
    except inv.EngineError as error:
        if os.environ.get("INVAR_REQUIRE_DEVICE") == engine:
            raise
        # An eligible fixed integer map must run when a device is present.
        # Never hide a toolchain mismatch, pending completion or execution defect.
        code = error.diagnostics["diagnostics"][0]["code"]
        assert code == engine.upper() + "_UNAVAILABLE"
        jsonschema.validate(error.diagnostics, SCHEMA)
        pytest.skip(f"{engine}: {error.diagnostics['diagnostics'][0]['message']}")
    assert receipt["result"] == 42
    assert receipt["backends"] == [engine]
    assert receipt["choices"] and all(c["backend"] == engine for c in receipt["choices"])
    jsonschema.validate(module.explain()["device_execution"], DEVICE_SCHEMA)


def test_f32_and_returned_arrays_remain_copying_on_any_selected_engine():
    for engine in ["reference", "native-serial", "cpu-parallel", "auto", "cuda", "metal"]:
        module = inv.compile_source(
            "fn same(xs: read Array<f32>) -> Array<f32> { var b = filled_f32(len(xs), 0.0); for i in 0u32..len(xs) { b[i] = xs[i]; } return b; }", engine=engine)
        xs = np.array([-0.0, 1.5], dtype=np.float32)
        receipt = module.call_reference("same", xs)
        assert receipt["engine"] == "reference" and receipt["copying"] is True
        assert "record_addresses" not in receipt
        assert receipt["result"].ctypes.data != xs.ctypes.data
        assert receipt["result"].tobytes() == xs.tobytes()


@pytest.mark.parametrize("engine", ["cpu-parallel", "auto", "cuda", "metal"])
def test_cli_binding_dispatch_parity(engine, tmp_path):
    cli = os.environ.get("INVAR_CLI")
    if not cli:
        pytest.skip("INVAR_CLI is not set for cross-process CLI parity")
    source = tmp_path / "map.inv"
    source.write_text(MAP)
    args = [cli, "run", str(source), "--engine", engine, "--format", "json"]
    kwargs = {}
    if engine == "cpu-parallel":
        args += ["--workers", "2"]
        kwargs["workers"] = 2
    result = subprocess.run(args, capture_output=True, text=True, timeout=90)
    document = json.loads(result.stdout)
    module = inv.compile_source(MAP, engine=engine)
    if result.returncode == 0:
        assert module.run_main(**kwargs) == document["result"]["value"] == 42
    else:
        rejected(document["diagnostics"][0]["code"], lambda: module.run_main(**kwargs))


@pytest.mark.parametrize("engine", ["cuda", "metal"])
@pytest.mark.parametrize("source", [
    MAP.replace("filled_u32(len(xs), 0u32)", "filled_u32(1u32, 0u32)"),
    MAP.replace("for i", "if false { for i").replace("if out[2u32]", "} if out[2u32]"),
])
def test_forced_device_live_guards_never_replay_on_cpu(engine, source):
    module = inv.compile_source(source, engine=engine)
    rejected(engine.upper() + "_UNAVAILABLE", module.run_main)
    assert module.receipt() is None
