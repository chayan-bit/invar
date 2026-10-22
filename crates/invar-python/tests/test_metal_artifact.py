"""#46: run_main(artifact=...) forwards a compiled-ahead Metal library as the CLI's --artifact does."""
import json
import os
import pathlib
import subprocess
import sys
import textwrap

import invar_python as inv
import jsonschema
import pytest

from conftest import child_timeout
from test_engines import MAP, SCHEMA, metal_library, rejected

METAL = inv.capabilities()["engine_features"]["metal"]
TRIPWIRE = "INVAR_METAL_SOURCE_TRIPWIRE"
RUNTIME_SOURCE = "INVAR_METAL_RUNTIME_SOURCE"


class BytesPath:
    def __fspath__(self):
        return b"/tmp/map.metallibc"


@pytest.mark.parametrize("engine", ["reference", "native-serial", "cpu-parallel", "cuda"])
def test_artifact_only_for_metal_and_auto(engine, tmp_path):
    module = inv.compile_source(MAP, engine=engine)
    error = rejected("PY_ENGINE_UNAVAILABLE", lambda: module.run_main(artifact=str(tmp_path / "map.metallibc")))
    assert "metal and auto" in str(error)
    assert module.receipt() is None


def test_artifact_refusal_keeps_the_previous_receipt(tmp_path):
    module = inv.compile_source(MAP, engine="reference")
    receipt = module.run_main(observe=True)
    rejected("PY_ENGINE_UNAVAILABLE", lambda: module.run_main(artifact=tmp_path / "map.metallibc"))
    assert module.receipt() == receipt


@pytest.mark.parametrize("artifact", [b"INVGMETL", bytearray(b"INVGMETL"), b"/tmp/map.metallibc", BytesPath(), 7, ["map"]])
@pytest.mark.parametrize("engine", ["metal", "auto"])
def test_artifact_is_a_text_path_never_bytes(engine, artifact):
    module = inv.compile_source(MAP, engine=engine)
    rejected("TYPE_MISMATCH", lambda: module.run_main(artifact=artifact), inv.ValidationError)
    assert module.receipt() is None


@pytest.mark.skipif(METAL, reason="a Metal build honors artifact; see the Mac test below")
@pytest.mark.parametrize("engine", ["metal", "auto"])
def test_artifact_without_metal_is_refused_before_any_backend(engine, tmp_path):
    module = inv.compile_source(MAP, engine=engine)
    library = tmp_path / "map.metallibc"
    for artifact in [str(library), library]:
        error = rejected("METAL_UNAVAILABLE", lambda: module.run_main(artifact=artifact))
        assert "artifact cannot be honored" in str(error)
    assert module.receipt() is None
    if engine == "auto":
        # Without the library auto runs; with it, the refusal came before any backend.
        receipt = module.run_main(observe=True)
        assert receipt["result"] == 42
        rejected("METAL_UNAVAILABLE", lambda: module.run_main(artifact=library))
        assert module.receipt() == receipt


CHILD = textwrap.dedent("""
    import json, pathlib, sys
    import invar_python as inv
    source, library, stale = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
    report = {}
    for engine, artifact in (("metal", library), ("auto", str(library))):
        module = inv.compile_source(source, engine=engine)
        receipt = module.run_main(artifact=artifact, observe=True)
        report[engine] = {"result": receipt["result"], "backends": receipt["backends"]}
        try:
            module.run_main(artifact=stale)
        except inv.EngineError as error:
            report[engine + "-stale"] = error.diagnostics
            report[engine + "-stale-receipt"] = module.receipt()
    try:
        inv.compile_source(source, engine="metal").run_main()
    except inv.EngineError as error:
        report["missing"] = error.diagnostics
    # Auto runs a sort on the CPU alone, and still refuses another program's library.
    sort = "fn main() -> i32 { var xs: Array<u32> = [2u32,1u32]; sort(xs); if xs[0u32] == 1u32 { return 42; } return 1; }"
    module = inv.compile_source(sort, engine="auto")
    report["sort"] = module.run_main()
    try:
        module.run_main(artifact=stale)
    except inv.EngineError as error:
        report["sort-stale"] = error.diagnostics
    print(json.dumps(report))
""")


def _require_metal():
    required = os.environ.get("INVAR_REQUIRE_DEVICE") == "metal"
    if not METAL:
        if required:
            pytest.fail("metal: required qualification feature was not built")
        pytest.skip("metal: extension built without the platform feature")
    cli = os.environ.get("INVAR_CLI")
    if not cli:
        if required:
            pytest.fail("metal: INVAR_CLI is required to build the compiled-ahead library")
        pytest.skip("INVAR_CLI is not set to build the compiled-ahead library")
    return cli


def _armed_env(runtime_source):
    env = {key: value for key, value in os.environ.items() if key != RUNTIME_SOURCE}
    env[TRIPWIRE] = "1"
    if runtime_source:
        env[RUNTIME_SOURCE] = "1"
    return env


def test_metal_artifact_runs_without_runtime_source(tmp_path):
    cli = _require_metal()
    stale_source = MAP.replace("wrapping_add(xs[i], 3u32)", "wrapping_add(xs[i], 4u32)")
    paths = {name: metal_library(cli, source, tmp_path / name) for name, source in (("map", MAP), ("stale", stale_source))}
    # The armed child aborts if the binding ever compiles MSL source at run time.
    child = subprocess.run([sys.executable, "-c", CHILD, MAP, str(paths["map"]), str(paths["stale"])],
                           capture_output=True, text=True, env=_armed_env(False), timeout=child_timeout(300))
    assert child.returncode == 0, (child.stdout, child.stderr)
    report = json.loads(child.stdout)
    assert report["metal"] == {"result": 42, "backends": ["metal"]}
    assert report["auto"]["result"] == 42
    assert set(report["auto"]["backends"]) <= {"mir-serial", "cpu-parallel", "metal"}
    for engine in ("metal", "auto"):
        stale = report[engine + "-stale"]
        jsonschema.validate(stale, SCHEMA)
        assert stale["diagnostics"][0]["code"] == "DEVICE_TOOLCHAIN_MISMATCH"
        assert "source_sha256" in json.dumps(stale)
        assert report[engine + "-stale-receipt"] is None
    assert report["sort"] == 42
    jsonschema.validate(report["sort-stale"], SCHEMA)
    assert report["sort-stale"]["diagnostics"][0]["code"] == "DEVICE_TOOLCHAIN_MISMATCH"
    jsonschema.validate(report["missing"], SCHEMA)
    assert report["missing"]["diagnostics"][0]["code"] == "METAL_UNAVAILABLE"
    # The tripwire is live in the binding's process: the developer source route aborts under it.
    tripped = subprocess.run(
        [sys.executable, "-c", "import invar_python as inv, sys; inv.compile_source(sys.argv[1], engine='metal').run_main()", MAP],
        capture_output=True, text=True, env=_armed_env(True), timeout=child_timeout(300))
    assert tripped.returncode != 0
    assert TRIPWIRE in tripped.stderr
