import ctypes
import os
import pathlib
import subprocess
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[3]


def child_timeout(seconds):
    value = os.environ.get("INVAR_TEST_TIMEOUT_SCALE", "1")
    if not value.isascii() or not value.isdecimal() or not 1 <= int(value) <= 64:
        raise ValueError("INVAR_TEST_TIMEOUT_SCALE: expected an integer in 1..=64")
    return seconds * int(value)


def pytest_addoption(parser):
    parser.addoption(
        "--scalar-engine", action="append", choices=["reference", "native-serial"],
        help="Run scalar artifact tests with this engine; repeat for both engines.",
    )


def pytest_generate_tests(metafunc):
    if "scalar_engine" in metafunc.fixturenames:
        engines = metafunc.config.getoption("scalar_engine") or ["reference"]
        metafunc.parametrize("scalar_engine", engines, scope="module")


@pytest.fixture(scope="session")
def c_library(tmp_path_factory):
    directory = tmp_path_factory.mktemp("l1_06_c")
    library = directory / ("fixture.dylib" if sys.platform == "darwin" else "fixture.so")
    subprocess.run(
        ["cc", "-dynamiclib" if sys.platform == "darwin" else "-shared", "-fPIC",
         "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "include"),
         str(pathlib.Path(__file__).with_name("fixture.c")), "-o", str(library)],
        check=True, timeout=child_timeout(30),
    )
    # Keep this trusted test fixture loaded so its counters survive the production
    # adapter's load/call/unload cycle. All tested calls still go through PyO3.
    loaded = ctypes.CDLL(str(library))
    yield str(library)
    del loaded
