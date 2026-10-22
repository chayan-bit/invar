"""Direct generated-ABI bridge check with the real INVP fixture child."""

import ctypes as c
import os
import tempfile
from pathlib import Path
import sys


class Config(c.Structure):
    _fields_ = [("model_xml_path", c.POINTER(c.c_uint8)),
                ("model_xml_path_len", c.c_uint32), ("model_kind", c.c_uint32)]


class Pose(c.Structure):
    _fields_ = [("translation_m", c.c_double * 3),
                ("quaternion_xyzw", c.c_double * 4)]


class Inputs(c.Structure):
    _fields_ = [("q", c.c_double * 7)]


class Outputs(c.Structure):
    _fields_ = [("pose", Pose)]


def bytes_buffer(value: bytes) -> c.Array:
    return (c.c_uint8 * len(value)).from_buffer_copy(value)


def case(lib: c.CDLL, child: Path, mode: str, expected: int) -> None:
    with tempfile.TemporaryDirectory(prefix="invar-process-bridge-") as folder:
        pidfile = Path(folder) / "pid"
        os.environ["INVAR_PROCESS_FIXTURE_MODE"] = mode
        os.environ["INVAR_PROCESS_FIXTURE_PIDFILE"] = str(pidfile)
        context = c.c_void_p()
        executable = bytes_buffer(os.fsencode(child))
        assert lib.invar_process_bridge_v1_init(c.byref(context), executable,
                                                len(executable)) == 0 and context.value
        try:
            pid = int(pidfile.read_text())
            q = Inputs((1, 2, 3, 0, 0, 0, 0))
            output = Outputs(Pose((111, 222, 333), (4, 5, 6, 7)))
            before = bytes(output)
            assert lib.invar_process_bridge_v1_step(context, c.byref(q), c.byref(output)) == 2
            assert bytes(output) == before
            model = bytes_buffer(b"/models/deterministic-fixture.xml")
            config = Config(model, len(model), 7)
            assert lib.invar_process_bridge_v1_configure(context, c.byref(config)) == 0
            assert lib.invar_process_bridge_v1_step(context, c.byref(q), c.byref(output)) == expected
            if expected:
                assert bytes(output) == before
            else:
                assert list(output.pose.translation_m) == [1, 2, 3]
                assert list(output.pose.quaternion_xyzw) == [0, 0, 0, 1]
            assert lib.invar_process_bridge_v1_shutdown(c.byref(context)) == (4 if expected else 0)
        finally:
            if context.value:
                lib.invar_process_bridge_v1_shutdown(c.byref(context))
        assert not context.value
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        else:
            raise AssertionError("bridge child was not reaped")


def main(library: Path, child: Path) -> None:
    lib = c.CDLL(str(library))
    lib.invar_process_bridge_v1_version.restype = c.c_char_p
    lib.invar_process_bridge_v1_init.argtypes = [c.POINTER(c.c_void_p),
                                                 c.POINTER(c.c_uint8), c.c_uint32]
    lib.invar_process_bridge_v1_configure.argtypes = [c.c_void_p, c.POINTER(Config)]
    lib.invar_process_bridge_v1_step.argtypes = [c.c_void_p, c.POINTER(Inputs),
                                                c.POINTER(Outputs)]
    lib.invar_process_bridge_v1_shutdown.argtypes = [c.POINTER(c.c_void_p)]
    assert lib.invar_process_bridge_v1_version() == b"invar-component-v1/invar-process-reference-abi-1"
    try:
        case(lib, child, "normal", 0)
        case(lib, child, "crash_step", 4)
    finally:
        os.environ.pop("INVAR_PROCESS_FIXTURE_MODE", None)
        os.environ.pop("INVAR_PROCESS_FIXTURE_PIDFILE", None)
    print("process bridge: real child lifecycle, pose, crash, and reap passed")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_bridge.py BRIDGE_LIBRARY EXECUTABLE_CHILD")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
