"""Direct reference client/child checks; not an Invar-generated harness."""

import ctypes as c
import math
import os
import signal
import socket
import subprocess
import struct
import tempfile
import threading
import time
from pathlib import Path
import sys


class Pose(c.Structure):
    _fields_ = [("translation_m", c.c_double * 3), ("quaternion_xyzw", c.c_double * 4)]


class Config(c.Structure):
    _fields_ = [
        ("model_xml_path", c.POINTER(c.c_uint8)),
        ("model_xml_path_len", c.c_uint32),
        ("model_kind", c.c_uint32),
    ]


def buffer(data: bytes) -> c.Array:
    return (c.c_uint8 * len(data)).from_buffer_copy(data)


def sentinel() -> Pose:
    return Pose((111.0, 222.0, 333.0), (4.0, 5.0, 6.0, 7.0))


def assert_reaped(pid: int) -> None:
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return
    raise AssertionError(f"child {pid} was not reaped by the C client")


def init(lib: c.CDLL, server: Path) -> c.c_void_p:
    context = c.c_void_p()
    path = buffer(os.fsencode(server))
    assert lib.invar_process_v1_init(c.byref(context), path, len(path)) == 0
    assert context.value
    return context


def configure(lib: c.CDLL, context: c.c_void_p, kind: int) -> None:
    path = buffer(b"/models/deterministic-fixture.xml")
    config = Config(path, len(path), kind)
    assert lib.invar_process_v1_configure(context, c.byref(config)) == 0


def child_case(lib: c.CDLL, server: Path, mode: str, expected: int) -> None:
    with tempfile.TemporaryDirectory(prefix="invar-process-fixture-") as folder:
        pidfile = Path(folder) / "pid"
        os.environ["INVAR_PROCESS_FIXTURE_MODE"] = mode
        os.environ["INVAR_PROCESS_FIXTURE_PIDFILE"] = str(pidfile)
        context = init(lib, server)
        pid = int(pidfile.read_text())
        configure(lib, context, 7)
        pose = sentinel()
        before = bytes(pose)
        joints = (c.c_double * 7)(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0)
        started = time.monotonic()
        status = lib.invar_process_v1_step(context, joints, 7, c.byref(pose))
        assert status == expected, (mode, status, expected)
        assert time.monotonic() - started < 5.0, mode
        if expected:
            assert bytes(pose) == before, mode
            assert lib.invar_process_v1_shutdown(c.byref(context)) == 4
        else:
            assert list(pose.translation_m) == [1.0, 2.0, 3.0]
            assert list(pose.quaternion_xyzw) == [0.0, 0.0, 0.0, 1.0]
            assert lib.invar_process_v1_shutdown(c.byref(context)) == 0
        assert not context.value
        assert_reaped(pid)


def killed_mid_step(lib: c.CDLL, server: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="invar-process-fixture-") as folder:
        pidfile = Path(folder) / "pid"
        stepping_file = Path(folder) / "stepping"
        os.environ["INVAR_PROCESS_FIXTURE_MODE"] = "hang_step"
        os.environ["INVAR_PROCESS_FIXTURE_PIDFILE"] = str(pidfile)
        os.environ["INVAR_PROCESS_FIXTURE_STEPPING_FILE"] = str(stepping_file)
        context = init(lib, server)
        pid = int(pidfile.read_text())
        configure(lib, context, 7)
        pose = sentinel()
        before = bytes(pose)
        joints = (c.c_double * 7)(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0)
        result = {}
        call = threading.Thread(target=lambda: result.setdefault(
            "status", lib.invar_process_v1_step(context, joints, 7, c.byref(pose))))
        call.start()
        deadline = time.monotonic() + 1.0
        while not stepping_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert stepping_file.exists(), "child did not enter step"
        os.kill(pid, signal.SIGKILL)
        call.join(timeout=5)
        assert not call.is_alive() and result.get("status") == 4, result
        assert bytes(pose) == before
        assert lib.invar_process_v1_shutdown(c.byref(context)) == 4 and not context.value
        assert_reaped(pid)
        os.environ.pop("INVAR_PROCESS_FIXTURE_STEPPING_FILE", None)


def server_boundary(server: Path) -> None:
    parent, child = socket.socketpair()
    if child.fileno() == 3:
        duplicate = os.dup(child.fileno())
        child.close()
        child = socket.socket(fileno=duplicate)
    actions = [(os.POSIX_SPAWN_DUP2, child.fileno(), 3)]
    if child.fileno() != 3:
        actions.append((os.POSIX_SPAWN_CLOSE, child.fileno()))
    env = os.environ.copy()
    for key in ("INVAR_PROCESS_FIXTURE_MODE", "INVAR_PROCESS_FIXTURE_PIDFILE",
                "INVAR_PROCESS_FIXTURE_STEPPING_FILE"):
        env.pop(key, None)
    pid = os.posix_spawn(str(server), [str(server)], env, file_actions=actions)
    child.close()
    parent.settimeout(2)

    def receive(size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = parent.recv(size - len(data))
            assert chunk, "fixture child closed during response"
            data.extend(chunk)
        return bytes(data)

    def exchange(op: int, sequence: int, body: bytes = b"") -> int:
        payload = struct.pack("<4sHBBIi", b"INVP", 1, op, 0, sequence, 0) + body
        parent.sendall(struct.pack("<I", len(payload)) + payload)
        (length,) = struct.unpack("<I", receive(4))
        response = receive(length)
        magic, version, reply_op, role, reply_sequence, status = struct.unpack_from("<4sHBBIi", response)
        assert (magic, version, reply_op, role, reply_sequence) == (b"INVP", 1, op, 1, sequence)
        assert len(response) == 16
        return status

    try:
        assert exchange(1, 1) == 0
        for sequence, bad in enumerate((b"\xc0\xaf", b"x" * 4097), start=2):
            body = struct.pack("<II", 7, len(bad)) + bad
            assert exchange(2, sequence, body) == 1
        path = b"/models/deterministic-fixture.xml"
        assert exchange(2, 4, struct.pack("<II", 7, len(path)) + path) == 0
        assert exchange(3, 5, struct.pack("<I6d", 6, *([0.0] * 6))) == 1
        assert exchange(3, 6, struct.pack("<I7d", 7, math.nan, *([0.0] * 6))) == 1
        assert exchange(4, 7) == 0
    finally:
        parent.close()
        waited, status = os.waitpid(pid, 0)
        assert waited == pid and os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


def main(library: Path, server: Path) -> None:
    output = subprocess.check_output(["nm", "-D", "--defined-only", str(library)], text=True)
    exported = {line.split()[-1] for line in output.splitlines() if line.split()}
    assert exported == {f"invar_process_v1_{name}" for name in
                        ("version", "init", "configure", "step", "shutdown")}, exported
    lib = c.CDLL(str(library))
    context_ptr = c.POINTER(c.c_void_p)
    byte_ptr = c.POINTER(c.c_uint8)
    lib.invar_process_v1_version.restype = c.c_char_p
    lib.invar_process_v1_init.argtypes = [context_ptr, byte_ptr, c.c_uint32]
    lib.invar_process_v1_configure.argtypes = [c.c_void_p, c.POINTER(Config)]
    lib.invar_process_v1_step.argtypes = [c.c_void_p, c.POINTER(c.c_double), c.c_uint32, c.POINTER(Pose)]
    lib.invar_process_v1_shutdown.argtypes = [context_ptr]

    assert lib.invar_process_v1_version() == b"invar-process-reference-abi-1"
    assert lib.invar_process_v1_init(None, None, 0) == 1
    assert lib.invar_process_v1_configure(None, None) == 2
    assert lib.invar_process_v1_shutdown(None) == 2
    for bad in (b"relative", b"/missing/invar-fixture", b"/bad\x00path", b"/\xc0\xaf", b"/" + b"x" * 4096):
        path = buffer(bad)
        context = c.c_void_p()
        assert lib.invar_process_v1_init(c.byref(context), path, len(path)) == 1, bad[:20]
        assert not context.value

    with tempfile.TemporaryDirectory(prefix="invar-process-fixture-") as folder:
        pidfile = Path(folder) / "pid"
        os.environ["INVAR_PROCESS_FIXTURE_MODE"] = "normal"
        os.environ["INVAR_PROCESS_FIXTURE_PIDFILE"] = str(pidfile)
        context = init(lib, server)
        pid = int(pidfile.read_text())
        executable = buffer(os.fsencode(server))
        assert lib.invar_process_v1_init(c.byref(context), executable, len(executable)) == 2
        pose = sentinel()
        before = bytes(pose)
        joints = (c.c_double * 6)(1.0, 2.0, 3.0, 0.0, 0.0, 0.0)
        assert lib.invar_process_v1_step(context, joints, 6, c.byref(pose)) == 2
        assert bytes(pose) == before
        assert lib.invar_process_v1_configure(context, None) == 1
        for bad in (b"", b"bad\x00path", b"\xc0\xaf", b"x" * 4097):
            path = buffer(bad)
            config = Config(path, len(path), 6)
            assert lib.invar_process_v1_configure(context, c.byref(config)) == 1
        path = buffer(b"/models/deterministic-fixture.xml")
        assert lib.invar_process_v1_configure(context, c.byref(Config(path, len(path), 8))) == 1
        configure(lib, context, 6)
        assert lib.invar_process_v1_configure(context, c.byref(Config(path, len(path), 6))) == 2
        for q, count in ((None, 6), (joints, 7), ((c.c_double * 6)(math.nan, 0, 0, 0, 0, 0), 6),
                         ((c.c_double * 6)(math.inf, 0, 0, 0, 0, 0), 6)):
            assert lib.invar_process_v1_step(context, q, count, c.byref(pose)) == 1
            assert bytes(pose) == before
        assert lib.invar_process_v1_step(context, joints, 6, None) == 1
        assert lib.invar_process_v1_step(context, joints, 6, c.byref(pose)) == 0
        assert list(pose.translation_m) == [1.0, 2.0, 3.0]
        assert list(pose.quaternion_xyzw) == [0.0, 0.0, 0.0, 1.0]
        assert lib.invar_process_v1_shutdown(c.byref(context)) == 0 and not context.value
        assert lib.invar_process_v1_shutdown(c.byref(context)) == 2
        assert_reaped(pid)

    for mode, status in (
        ("component_error_step", 3), ("crash_step", 4), ("hang_step", 5),
        ("truncated_step", 6), ("oversized_step", 6), ("bad_magic_step", 6),
        ("bad_version_step", 6), ("bad_op_step", 6), ("bad_role_step", 6),
        ("bad_sequence_step", 6), ("bad_status_step", 6),
        ("bad_body_step", 6), ("bad_failure_body_step", 6),
        ("nan_pose_step", 6), ("bad_quaternion_step", 6),
    ):
        child_case(lib, server, mode, status)

    killed_mid_step(lib, server)

    for mode, status in (("hang_shutdown", 5), ("crash_shutdown", 4)):
        with tempfile.TemporaryDirectory(prefix="invar-process-fixture-") as folder:
            pidfile = Path(folder) / "pid"
            os.environ["INVAR_PROCESS_FIXTURE_MODE"] = mode
            os.environ["INVAR_PROCESS_FIXTURE_PIDFILE"] = str(pidfile)
            context = init(lib, server)
            pid = int(pidfile.read_text())
            configure(lib, context, 6)
            assert lib.invar_process_v1_shutdown(c.byref(context)) == status
            assert not context.value
            assert_reaped(pid)

    server_boundary(server)

    os.environ.pop("INVAR_PROCESS_FIXTURE_MODE", None)
    os.environ.pop("INVAR_PROCESS_FIXTURE_PIDFILE", None)
    print("INVP process fixture: lifecycle, bounds, faults, unchanged output, and child reaping passed")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_client.py SHARED_LIBRARY FIXTURE_SERVER")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
