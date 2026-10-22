"""Exercise the hand-authored reference C ABI; not an Invar-generated harness."""

import ctypes as c
import hashlib
import json
import math
import sys
from pathlib import Path


OK, BAD_ARGUMENT, BAD_STATE, BAD_COUNT, NONFINITE = range(5)


class Pose(c.Structure):
    _fields_ = [("translation_m", c.c_double * 3), ("quaternion_xyzw", c.c_double * 4)]


class Config(c.Structure):
    _fields_ = [
        ("model_xml_path", c.POINTER(c.c_uint8)),
        ("model_xml_path_len", c.c_uint32),
        ("model_kind", c.c_uint32),
    ]


def encoded_config(path: bytes, kind: int) -> tuple[Config, c.Array]:
    buffer = (c.c_uint8 * len(path)).from_buffer_copy(path)
    return Config(buffer, len(path), kind), buffer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_pose(actual: Pose, expected: dict) -> None:
    for value, golden in zip(actual.translation_m, expected["translation_m"], strict=True):
        assert math.isclose(value, golden, abs_tol=1e-12, rel_tol=0.0), (value, golden)
    expected_xyzw = [*expected["quaternion_wxyz"][1:], expected["quaternion_wxyz"][0]]
    distance = min(
        math.dist(actual.quaternion_xyzw, expected_xyzw),
        math.dist(actual.quaternion_xyzw, [-value for value in expected_xyzw]),
    )
    assert 2.0 * math.asin(min(1.0, distance / 2.0)) <= 1e-12, distance
    assert math.isclose(math.sqrt(sum(value * value for value in actual.quaternion_xyzw)),
                        1.0, abs_tol=1e-12, rel_tol=0.0)


def main(library: Path, goldens: Path, model_root: Path) -> None:
    lib = c.CDLL(str(library))
    context_ptr = c.POINTER(c.c_void_p)
    lib.invar_pinocchio_v1_version.restype = c.c_char_p
    lib.invar_pinocchio_v1_init.argtypes = [context_ptr]
    lib.invar_pinocchio_v1_configure.argtypes = [c.c_void_p, c.POINTER(Config)]
    lib.invar_pinocchio_v1_step.argtypes = [c.c_void_p, c.POINTER(c.c_double), c.c_uint32, c.POINTER(Pose)]
    lib.invar_pinocchio_v1_shutdown.argtypes = [context_ptr]

    assert lib.invar_pinocchio_v1_version() == b"invar-pinocchio-reference-abi-1/pinocchio-4.1.0"
    assert lib.invar_pinocchio_v1_init(None) == BAD_ARGUMENT
    assert lib.invar_pinocchio_v1_configure(None, None) == BAD_STATE
    assert lib.invar_pinocchio_v1_shutdown(None) == BAD_STATE

    record = json.loads(goldens.read_text())
    assert record["pinocchio_version"] == "4.1.0"
    total = 0
    for arm in record["arms"]:
        model = model_root / arm["model_file"]
        assert sha256(model) == arm["model_sha256"]
        assert sha256(model_root / arm["license_file"]) == arm["license_sha256"]
        count = len(arm["joint_names"])
        context = c.c_void_p()
        pose = Pose((111.0, 222.0, 333.0), (4.0, 5.0, 6.0, 7.0))
        before = bytes(pose)
        sample = (c.c_double * count)(*arm["vectors"][0]["q_rad"])

        assert lib.invar_pinocchio_v1_init(c.byref(context)) == OK and context.value
        assert lib.invar_pinocchio_v1_init(c.byref(context)) == BAD_STATE
        assert lib.invar_pinocchio_v1_step(context, sample, count, c.byref(pose)) == BAD_STATE
        assert bytes(pose) == before
        assert lib.invar_pinocchio_v1_configure(context, None) == BAD_ARGUMENT
        for bad_path in (b"", b"missing.xml", b"bad\x00name.xml", b"\xc0\xaf", b"\xc3", b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"x" * 4097):
            config, keepalive = encoded_config(bad_path, count)
            assert lib.invar_pinocchio_v1_configure(context, c.byref(config)) == BAD_ARGUMENT, bad_path[:20]
        config, keepalive = encoded_config(str(model).encode("utf-8"), 99)
        assert lib.invar_pinocchio_v1_configure(context, c.byref(config)) == BAD_ARGUMENT
        config, keepalive = encoded_config(str(model).encode("utf-8"), count)
        assert lib.invar_pinocchio_v1_configure(context, c.byref(config)) == OK
        for index in range(len(keepalive)):
            keepalive[index] = 0  # Configure consumed the call-scoped path bytes.
        assert lib.invar_pinocchio_v1_configure(context, c.byref(config)) == BAD_STATE

        for joints, size, expected_status in (
            (sample, count - 1, BAD_COUNT),
            (None, count, BAD_ARGUMENT),
            ((c.c_double * count)(math.nan, *arm["vectors"][0]["q_rad"][1:]), count, NONFINITE),
            ((c.c_double * count)(math.inf, *arm["vectors"][0]["q_rad"][1:]), count, NONFINITE),
            ((c.c_double * count)(1000.0, *arm["vectors"][0]["q_rad"][1:]), count, BAD_ARGUMENT),
        ):
            assert lib.invar_pinocchio_v1_step(context, joints, size, c.byref(pose)) == expected_status
            assert bytes(pose) == before
        assert lib.invar_pinocchio_v1_step(context, sample, count, None) == BAD_ARGUMENT
        assert bytes(pose) == before

        for vector in arm["vectors"]:
            joints = (c.c_double * count)(*vector["q_rad"])
            assert lib.invar_pinocchio_v1_step(context, joints, count, c.byref(pose)) == OK
            check_pose(pose, vector)
            total += 1
        assert lib.invar_pinocchio_v1_shutdown(c.byref(context)) == OK and not context.value
        assert lib.invar_pinocchio_v1_shutdown(c.byref(context)) == BAD_STATE
        assert lib.invar_pinocchio_v1_step(context, sample, count, c.byref(pose)) == BAD_STATE

    print(f"Pinocchio reference C ABI: {total} independent goldens, C layouts, lifecycle/input cases passed")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: test_pinocchio_abi.py SHARED_LIBRARY GOLDENS_JSON MODEL_ROOT")
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
