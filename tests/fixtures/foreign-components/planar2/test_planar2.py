"""Direct private-fixture check; not the generated Invar contract harness."""

import ctypes as c
import json
import math
import sys
from pathlib import Path


class Arm(c.Structure):
    _fields_ = [("state", c.c_uint), ("proximal_m", c.c_double), ("distal_m", c.c_double)]


def main(library_path: Path, golden_path: Path) -> None:
    lib = c.CDLL(str(library_path))
    arm_ptr = c.POINTER(Arm)
    dbl_ptr = c.POINTER(c.c_double)
    lib.fixture_planar2_init.argtypes = [arm_ptr]
    lib.fixture_planar2_configure.argtypes = [arm_ptr, c.c_double, c.c_double]
    lib.fixture_planar2_step.argtypes = [arm_ptr, dbl_ptr, c.c_size_t, dbl_ptr, dbl_ptr, dbl_ptr]
    lib.fixture_planar2_shutdown.argtypes = [arm_ptr]
    lib.fixture_planar2_version.restype = c.c_char_p

    ok, bad_state, bad_input = 0, 1, 2
    arm = Arm()
    q = (c.c_double * 2)(0.0, 0.0)
    x, y, yaw = c.c_double(123.0), c.c_double(456.0), c.c_double(789.0)

    assert lib.fixture_planar2_version() == b"planar2-fixture-1"
    assert lib.fixture_planar2_init(None) == bad_input
    assert lib.fixture_planar2_configure(c.byref(arm), 0.4, 0.3) == bad_state
    assert lib.fixture_planar2_step(c.byref(arm), q, 2, c.byref(x), c.byref(y), c.byref(yaw)) == bad_state
    assert lib.fixture_planar2_shutdown(c.byref(arm)) == bad_state
    assert lib.fixture_planar2_init(c.byref(arm)) == ok
    assert lib.fixture_planar2_init(c.byref(arm)) == bad_state
    assert lib.fixture_planar2_step(c.byref(arm), q, 2, c.byref(x), c.byref(y), c.byref(yaw)) == bad_state
    for lengths in ((0.0, 0.3), (-0.4, 0.3), (math.nan, 0.3), (0.4, math.inf)):
        assert lib.fixture_planar2_configure(c.byref(arm), *lengths) == bad_input

    goldens = json.loads(golden_path.read_text())
    assert goldens["seed"] == 20260927
    assert lib.fixture_planar2_configure(c.byref(arm), *goldens["lengths_m"]) == ok
    assert lib.fixture_planar2_configure(c.byref(arm), 0.4, 0.3) == bad_state
    assert lib.fixture_planar2_step(c.byref(arm), q, 1, c.byref(x), c.byref(y), c.byref(yaw)) == bad_input
    assert lib.fixture_planar2_step(c.byref(arm), None, 2, c.byref(x), c.byref(y), c.byref(yaw)) == bad_input
    assert lib.fixture_planar2_step(c.byref(arm), q, 2, None, c.byref(y), c.byref(yaw)) == bad_input
    assert (x.value, y.value, yaw.value) == (123.0, 456.0, 789.0)
    for invalid in (math.nan, math.inf, -math.inf):
        bad_q = (c.c_double * 2)(invalid, 0.0)
        assert lib.fixture_planar2_step(c.byref(arm), bad_q, 2, c.byref(x), c.byref(y), c.byref(yaw)) == bad_input
    huge_q = (c.c_double * 2)(1e308, 1e308)
    assert lib.fixture_planar2_step(c.byref(arm), huge_q, 2, c.byref(x), c.byref(y), c.byref(yaw)) == bad_input
    assert (x.value, y.value, yaw.value) == (123.0, 456.0, 789.0)

    for vector in goldens["vectors"]:
        q = (c.c_double * 2)(*vector["q_rad"])
        assert lib.fixture_planar2_step(c.byref(arm), q, 2, c.byref(x), c.byref(y), c.byref(yaw)) == ok
        for actual, key in ((x.value, "x_m"), (y.value, "y_m"), (yaw.value, "yaw_rad")):
            assert math.isclose(actual, vector[key], rel_tol=0.0, abs_tol=1e-12), (actual, vector)

    assert lib.fixture_planar2_shutdown(c.byref(arm)) == ok
    assert lib.fixture_planar2_step(c.byref(arm), q, 2, c.byref(x), c.byref(y), c.byref(yaw)) == bad_state
    assert lib.fixture_planar2_shutdown(c.byref(arm)) == bad_state

    huge_arm = Arm()
    assert lib.fixture_planar2_init(c.byref(huge_arm)) == ok
    assert lib.fixture_planar2_configure(c.byref(huge_arm), 1e308, 1e308) == ok
    before = (x.value, y.value, yaw.value)
    assert lib.fixture_planar2_step(c.byref(huge_arm), (c.c_double * 2)(0.0, 0.0), 2,
                                    c.byref(x), c.byref(y), c.byref(yaw)) == bad_input
    assert (x.value, y.value, yaw.value) == before
    assert lib.fixture_planar2_shutdown(c.byref(huge_arm)) == ok
    print(f"planar2: {len(goldens['vectors'])} independent goldens and lifecycle/input cases passed")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_planar2.py LIBRARY GOLDENS_JSON")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
