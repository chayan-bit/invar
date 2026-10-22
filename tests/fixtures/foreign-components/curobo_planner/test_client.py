#!/usr/bin/env python3
"""Direct ABI/protocol check in the pinned cuRobo venv; no GPU work required."""

import ctypes as c
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[4] /
                       "examples/cells/reach_place/components/curobo_planner"))
from planner_server import (HELD, HELD_COVER_SPHERES, HOME, PINNED_MODELS,
                            RealPlanner, single_plan, verify_pinned_model)

LIB = c.CDLL(sys.argv[1])
SERVER = Path(__file__).with_name("fake_server.py").resolve()


class Pose(c.Structure):
    _fields_ = [("xyz_m", c.c_double * 3), ("quaternion_xyzw", c.c_double * 4)]


class Text(c.Structure):
    _fields_ = [("bytes", c.POINTER(c.c_uint8)), ("length", c.c_uint16)]


class Config(c.Structure):
    _fields_ = [("robot_kind", c.c_uint32), ("joint_count", c.c_uint32),
                ("model_sha256", c.c_uint8 * 32),
                ("robot_config_sha256", c.c_uint8 * 32),
                ("scene_sha256", c.c_uint8 * 32),
                ("joint_limits_rad", (c.c_double * 2) * 7),
                ("table_pose", Pose), ("table_half_extents_m", c.c_double * 3),
                ("cube_edge_m", c.c_double), ("base_frame", Text),
                ("tool_frame", Text), ("joint_names", Text * 7),
                ("robot_config_path", Text), ("scene_config_path", Text)]


class Request(c.Structure):
    _fields_ = [("joint_count", c.c_uint32), ("start_q_rad", c.c_double * 7),
                ("goal_q_rad", c.c_double * 7),
                ("target_base_from_tool", Pose), ("cube_base_pose", Pose),
                ("tool_from_cube", Pose), ("phase", c.c_uint32)]


class Waypoint(c.Structure):
    _fields_ = [("time_from_start_s", c.c_double), ("q_rad", c.c_double * 7),
                ("dq_rad_s", c.c_double * 7)]


class Result(c.Structure):
    _fields_ = [("dt_s", c.c_double), ("count", c.c_uint32),
                ("waypoints", Waypoint * 256)]


Context = c.c_void_p
LIB.invar_planner_v1_init.argtypes = [c.POINTER(Context), c.POINTER(c.c_uint8), c.c_uint32]
LIB.invar_planner_v1_configure.argtypes = [Context, c.POINTER(Config)]
LIB.invar_planner_v1_plan.argtypes = [Context, c.POINTER(Request), c.POINTER(Result)]
LIB.invar_planner_v1_cancel.argtypes = [Context]
LIB.invar_planner_v1_shutdown.argtypes = [c.POINTER(Context)]


def text(value):
    data = value.encode()
    owned = (c.c_uint8 * len(data)).from_buffer_copy(data)
    return Text(owned, len(data)), owned


def pose(x=0):
    return Pose((x, 0, 0), (0, 0, 0, 1))


def configured(count):
    context = Context()
    server = str(SERVER).encode()
    path = (c.c_uint8 * len(server)).from_buffer_copy(server)
    assert LIB.invar_planner_v1_init(c.byref(context), path, len(server)) == 0
    config = Config()
    config.robot_kind = config.joint_count = count
    config.table_pose = pose()
    config.table_half_extents_m[:] = (0.5, 0.5, 0.05)
    config.cube_edge_m = 0.05
    keep = []
    for field, value in (("base_frame", "base"), ("tool_frame", "tool"),
                         ("robot_config_path", "/tmp/robot.yml"),
                         ("scene_config_path", "/tmp/scene.yml")):
        item, owned = text(value)
        setattr(config, field, item)
        keep.append(owned)
    for i in range(count):
        config.joint_limits_rad[i][:] = (-2, 2)
        item, owned = text(f"joint{i+1}")
        config.joint_names[i] = item
        keep.append(owned)
    assert LIB.invar_planner_v1_configure(context, c.byref(config)) == 0
    request = Request()
    request.joint_count = count
    request.target_base_from_tool = pose(0.4)
    request.cube_base_pose = pose(0.5)
    request.tool_from_cube = pose(0.1)
    return context, request


def check_normal(count):
    context, request = configured(count)
    result = Result()
    result.count = 987
    request.phase = HELD
    assert LIB.invar_planner_v1_plan(context, c.byref(request), c.byref(result)) == 0
    assert result.count == 2 and result.dt_s == 0.1
    assert result.waypoints[1].time_from_start_s == 0.1
    before = c.string_at(c.byref(result), c.sizeof(result))
    # Only a home request carries a joint goal; it must still lie inside the limits.
    request.goal_q_rad[0] = 0.5
    assert LIB.invar_planner_v1_plan(context, c.byref(request), c.byref(result)) == 1
    request.goal_q_rad[0] = -0.0
    assert LIB.invar_planner_v1_plan(context, c.byref(request), c.byref(result)) == 1
    assert c.string_at(c.byref(result), c.sizeof(result)) == before
    request.phase = HOME
    request.goal_q_rad[0] = 0.5
    assert LIB.invar_planner_v1_plan(context, c.byref(request), c.byref(result)) == 0
    request.goal_q_rad[0] = 3
    assert LIB.invar_planner_v1_plan(context, c.byref(request), c.byref(result)) == 1
    request.goal_q_rad[0] = 0
    request.phase = HOME + 1
    assert LIB.invar_planner_v1_plan(context, c.byref(request), c.byref(result)) == 1
    request.phase = HELD
    request.start_q_rad[0] = math.nan
    before = c.string_at(c.byref(result), c.sizeof(result))
    assert LIB.invar_planner_v1_plan(context, c.byref(request), c.byref(result)) == 1
    assert c.string_at(c.byref(result), c.sizeof(result)) == before
    request.start_q_rad[0] = 3
    assert LIB.invar_planner_v1_plan(context, c.byref(request), c.byref(result)) == 1
    assert c.string_at(c.byref(result), c.sizeof(result)) == before
    assert LIB.invar_planner_v1_shutdown(c.byref(context)) == 0 and not context.value


def check_failure(mode, expected, count=7):
    os.environ["INVAR_PLANNER_FAKE_MODE"] = mode
    context, request = configured(count)
    result = Result()
    result.count = 987
    before = c.string_at(c.byref(result), c.sizeof(result))
    assert LIB.invar_planner_v1_plan(context, c.byref(request), c.byref(result)) == expected
    assert c.string_at(c.byref(result), c.sizeof(result)) == before
    assert LIB.invar_planner_v1_shutdown(c.byref(context)) == 4
    os.environ.pop("INVAR_PLANNER_FAKE_MODE")


def check_edge(count):
    os.environ["INVAR_PLANNER_FAKE_MODE"] = "edge"
    context, request = configured(count)
    result = Result()
    assert LIB.invar_planner_v1_plan(context, c.byref(request), c.byref(result)) == 0
    assert result.waypoints[1].q_rad[count - 1] == 2.0
    assert result.waypoints[1].dq_rad_s[count - 1] == (6.28 if count == 6 else 2.61)
    assert LIB.invar_planner_v1_shutdown(c.byref(context)) == 0
    os.environ.pop("INVAR_PLANNER_FAKE_MODE")


def check_model_mapping(count):
    digest, base, tool, names, limits = PINNED_MODELS[count]
    config = {"kind": count, "digests": [bytes.fromhex(digest)], "base": base,
              "tool": tool, "names": list(names), "limits": list(limits)}
    verify_pinned_model(config)
    if count == 7:
        # Reach the cover guard with valid file digests and frame/attachment
        # fields, rather than failing earlier on an incomplete configuration.
        with tempfile.TemporaryDirectory() as directory:
            robot, scene = Path(directory) / "robot.json", Path(directory) / "scene.json"
            robot.write_text(json.dumps({"robot_cfg": {"kinematics": {
                "base_link": base, "tool_frames": [tool],
                "cspace": {"joint_names": list(names), "position_limit_clip": None},
                "collision_link_names": [base, "attached_object"],
                "collision_spheres": {base: []},
                "extra_collision_spheres": {"attached_object": HELD_COVER_SPHERES},
                "extra_links": {"attached_object": {
                    "parent_link_name": tool, "fixed_transform": [0, 0, 0, 1, 0, 0, 0]}},
            }}}))
            scene.write_text("{}")
            candidate = config | {"count": count, "robot_path": str(robot), "scene_path": str(scene),
                                  "digests": [bytes.fromhex(digest),
                                              hashlib.sha256(robot.read_bytes()).digest(),
                                              hashlib.sha256(scene.read_bytes()).digest()]}
            try:
                RealPlanner(candidate)
            except ValueError as error:
                assert "collision spheres" in str(error)
            else:
                raise AssertionError("Panda planner accepted unverified spheres")
    for field, value in (("names", list(reversed(names))),
                         ("limits", [(-10.0, 10.0), *limits[1:]]),
                         ("digests", [bytes(32)])):
        invalid = config | {field: value}
        try:
            verify_pinned_model(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted wrong pinned model {field}")


def check_cancel():
    os.environ["INVAR_PLANNER_FAKE_MODE"] = "hang"
    with tempfile.TemporaryDirectory() as directory:
        entered = Path(directory) / "entered"
        os.environ["INVAR_PLANNER_FAKE_ENTERED"] = str(entered)
        context, request = configured(7)
        result = Result()
        result.count = 987
        status = []
        worker = threading.Thread(target=lambda: status.append(
            LIB.invar_planner_v1_plan(context, c.byref(request), c.byref(result))))
        worker.start()
        limit = time.monotonic() + 2
        while not entered.exists() and time.monotonic() < limit:
            time.sleep(0.01)
        assert entered.exists()
        assert LIB.invar_planner_v1_cancel(context) == 7
        worker.join(timeout=2)
        assert status == [7] and result.count == 987
        assert LIB.invar_planner_v1_shutdown(c.byref(context)) == 4
    os.environ.pop("INVAR_PLANNER_FAKE_MODE")
    os.environ.pop("INVAR_PLANNER_FAKE_ENTERED")


check_normal(6)
check_normal(7)
# cuRobo 0.8.0 shape [batch][goal set][time][joint]: exactly one plan is unpacked.
assert single_plan([[[[0.0, 1.0], [2.0, 3.0]]]]) == [[0.0, 1.0], [2.0, 3.0]]
for rows in ([[[[0.0]]], [[[1.0]]]], [[[[0.0]], [[1.0]]]], []):
    try:
        single_plan(rows)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted plan batch {rows}")
for count in (6, 7):
    check_model_mapping(count)
    check_edge(count)
    for mode, status in (("outside_q", 3), ("outside_dq", 3),
                         ("reply_bad_q", 6), ("reply_bad_dq", 6)):
        check_failure(mode, status, count)
check_failure("fail", 3)
check_failure("crash", 4)
check_failure("bad_reply", 6)
check_failure("reply_bad_length", 6)
check_failure("reply_bad_version", 6)
check_failure("hang", 5)
check_cancel()
print("planner boundary: 6/7 DoF, phases and home joint goal, position/speed edges and violations, failure, malformed, timeout, cancellation, unchanged output OK")
