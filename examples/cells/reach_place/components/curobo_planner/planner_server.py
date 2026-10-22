#!/usr/bin/env python3
"""INVP v3 cuRobo process adapter. GPU imports occur only on configure."""

import hashlib
import json
import math
import os
from pathlib import Path
import struct

FD = 3
HEADER = struct.Struct("<4sHBBIi")
VERSION = 3
MAX_FRAME = 65536
MAX_WAYPOINTS = 256

# Pinned URDF active-joint velocity limits; not a general foreign-component policy.
VELOCITY_LIMITS = {6: (3.14, 3.14, 3.14, 6.28, 6.28, 6.28),
                   7: (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)}
PINNED_MODELS = {
    6: ("5e28414e03835995fe9fed160a164718c17607e33af0b7031097bd730542aae2",
        "base_link", "tool0",
        ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"),
        ((-6.283185307179586, 6.283185307179586),) * 2 +
        ((-3.1415, 3.1415),) +
        ((-6.283185307179586, 6.283185307179586),) * 3),
    7: ("eeecd06212a1ed160807243a1046ac0a1ce07d7e49370a4a17d181bed5a6f351",
        "panda_link0", "panda_hand",
        tuple(f"panda_joint{i}" for i in range(1, 8)),
        ((-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
         (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525),
         (-2.8973, 2.8973))),
}
# The Panda cover prepare_models.py emits from the pinned cuRobo fit; check_panda_spheres.py
# proves it covers the accepted Menagerie collision geometry within 2 mm (decision 2026-10-14).
PANDA_COVER_SHA256 = "20be84cc61a606b80b3e7ac2f7c6bea96150245180f2755a1e1552e7bbbbe5e5"
POSITION_LIMIT_CLIP = {6: [0, 0, math.pi - 3.1415, 0, 0, 0], 7: None}

# Planner world phases (planner-component-v2); the child keeps one world model per phase.
FREE, APPROACH, HELD, HELD_CONTACT, HOME = range(5)
# Held-cube cover: circumspheres of the outer cells of a k^3 split of the cube.
HELD_COVER_K = 4
HELD_COVER_SPHERES = HELD_COVER_K ** 3 - (HELD_COVER_K - 2) ** 3
# In HELD_CONTACT the layer facing the table is pulled inside the cube until its lowest
# point is this far above the contact face, uncovering only the table-contact slab.
CONTACT_CLEARANCE_M = 0.0005


def exact(size):
    chunks = bytearray()
    while len(chunks) < size:
        part = os.read(FD, size - len(chunks))
        if not part:
            if not chunks:
                return None
            raise EOFError("truncated frame")
        chunks.extend(part)
    return bytes(chunks)


def frame():
    prefix = exact(4)
    if prefix is None:
        return None
    (size,) = struct.unpack("<I", prefix)
    if not HEADER.size <= size <= MAX_FRAME:
        raise ValueError("frame length")
    return exact(size)


def send(op, sequence, status=0, body=b""):
    payload = HEADER.pack(b"INVP", VERSION, op, 1, sequence, status) + body
    if len(payload) > MAX_FRAME:
        raise ValueError("reply capacity")
    data = memoryview(struct.pack("<I", len(payload)) + payload)
    while data:
        size = os.write(FD, data)
        if size <= 0:
            raise OSError("short write")
        data = data[size:]


def pose_valid(pose):
    return (len(pose) == 7 and all(math.isfinite(x) for x in pose)
            and abs(math.sqrt(sum(x * x for x in pose[3:])) - 1) <= 1e-12)


def wxyz(pose):
    return [*pose[:3], pose[6], *pose[3:6]]


def compose(a, b):
    x, y, z, w = a[3:]
    vx, vy, vz = b[:3]
    tx, ty, tz = 2*(y*vz-z*vy), 2*(z*vx-x*vz), 2*(x*vy-y*vx)
    point = (a[0] + vx + w*tx + y*tz-z*ty,
             a[1] + vy + w*ty + z*tx-x*tz,
             a[2] + vz + w*tz + x*ty-y*tx)
    bx, by, bz, bw = b[3:]
    return (*point, w*bx + x*bw + y*bz-z*by,
            w*by - x*bz + y*bw + z*bx,
            w*bz + x*by - y*bx + z*bw,
            w*bw - x*bx - y*by - z*bz)


class Reader:
    def __init__(self, data):
        self.data = data
        self.at = 0

    def take(self, fmt):
        spec = struct.Struct("<" + fmt)
        if self.at + spec.size > len(self.data):
            raise ValueError("short body")
        result = spec.unpack_from(self.data, self.at)
        self.at += spec.size
        return result if len(result) != 1 else result[0]

    def text(self, limit, absolute=False):
        size = self.take("H")
        if not 0 < size <= limit or self.at + size > len(self.data):
            raise ValueError("text length")
        raw = self.data[self.at:self.at + size]
        self.at += size
        value = raw.decode("utf-8", errors="strict")
        if "\0" in value or (absolute and not value.startswith("/")):
            raise ValueError("text value")
        return value

    def done(self):
        if self.at != len(self.data):
            raise ValueError("trailing body")


def parse_config(body):
    r = Reader(body)
    kind, count = r.take("II")
    if kind != count or count not in (6, 7):
        raise ValueError("robot kind")
    digests = [r.take("32s") for _ in range(3)]
    limits = [r.take("dd") for _ in range(count)]
    table = r.take("7d")
    half = r.take("3d")
    cube_edge = r.take("d")
    base, tool = r.text(64), r.text(64)
    names = [r.text(64) for _ in range(count)]
    robot_path, scene_path = r.text(4096, True), r.text(4096, True)
    r.done()
    if (not pose_valid(table) or not all(math.isfinite(x) and x > 0 for x in half)
            or cube_edge != 0.05 or len(set(names)) != count
            or any(not all(math.isfinite(x) for x in pair) or pair[0] >= pair[1]
                   for pair in limits)):
        raise ValueError("config values")
    return dict(kind=kind, count=count, digests=digests, limits=limits,
                table=table, half=half, cube_edge=cube_edge, base=base,
                tool=tool, names=names, robot_path=robot_path, scene_path=scene_path)


def parse_plan(body, config):
    r = Reader(body)
    count = r.take("I")
    if count != config["count"]:
        raise ValueError("joint count")
    q, goal_q = r.take("d" * count), r.take("d" * count)
    target, cube = r.take("7d"), r.take("7d")
    phase = r.take("I")
    tool_from_cube = r.take("7d")
    r.done()

    def inside(joints):
        return all(math.isfinite(x) and lo <= x <= hi
                   for x, (lo, hi) in zip(joints, config["limits"]))

    def packed(joints):
        return struct.pack("<%dd" % count, *joints)

    if (phase > HOME or not all(map(pose_valid, (target, cube, tool_from_cube)))
            or not inside(q) or not inside(goal_q)
            or (phase != HOME and packed(goal_q) != packed(q))):
        raise ValueError("plan values")
    return dict(q=q, goal_q=goal_q, target=target, cube=cube, phase=phase,
                tool_from_cube=tool_from_cube)


def pack_plan(dt, waypoints, config):
    if not math.isfinite(dt) or dt <= 0 or not 2 <= len(waypoints) <= MAX_WAYPOINTS:
        raise ValueError("trajectory shape")
    n = config["count"]
    result = bytearray(struct.pack("<dI", dt, len(waypoints)))
    previous = -1.0
    for t, q, dq in waypoints:
        if (not math.isfinite(t) or t < 0 or t <= previous or len(q) != n or len(dq) != n
                or any(not math.isfinite(x) or x < lo or x > hi
                       for x, (lo, hi) in zip(q, config["limits"]))
                or any(not math.isfinite(x) or abs(x) > limit
                       for x, limit in zip(dq, VELOCITY_LIMITS[n], strict=True))):
            raise ValueError("trajectory values")
        previous = t
        result += struct.pack("<" + "d" * (1 + 2 * n), t, *q, *dq)
    return bytes(result)


def single_plan(rows):
    """cuRobo 0.8.0 plans are [batch][goal set][time][joint]; one problem is posed."""
    if len(rows) != 1 or len(rows[0]) != 1:
        raise ValueError("trajectory batch")
    return rows[0][0]


def verify_file(path, digest):
    file = Path(path)
    if (not file.is_file() or file.stat().st_size > 16 * 1024 * 1024
            or hashlib.sha256(file.read_bytes()).digest() != digest):
        raise ValueError("file digest")


def verify_pinned_model(config):
    digest, base, tool, names, limits = PINNED_MODELS[config["kind"]]
    if (config["digests"][0] != bytes.fromhex(digest) or
            config["base"] != base or config["tool"] != tool or
            tuple(config["names"]) != names or
            tuple(map(tuple, config["limits"])) != limits):
        raise ValueError("pinned model mapping or limits")


def cover_sha256(spheres):
    """Same canonical digest as check_panda_spheres.cover_sha256."""
    rows = {link: [[*map(float, s["center"]), float(s["radius"])] for s in spheres[link]]
            for link in sorted(spheres)}
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def rotate(pose, vector):
    return compose((0.0, 0.0, 0.0, *pose[3:]), (*vector, 0.0, 0.0, 0.0, 1.0))[:3]


def contact_face(cube, table):
    """(axis, sign) of the cube face toward the table top: the face whose outward
    normal points most nearly down the table normal."""
    up = rotate(table, (0.0, 0.0, 1.0))
    dots = [sum(a * b for a, b in zip(rotate(cube, axis), up))
            for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))]
    axis = max(range(3), key=lambda i: abs(dots[i]))
    return axis, -1.0 if dots[axis] > 0 else 1.0


def cube_cover(edge, contact=None):
    """Cube-frame spheres (x, y, z, r) covering the cube surface: the circumsphere of
    every outer cell of a k^3 split, which bulges (sqrt(3) - 1) * edge / (2k) past each
    face. With `contact` (axis, sign), the layer on that face is pulled inward so its
    outermost point is CONTACT_CLEARANCE_M inside the face."""
    cell = edge / HELD_COVER_K
    radius = math.sqrt(3) * cell / 2 + 1e-6
    outer = edge / 2 - cell / 2
    grid = [-outer + i * cell for i in range(HELD_COVER_K)]
    spheres = []
    for x in grid:
        for y in grid:
            for z in grid:
                center = [x, y, z]
                if all(abs(abs(v) - outer) > 1e-12 for v in center):
                    continue
                if contact and abs(center[contact[0]] * contact[1] - outer) <= 1e-12:
                    center[contact[0]] = contact[1] * (edge / 2 - CONTACT_CLEARANCE_M - radius)
                spheres.append((*center, radius))
    if len(spheres) != HELD_COVER_SPHERES:
        raise ValueError("held cube cover")
    return spheres


class RealPlanner:
    def __init__(self, config):
        verify_pinned_model(config)
        import torch
        import yaml
        import curobo
        from curobo import runtime
        from curobo.content import get_assets_path
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo._src.geom.types import SceneCfg
        from curobo._src.state.state_joint import JointState
        from curobo._src.types.tool_pose import GoalToolPose
        from curobo._src.types.pose import Pose
        if curobo.__version__ != "0.8.0":
            raise ValueError("cuRobo version")
        verify_file(config["robot_path"], config["digests"][1])
        verify_file(config["scene_path"], config["digests"][2])
        robot_yaml = yaml.safe_load(Path(config["robot_path"]).read_text())
        kin = robot_yaml["robot_cfg"]["kinematics"]
        if (kin["base_link"] != config["base"] or
                kin["tool_frames"] != [config["tool"]] or
                kin["cspace"]["joint_names"][:config["count"]] != config["names"] or
                set(kin["collision_link_names"]) !=
                    (set(kin["collision_spheres"]) |
                     set(kin.get("extra_collision_spheres", {}))) or
                kin.get("extra_collision_spheres", {}).get("attached_object", 0)
                    != HELD_COVER_SPHERES or
                kin.get("extra_links", {}).get("attached_object", {}).get("parent_link_name")
                    != config["tool"] or
                kin["extra_links"]["attached_object"].get("fixed_transform")
                    != [0, 0, 0, 1, 0, 0, 0]):
            raise ValueError("robot frames")
        if kin["cspace"].get("position_limit_clip") != POSITION_LIMIT_CLIP[config["kind"]]:
            raise ValueError("position clip")
        if (config["kind"] == 7 and
                cover_sha256(kin["collision_spheres"]) != PANDA_COVER_SHA256):
            raise ValueError("Panda collision spheres do not cover the accepted model")
        urdf = Path(kin["urdf_path"])
        if not urdf.is_absolute():
            urdf = get_assets_path() / urdf
        verify_file(urdf, config["digests"][0])
        scene_yaml = yaml.safe_load(Path(config["scene_path"]).read_text())
        cuboids = scene_yaml.get("cuboid", {})
        if set(scene_yaml) != {"cuboid"} or set(cuboids) != {"table", "cube"}:
            raise ValueError("scene recipe")
        table, cube = cuboids["table"], cuboids["cube"]
        if (len(table["dims"]) != 3 or len(table["pose"]) != 7 or
                len(cube["dims"]) != 3 or
                any(abs(a - b) > 1e-9 for a, b in zip(table["dims"],
                                                   (2*x for x in config["half"])))
                or any(abs(a - b) > 1e-9 for a, b in zip(table["pose"],
                                                         wxyz(config["table"])))
                or cube["dims"] != [0.05, 0.05, 0.05]):
            raise ValueError("scene geometry")
        self.config = config
        self.torch, self.SceneCfg = torch, SceneCfg
        self.JointState, self.GoalToolPose, self.Pose = JointState, GoalToolPose, Pose
        self.planner = MotionPlanner(MotionPlannerCfg.create(
            robot=config["robot_path"], scene_model=scene_yaml,
            collision_cache={"obb": 2}, random_seed=123))
        if (self.planner.joint_names != config["names"] or
                self.planner.tool_frames != [config["tool"]]):
            raise ValueError("active joint mapping")
        # Home plans a joint goal, which changes the trajectory optimizer's goal buffer
        # layout; cuRobo 0.8.0 then recaptures its CUDA graphs, which its runtime only
        # allows with cuda_graph_reset (CUDA 12 or newer).
        runtime.cuda_graph_reset = True
        # cuRobo 0.8.0 MotionPlanner.attachment_manager reads an attribute TrajOptSolver
        # lacks; the manager lives on the solver core. Attachments are only honored if
        # IK, trajectory optimization and the graph planner read one sphere buffer.
        planner = self.planner
        buffers = {k.config.kinematics_config.link_spheres.data_ptr() for k in (
            planner.ik_solver.core.kinematics, planner.trajopt_solver.core.kinematics,
            planner.graph_planner.kinematics)}
        if len(buffers) != 1:
            raise ValueError("attachment sphere buffer is not shared")
        self.attachments = planner.trajopt_solver.core.attachment_manager
        self.planner.warmup(enable_graph=True, num_warmup_iterations=5)
        self.was_attached = False

    def joints(self, values):
        return self.JointState.from_position(
            self.torch.tensor([values], device="cuda", dtype=self.torch.float32),
            joint_names=self.config["names"])

    def plan(self, request):
        c, p, torch = self.config, self.planner, self.torch
        phase = request["phase"]
        q = self.joints(request["q"])
        # cuRobo's attach/detach pattern: every request rebuilds its phase's world.
        if self.was_attached:
            self.attachments.detach()
            self.was_attached = False
        cuboids = {"table": {"dims": [2*x for x in c["half"]],
                             "pose": wxyz(c["table"])}}
        # The cube is an obstacle unless the tool is closing on it, leaving it or holding it.
        if phase in (FREE, HOME):
            cuboids["cube"] = {"dims": [0.05]*3, "pose": wxyz(request["cube"])}
        p.update_world(self.SceneCfg.create({"cuboid": cuboids}))
        if phase in (HELD, HELD_CONTACT):
            tool_pose = p.compute_kinematics(q).tool_poses.get_link_pose(c["tool"])
            position = tool_pose.position.reshape(-1).cpu().tolist()
            quaternion = tool_pose.quaternion.reshape(-1).cpu().tolist()
            predicted = compose([*position, *quaternion[1:], quaternion[0]],
                                request["tool_from_cube"])
            actual = request["cube"]
            if (max(abs(a-b) for a, b in zip(predicted[:3], actual[:3])) > 1e-3
                    or abs(sum(a*b for a, b in zip(predicted[3:], actual[3:]))) < 1-1e-3):
                raise ValueError("attachment transform disagrees with FK")
            # The cover is placed at the FK-checked grasp in the tool frame, which is the
            # identity-fixed attached_object frame, so no world offset is involved.
            contact = (contact_face(request["cube"], c["table"])
                       if phase == HELD_CONTACT else None)
            grasp = request["tool_from_cube"]
            spheres = [(*compose(grasp, (x, y, z, 0.0, 0.0, 0.0, 1.0))[:3], r)
                       for x, y, z, r in cube_cover(c["cube_edge"], contact)]
            self.attachments.update(
                torch.tensor(spheres, device="cuda", dtype=torch.float32), q,
                link_name="attached_object")
            self.was_attached = True
        if phase == HOME:
            result = p.plan_cspace(self.joints(request["goal_q"]), q)
        else:
            result = self.plan_pose(request["target"], q)
        if result is None or not result.success.any():
            raise RuntimeError("planner failure")
        trajectory = result.get_interpolated_plan()
        dt = float(p.trajopt_solver.config.interpolation_dt)
        # Locked joints (the Panda fingers) trail the active joints in cuRobo's plan.
        n = c["count"]
        if list(trajectory.joint_names)[:n] != c["names"]:
            raise ValueError("trajectory joint mapping")
        positions = [row[:n] for row in single_plan(trajectory.position.cpu().tolist())]
        velocities = [row[:n] for row in single_plan(trajectory.velocity.cpu().tolist())]
        return dt, [(i*dt, pos, vel) for i, (pos, vel) in enumerate(zip(positions, velocities))]

    def plan_pose(self, target, q):
        c, torch = self.config, self.torch
        goal = self.GoalToolPose(
            tool_frames=[c["tool"]],
            position=torch.tensor([[[[[*target[:3]]]]]], device="cuda", dtype=torch.float32),
            quaternion=torch.tensor([[[[[target[6], *target[3:6]]]]]],
                                    device="cuda", dtype=torch.float32))
        return self.planner.plan_pose(goal, q)


def serve(factory=RealPlanner):
    started = False
    config = planner = None
    last_sequence = 0
    while True:
        payload = frame()
        if payload is None:
            return
        magic, version, op, role, sequence, status = HEADER.unpack_from(payload)
        if (magic != b"INVP" or version != VERSION or role != 0 or status != 0
                or sequence <= last_sequence):
            raise ValueError("request header")
        last_sequence = sequence
        body = payload[HEADER.size:]
        try:
            if op == 1:
                if started or body:
                    raise ValueError("init order")
                started = True
                send(op, sequence)
            elif op == 2:
                if not started or config is not None:
                    send(op, sequence, 2)
                    continue
                parsed = parse_config(body)
                try:
                    instance = factory(parsed)
                except Exception:
                    send(op, sequence, 3)
                    continue
                config, planner = parsed, instance
                send(op, sequence)
            elif op == 3:
                if planner is None:
                    send(op, sequence, 2)
                    continue
                request = parse_plan(body, config)
                try:
                    dt, waypoints = planner.plan(request)
                    reply = pack_plan(dt, waypoints, config)
                except Exception:
                    send(op, sequence, 3)
                    continue
                send(op, sequence, body=reply)
            elif op == 4:
                if not started or body:
                    raise ValueError("shutdown order")
                send(op, sequence)
                return
            else:
                raise ValueError("operation")
        except (ValueError, UnicodeError, struct.error):
            send(op, sequence, 1)


if __name__ == "__main__":
    serve()
    # Every reply is already written to the socket. Interpreter finalization with
    # torch/CUDA loaded takes 0.7-0.8 s on the qualification host, a large share of the
    # bridge's 2 s exit allowance; exiting directly leaves only the driver's teardown.
    os._exit(0)
