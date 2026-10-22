"""Offline S1-06 acceptance measurements from original MCAPs, never a task replay.

Usage: task_recording_metrics.py REPO WORKSPACE ARM RUN_DIR [RUN_DIR ...]
Uses the public Pinocchio binding independently of the generated FK shim.
"""

import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from fk_mujoco_crosscheck import digest


MOVES = ("pregrasp", "descend", "lift", "transfer", "place_descend", "retreat", "home")


def transform(values):
    x, y, z, w = values[3:]
    return pin.SE3(pin.Quaternion(w, x, y, z).matrix(), np.asarray(values[:3]))


def pose(value):
    q = pin.Quaternion(value.rotation)
    return [*value.translation, q.x, q.y, q.z, q.w]


def error(actual, target):
    a, b = np.asarray(actual), np.asarray(target)
    same, opposite = np.linalg.norm(a[3:] - b[3:]), np.linalg.norm(a[3:] + b[3:])
    return [float(np.linalg.norm(a[:3] - b[:3])),
            float(4 * math.atan2(min(same, opposite), max(same, opposite)))]


def finite(values, count):
    if len(values) != count or not all(math.isfinite(x) for x in values):
        raise ValueError("invalid recorded numeric vector")
    return list(values)


def measure_failure(run):
    files = list((run / "recording").glob("*.mcap"))
    if len(files) != 1 or not 0 < files[0].stat().st_size <= 128 * 1024 * 1024:
        raise ValueError("expected one finalized failure MCAP")
    if (run / "launch-exit.txt").read_text().strip() != "launch_exit=1":
        raise ValueError("killed planner must fail launch")
    log = (run / "launch.log").read_text()
    if '"code":"S1406"' not in log or not (run / "planner-killed.txt").is_file():
        raise ValueError("missing typed killed-planner failure")
    return {"run": run.name, "code": "S1406", "launch_exit": 1,
            "mcap_sha256": digest(files[0]), "mcap_bytes": files[0].stat().st_size,
            "launch_log_sha256": digest(run / "launch.log")}


def measure(run, binding, mjmodel, pinmodel, spec):
    files = list((run / "recording").glob("*.mcap"))
    if len(files) != 1 or not 0 < files[0].stat().st_size <= 128 * 1024 * 1024:
        raise ValueError("expected one bounded MCAP")
    if (run / "launch-exit.txt").read_text().strip() != "launch_exit=0":
        raise ValueError("acceptance run must have clean launch exit")
    events = []
    for line in (run / "launch.log").read_text().splitlines():
        start = line.find('{"component"')
        if start >= 0 and '"phase"' in line:
            events.append(json.loads(line[start:]))
    completion = next(e["simulation_ns"] for e in events if e["phase"] == "complete")
    exits = []
    for name in MOVES:
        index = next(i for i, e in enumerate(events) if e["phase"].startswith(name + "/executing"))
        exits.append(next(e["simulation_ns"] for e in events[index + 1:]
                          if not e["phase"].startswith(name + "/")))

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(run / "recording"), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    required = ("/invar/cell_snapshot", "/sequencer/request", "/sequencer/phase", "/sequencer/approved")
    kinds = {t: get_message(topics[t]) for t in required}
    counts = dict.fromkeys(required, 0)
    snapshots, requests, phases, plans, request_records = {}, [], [], [], []
    last, total, previous, first = None, 0, None, None
    excess, forbidden = -math.inf, 0
    source = bytes.fromhex(binding["source_ir_sha256"])
    order = bytes.fromhex(binding["joint_order_sha256"])
    robot_order = bytes.fromhex(binding["robot_joint_order_sha256"])
    while reader.has_next():
        topic, raw = reader.read_next_ext()[:2]
        if topic not in kinds:
            continue
        counts[topic] += 1
        if sum(counts.values()) > 250_000:
            raise ValueError("record limit exceeded")
        m = deserialize_message(raw, kinds[topic])
        if topic == "/sequencer/request":
            requests.append(finite(m.target_base_from_tool, 7))
            request_records.append({"sha256": hashlib.sha256(raw).hexdigest(),
                                    "start_q_rad": finite(m.start_q_rad, binding["joint_count"])})
        elif topic == "/sequencer/phase":
            phases.append(m.data)
        elif topic == "/sequencer/approved":
            # Exact trajectory bytes include times, positions and velocities.
            plans.append(hashlib.sha256(raw).hexdigest())
        else:
            total += 1
            if (bytes(m.source_ir_sha256) != source or bytes(m.joint_order_sha256) != order
                    or bytes(m.robot_joint_order_sha256) != robot_order):
                raise ValueError("snapshot source/joint identity mismatch")
            current = (m.simulation_ns, m.step_index)
            if previous is not None and (current[0] <= previous[0] or current[1] != previous[1] + 1):
                raise ValueError("noncontiguous snapshot sequence")
            previous = current
            if first is None:
                first = {"simulation_ns": m.simulation_ns, "q_rad": list(m.q_rad), "dq_rad_s": list(m.dq_rad_s)}
            finite(m.q_rad, binding["joint_count"])
            finite(m.dq_rad_s, binding["joint_count"])
            finite(m.world_from_tool, 7)
            finite(m.world_from_cube, 7)
            finite(m.robot_q_si, len(binding["robot_joints"]))
            finite(m.robot_dq_si_per_s, len(binding["robot_joints"]))
            for q, dq, joint in zip(m.robot_q_si, m.robot_dq_si_per_s, binding["robot_joints"], strict=True):
                lo, hi = joint["position_limits_si"]
                excess = max(excess, lo - q, q - hi, abs(dq) - joint["velocity_limit_si_per_s"])
            ids, classes = m.contact_geom_ids, m.geom_classes
            if len(ids) % 2 or len(ids) > 2048 or len(classes) > 4096:
                raise ValueError("invalid recorded contacts")
            for a, b in zip(ids[::2], ids[1::2], strict=True):
                if not (0 <= a < len(classes) and 0 <= b < len(classes)):
                    raise ValueError("contact index outside geometry classes")
                pair = sorted((classes[a], classes[b]))
                if 1 in pair and pair != [1, 3]:
                    forbidden += 1
            if m.simulation_ns in exits:
                snapshots[m.simulation_ns] = m
            if m.simulation_ns == completion:
                last = m
    if len(requests) != 7 or len(plans) != 7 or len(snapshots) != 7 or last is None:
        raise ValueError("missing exact waypoint/completion snapshots or trajectories")
    if phases != [e["phase"] for e in events] or len(phases) != 24:
        raise ValueError("phase topic/log disagreement")

    projection = binding["tool_projection"]
    offset = projection["reference_from_cell"]
    reference_from_cell = transform(offset["translation_m"] + offset["quaternion_xyzw"])
    base = projection["source_base_from_reference_base"]
    source_from_reference = transform(base["translation_m"] + base["quaternion_xyzw"])
    world_from_base = transform(binding["world_from_base"])
    pinbase, pintool = (pinmodel.getFrameId(spec[key]) for key in ("base_frame", "tool_frame"))
    if max(pinbase, pintool) >= pinmodel.nframes:
        raise ValueError("missing oracle frames")
    pindata, mjdata = pinmodel.createData(), mujoco.MjData(mjmodel)
    mjsite = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_SITE, "invar_tool")
    if mjsite < 0:
        raise ValueError("missing generated tool site")
    waypoints = []
    for name, ns, target in zip(MOVES, exits, requests, strict=True):
        snapshot = snapshots[ns]
        q = pin.neutral(pinmodel)
        for joint, value in zip(binding["joint_names"], snapshot.q_rad, strict=True):
            jid = pinmodel.getJointId(joint)
            if jid >= pinmodel.njoints or pinmodel.nqs[jid] != 1:
                raise ValueError("invalid oracle joint")
            q[pinmodel.idx_qs[jid]] = value
            mjdata.qpos[mjmodel.joint(joint).qposadr] = value
        pin.forwardKinematics(pinmodel, pindata, q)
        pin.updateFramePlacements(pinmodel, pindata)
        predicted = world_from_base * source_from_reference * pindata.oMf[pinbase].inverse() * pindata.oMf[pintool] * reference_from_cell
        mujoco.mj_forward(mjmodel, mjdata)
        site = pin.SE3(mjdata.site_xmat[mjsite].reshape(3, 3).copy(), mjdata.site_xpos[mjsite].copy())
        waypoints.append({"move": name, "simulation_ns": ns, "q_rad": list(snapshot.q_rad),
                          "pinocchio_world_from_tool": pose(predicted),
                          "recorded_world_from_tool": list(snapshot.world_from_tool),
                          "pinocchio_vs_recorded_m_rad": error(pose(predicted), snapshot.world_from_tool),
                          "pinocchio_vs_mujoco_m_rad": error(pose(predicted), pose(site)),
                          "tool_error_m_rad": error(pose(predicted), pose(world_from_base * transform(target)))})
    final_fields = {k: list(getattr(last, k)) for k in
                    ("q_rad", "dq_rad_s", "robot_q_si", "robot_dq_si_per_s", "world_from_tool", "world_from_cube")}
    final_fields["attached"] = [float(last.attached)]
    encoded = b"".join(struct.pack("<d", x) for values in final_fields.values() for x in values)
    return {"run": run.name, "launch_exit": 0, "seed": 123, "mcap_sha256": digest(files[0]),
            "mcap_bytes": files[0].stat().st_size, "launch_log_sha256": digest(run / "launch.log"), "complete_sim_ns": completion,
            "snapshots": total, "topic_counts": counts, "waypoints": waypoints,
            "max_joint_or_velocity_limit_excess": excess, "forbidden_robot_contacts": forbidden,
            "final_cube_error_m_rad": error(last.world_from_cube, pose(world_from_base * transform(binding["final_cube_base"]))),
            "home_error_rad": max(abs(a - b) for a, b in zip(last.q_rad, binding["home_q_rad"], strict=True)),
            "final_attached": bool(last.attached), "final_recorded_state": final_fields,
            "final_recorded_state_bits": list(struct.unpack(f"<{len(encoded) // 8}Q", encoded)),
            "final_recorded_state_sha256": hashlib.sha256(encoded).hexdigest(),
            "first_recorded_snapshot": first,
            "planner_inputs": request_records,
            "approved_trajectory_sha256": plans}


def main(repo, workspace, arm, runs, failed_run=None):
    config = workspace / "install/invar_cell_bringup/share/invar_cell_bringup/config"
    binding_path = config / "sequencer-binding-v1.json"
    binding = json.loads(binding_path.read_bytes())
    spec = next(a for a in json.loads((repo / "tests/fixtures/foreign-components/pinocchio_goldens.json").read_bytes())["arms"] if a["arm"] == arm)
    model_path = repo / "examples/cells/reach_place/assets/menagerie" / spec["model_file"]
    if digest(model_path) != binding["tool_projection"]["reference_model_sha256"]:
        raise ValueError("model identity differs from binding")
    scene = config / "mjcf/scene.xml"
    mjmodel = mujoco.MjModel.from_xml_path(str(scene))
    pinmodel = pin.buildModelFromMJCF(str(model_path))
    results = [measure(r, binding, mjmodel, pinmodel, spec) for r in runs]
    build_log = workspace.parent / f"{arm}-build.log"
    summaries = [line for line in build_log.read_text().splitlines() if line.startswith("Summary:")]
    if len(summaries) != 1 or "packages finished" not in summaries[0] or "failed" in summaries[0].lower():
        raise ValueError("missing clean build summary")
    packages = int(summaries[0].split()[1])
    equal = len({r["final_recorded_state_sha256"] for r in results}) == 1
    print(json.dumps({"schema": "invar-s106-run-evidence-v1", "arm": arm,
                      "collector_sha256": digest(Path(__file__)),
                      "source_ir_sha256": binding["source_ir_sha256"],
                      "build_packages_finished": packages, "build_log_sha256": digest(build_log),
                      "binding_sha256": digest(binding_path), "scene_sha256": digest(scene),
                      "model_sha256": digest(model_path), "runs": results,
                      "runtime_failures": [measure_failure(failed_run)] if failed_run else [],
                      "versions": {"pinocchio": pin.__version__, "mujoco": mujoco.__version__, "numpy": np.__version__},
                      "determinism": {"class": "recorded-subset-bit-exact" if equal else "not-bit-exact",
                                      "full_mujoco_state_recorded": False,
                                      "scope": "completion snapshot q/dq, all robot joints, tool/cube poses, attachment; excludes simulation time"}},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    if len(sys.argv) < 7:
        raise SystemExit("usage: task_recording_metrics.py REPO WORKSPACE ARM RUN RUN RUN")
    failed = None
    if len(sys.argv) >= 9 and sys.argv[-2] == "--killed-planner":
        failed = Path(sys.argv[-1])
        del sys.argv[-2:]
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], [Path(p) for p in sys.argv[4:]], failed)
