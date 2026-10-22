"""Independent RIHS01 golden check; not a substitute for pinned ROS build parity."""
import hashlib
import json
from pathlib import Path


def field(name, type_id, capacity=0, nested=""):
    return {"name": name, "type": {
        "type_id": type_id, "capacity": capacity,
        "string_capacity": 0, "nested_type_name": nested,
    }}


def description(name, fields):
    return {"type_name": f"invar_cell_interfaces/msg/{name}", "fields": fields}


def expected_hashes():
    result = {}
    for n in (6, 7):
        waypoint = description(f"PlannerWaypoint{n}V1", [
            field("time_from_start_s", 11), field("q_rad", 59, n),
            field("dq_rad_s", 59, n),
        ])
        request = description(f"PlannerRequest{n}V2", [
            field("start_q_rad", 59, n), field("goal_q_rad", 59, n),
            field("target_base_from_tool", 59, 7),
            field("cube_base_pose", 59, 7), field("tool_from_cube", 59, 7),
            field("phase", 7),
        ])
        trajectory = description(f"PlannerTrajectory{n}V1", [
            field("dt_s", 11), field("waypoints", 97, 256, waypoint["type_name"]),
        ])
        for current, references in ((request, []), (waypoint, []), (trajectory, [waypoint])):
            hashable = {"type_description": current,
                        "referenced_type_descriptions": references}
            digest = hashlib.sha256(json.dumps(hashable).encode("utf-8")).hexdigest()
            result[current["type_name"].split("/")[-1]] = "RIHS01_" + digest
    return result


if __name__ == "__main__":
    fixture = Path(__file__).with_name("planner-ros-hashes.json")
    actual = json.loads(fixture.read_text())
    expected = expected_hashes()
    if actual != expected:
        raise SystemExit("planner ROS golden hashes differ from independent reference")
    print(f"{len(expected)} planner ROS hash references verified; ROS build parity remains pending")
