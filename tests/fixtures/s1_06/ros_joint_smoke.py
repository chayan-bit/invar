import math
import time

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_msgs.msg import Int32MultiArray
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

rclpy.init()
node = rclpy.create_node("invar_s106_joint_smoke")
latest = [None]
node.create_subscription(JointState, "/joint_states", lambda msg: latest.__setitem__(0, msg), 10)
publisher = node.create_publisher(
    JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10
)
attachment = [None]
node.create_subscription(Bool, "/invar/attachment_active", lambda msg: attachment.__setitem__(0, msg.data), 10)
tool_pose = [None]
fk_pose = [None]
object_pose = [None]
contacts = [None]
node.create_subscription(PoseStamped, "/invar/tool_pose", lambda msg: tool_pose.__setitem__(0, msg), 10)
node.create_subscription(Pose, "/fk/pose", lambda msg: fk_pose.__setitem__(0, msg), 10)
node.create_subscription(PoseStamped, "/invar/object_pose", lambda msg: object_pose.__setitem__(0, msg), 10)
node.create_subscription(Int32MultiArray, "/invar/contact_geom_ids", lambda msg: contacts.__setitem__(0, msg), 10)
client = node.create_client(SetBool, "/invar/set_attachment")
try:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and (
        latest[0] is None or publisher.get_subscription_count() == 0
    ):
        rclpy.spin_once(node, timeout_sec=0.1)
    assert latest[0] is not None, "joint states unavailable"
    assert publisher.get_subscription_count() > 0, "trajectory subscriber unavailable"
    initial = dict(zip(latest[0].name, latest[0].position))
    assert all(name in initial for name in JOINTS), "arm joints missing"
    target = initial[JOINTS[0]] + 0.08
    assert -6.28319 <= target <= 6.28319, "target exceeds declared joint range"

    trajectory = JointTrajectory()
    trajectory.joint_names = JOINTS
    point = JointTrajectoryPoint()
    point.positions = [target if name == JOINTS[0] else initial[name] for name in JOINTS]
    point.time_from_start.sec = 3
    trajectory.points = [point]
    publisher.publish(trajectory)

    deadline = time.monotonic() + 9
    moved = False
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if latest[0] is not None:
            current = dict(zip(latest[0].name, latest[0].position))
            moved = abs(current[JOINTS[0]] - initial[JOINTS[0]]) > 0.04
            if moved:
                break
    assert moved, "shoulder joint did not follow trajectory"
    print("joint_motion_pass")

    def wait_for(predicate, message):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if predicate():
                return
        raise AssertionError(message)

    wait_for(lambda: client.service_is_ready() and attachment[0] is False, "attachment plugin unavailable")
    wait_for(lambda: tool_pose[0] is not None and object_pose[0] is not None and contacts[0] is not None, "MuJoCo telemetry unavailable")
    wait_for(lambda: fk_pose[0] is not None, "Pinocchio FK pose unavailable")
    assert all(math.isfinite(value) for value in [fk_pose[0].position.x, fk_pose[0].position.y, fk_pose[0].position.z])
    assert abs(sum(value * value for value in [fk_pose[0].orientation.w, fk_pose[0].orientation.x, fk_pose[0].orientation.y, fk_pose[0].orientation.z]) - 1) < 0.000001
    print("fk_pose_pass")
    settled_until = time.monotonic() + 2
    while time.monotonic() < settled_until:
        rclpy.spin_once(node, timeout_sec=0.1)
    offset = math.dist(
        [fk_pose[0].position.x, fk_pose[0].position.y, fk_pose[0].position.z],
        [tool_pose[0].pose.position.x, tool_pose[0].pose.position.y, tool_pose[0].pose.position.z],
    )
    print(f"fk_pose_xyz=({fk_pose[0].position.x:.9g},{fk_pose[0].position.y:.9g},{fk_pose[0].position.z:.9g})")
    print(f"mujoco_tool_xyz=({tool_pose[0].pose.position.x:.9g},{tool_pose[0].pose.position.y:.9g},{tool_pose[0].pose.position.z:.9g})")
    print("arm_joint_positions=" + ",".join(f"{dict(zip(latest[0].name, latest[0].position))[name]:.9g}" for name in JOINTS))
    print(f"fk_tool_translation_error_m={offset:.9g}")
    assert 0.01 < object_pose[0].pose.position.z < 0.04, "cube telemetry is outside the cell"
    assert all(math.isfinite(value) for value in [tool_pose[0].pose.position.x, tool_pose[0].pose.position.y, tool_pose[0].pose.position.z])
    assert abs(sum(value * value for value in [tool_pose[0].pose.orientation.w, tool_pose[0].pose.orientation.x, tool_pose[0].pose.orientation.y, tool_pose[0].pose.orientation.z]) - 1) < 0.000001
    assert len(contacts[0].data) % 2 == 0, "contact geom IDs are not paired"
    for requested in [True, False]:
        future = client.call_async(SetBool.Request(data=requested))
        wait_for(lambda: future.done(), "attachment service timed out")
        assert future.result().success, future.result().message
        wait_for(lambda: attachment[0] is requested, "MuJoCo weld state did not change")
    print("attachment_toggle_pass")
finally:
    node.destroy_node()
    rclpy.shutdown()
