// CPU-only stand-ins exercise generated conversion logic; this is not a ROS build.
#include "planner@N@_conversions.hpp"
#include <cassert>
#include <limits>
#include <vector>
namespace binding = invar_planner@N@_ros_v1;
struct RequestMessage {
  std::vector<double> start_q_rad = std::vector<double>(@N@, 0.0);
  std::vector<double> goal_q_rad = std::vector<double>(@N@, 0.0);
  std::array<double, 7> target_base_from_tool{0, 0, 0, 0, 0, 0, 1};
  std::array<double, 7> cube_base_pose{0, 0, 0, 0, 0, 0, 1};
  std::array<double, 7> tool_from_cube{0, 0, 0, 0, 0, 0, 1};
  unsigned phase = 0;
};
struct PointMessage {
  double time_from_start_s;
  std::array<double, @N@> q_rad{}, dq_rad_s{};
};
struct TrajectoryMessage {
  double dt_s = 123.0;
  std::vector<PointMessage> waypoints;
};
struct ControllerPoint {
  std::vector<double> positions, velocities;
  struct Duration { std::int32_t sec = 0; std::uint32_t nanosec = 0; } time_from_start;
};
struct ControllerMessage {
  std::vector<std::string> joint_names;
  std::vector<ControllerPoint> points;
};
int main() {
  binding::Limits limits;
  for (auto & limit : limits) limit = {-2.0, 2.0};
  binding::Speeds speeds;
  speeds.fill(2.0);
  RequestMessage request;
  inv_planner_request@N@_f64_v2 wire{};
  assert(binding::request_from_ros(request, limits, wire));
  assert(wire.target_base_from_tool.quaternion_xyzw[3] == 1.0);
  // Home (4) carries a joint goal; it must still lie inside the limits.
  auto home = request;
  home.phase = 4;
  home.goal_q_rad[0] = 1.5;
  assert(binding::request_from_ros(home, limits, wire));
  assert(wire.phase == 4 && wire.goal_q_rad[0] == 1.5);
  for (unsigned fault = 0; fault < 9; ++fault) {
    auto bad = request;
    if (fault == 0) bad.start_q_rad.pop_back();
    if (fault == 1) bad.phase = 5;
    if (fault == 2) bad.cube_base_pose[6] = 2.0;
    if (fault == 3) bad.start_q_rad[0] = std::numeric_limits<double>::quiet_NaN();
    if (fault == 4) bad.goal_q_rad.pop_back();
    if (fault == 5) bad.goal_q_rad[0] = 1.5; // A joint goal outside the home phase.
    if (fault == 6) bad.goal_q_rad[0] = -0.0; // Not bitwise the start.
    if (fault == 7) { bad.phase = 4; bad.goal_q_rad[0] = 2.5; }
    if (fault == 8) { bad.phase = 4; bad.goal_q_rad[0] = std::numeric_limits<double>::quiet_NaN(); }
    wire.start_q_rad[0] = 1.234;
    assert(!binding::request_from_ros(bad, limits, wire));
    assert(wire.start_q_rad[0] == 1.234);
  }
  inv_planner_trajectory@N@_f64_v1 trajectory{};
  trajectory.dt_s = 0.1;
  trajectory.count = 256;
  for (unsigned i = 0; i < 256; ++i) trajectory.waypoints[i].time_from_start_s = i * 0.1;
  TrajectoryMessage output;
  assert(binding::trajectory_to_ros(trajectory, limits, speeds, output));
  assert(output.waypoints.size() == 256);
  assert(output.waypoints[255].time_from_start_s == 25.5);
  for (unsigned fault = 0; fault < 6; ++fault) {
    auto bad = trajectory;
    if (fault == 0) bad.count = 257;
    if (fault == 1) bad.count = 1;
    if (fault == 2) bad.dt_s = 0.0;
    if (fault == 3) bad.waypoints[1].time_from_start_s = 0.0;
    if (fault == 4) bad.waypoints[255].q_rad[0] = 3.0;
    if (fault == 5) bad.waypoints[255].dq_rad_s[0] = 3.0;
    output.dt_s = 123.0;
    assert(!binding::trajectory_to_ros(bad, limits, speeds, output));
    assert(output.dt_s == 123.0);
    assert(output.waypoints.size() == 256);
  }
  assert(binding::trajectory_to_ros(trajectory, limits, speeds, output));
  std::vector<std::string> names;
  for (unsigned i = 0; i < @N@; ++i) names.push_back("joint" + std::to_string(i));
  ControllerMessage controller;
  assert(binding::controller_trajectory_from_ros(output, names, limits, speeds, controller));
  assert(controller.joint_names == names && controller.points.size() == 256);
  assert(controller.points[255].positions.size() == @N@);
  assert(controller.points[255].time_from_start.sec == 25);
  assert(controller.points[255].time_from_start.nanosec == 500000000);
  for (unsigned fault = 0; fault < 6; ++fault) {
    auto bad = output;
    auto bad_names = names;
    if (fault == 0) bad.waypoints[1].time_from_start_s = 1e-12; // Collapses to zero nanoseconds.
    if (fault == 1) bad.waypoints[255].time_from_start_s = 2147483648.0; // ROS duration overflow.
    if (fault == 2) bad.waypoints[255].dq_rad_s[0] = 3.0;
    if (fault == 3) bad.waypoints.resize(257);
    if (fault == 4) bad_names[1] = bad_names[0];
    if (fault == 5) bad_names.pop_back();
    controller.joint_names = {"unchanged"};
    assert(!binding::controller_trajectory_from_ros(bad, bad_names, limits, speeds, controller));
    assert(controller.joint_names == std::vector<std::string>{"unchanged"});
    assert(controller.points.size() == 256);
  }
}
