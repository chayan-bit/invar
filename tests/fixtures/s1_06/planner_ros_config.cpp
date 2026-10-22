// Uses generated boundary code with CPU values only; no ROS runtime or planner.
#include "planner_config.hpp"
#include <cassert>
#include <limits>
namespace binding = invar_planner@N@_ros_v1;
int main(int argc, char ** argv) {
  assert(argc == 2);
  binding::ConfigValues values;
  values.model_sha256 = std::string(64, '1');
  values.robot_config_sha256 = std::string(64, '2');
  values.scene_sha256 = std::string(64, '3');
  values.base_frame = "base";
  values.tool_frame = "tool";
  values.robot_config_path = argv[1];
  values.scene_config_path = argv[1];
  values.table_pose = {0, 0, 0, 0, 0, 0, 1};
  values.table_half_extents_m = {0.5, 0.5, 0.05};
  for (unsigned i = 0; i < @N@; ++i) {
    values.joint_names.push_back("joint" + std::to_string(i));
    values.joint_limits_rad.push_back(-2.0);
    values.joint_limits_rad.push_back(2.0);
    values.velocity_limits_rad_s.push_back(2.0);
  }
  inv_planner_config_wire_v1 wire{};
  binding::Limits limits{};
  binding::Speeds speeds{};
  assert(binding::configure_values(values, wire, limits, speeds));
  assert(wire.joint_count == @N@ && wire.model_sha256[0] == 0x11);
  assert(wire.robot_config_sha256[31] == 0x22 && wire.scene_sha256[31] == 0x33);
  assert(wire.base_frame.length == 4 && wire.base_frame.bytes == reinterpret_cast<const uint8_t *>(values.base_frame.data()));
  assert(limits[@N@ - 1][0] == -2.0 && speeds[@N@ - 1] == 2.0);
  for (unsigned fault = 0; fault < 9; ++fault) {
    auto bad = values;
    if (fault == 0) bad.model_sha256[0] = 'x';
    if (fault == 1) bad.joint_names[1] = bad.joint_names[0];
    if (fault == 2) bad.joint_limits_rad.pop_back();
    if (fault == 3) bad.joint_limits_rad[0] = 4.0;
    if (fault == 4) bad.velocity_limits_rad_s[0] = std::numeric_limits<double>::quiet_NaN();
    if (fault == 5) bad.table_pose[6] = 0.0;
    if (fault == 6) bad.robot_config_path = "relative.yml";
    if (fault == 7) bad.cube_edge_m = 0.1;
    if (fault == 8) bad.base_frame = std::string("base\0hidden", 11);
    wire.joint_count = 99;
    limits[0][0] = 123.0;
    speeds[0] = 123.0;
    assert(!binding::configure_values(bad, wire, limits, speeds));
    assert(wire.joint_count == 99 && limits[0][0] == 123.0 && speeds[0] == 123.0);
  }
}
