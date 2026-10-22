// Executes the generated bridge with CPU ROS stand-ins. No ROS/GPU qualification.
#define main generated_controller_bridge_main
#include "bridge.cpp"
#undef main
#include <cassert>
#include <limits>

Plan valid_plan(unsigned count) {
  Plan plan;
  plan.dt_s = 0.1;
  plan.waypoints.resize(count);
  for (unsigned i = 0; i < count; ++i) {
    auto & point = plan.waypoints[i];
    point.time_from_start_s = i * 0.1;
    point.q_rad.fill(0.0);
    point.dq_rad_s.fill(0.0);
    if constexpr (@N@ == 7) {
      // Panda joint4 excludes zero and joint6 permits this interior value.
      point.q_rad[3] = -1.0;
      point.q_rad[5] = 1.0;
    }
  }
  return plan;
}

void reset_capture() {
  rclcpp::receive = {};
  rclcpp::spin_hook = {};
  rclcpp::publications = rclcpp::shutdowns = 0;
  rclcpp::Publisher<Command>::last_message = {};
}

int main() {
  for (unsigned count : {2u, 256u}) {
    reset_capture();
    const auto plan = valid_plan(count);
    rclcpp::spin_hook = [&] { rclcpp::send(plan); };
    assert(generated_controller_bridge_main(0, nullptr) == 0);
    assert(rclcpp::publications == 1 && rclcpp::shutdowns == 1);
    const auto & command = rclcpp::Publisher<Command>::last_message;
    assert(command.joint_names.size() == @N@);
    assert(command.joint_names[0] == "@FIRST_JOINT@");
    assert(command.points.size() == count);
    for (unsigned k = 0; k < count; ++k) {
      assert(command.points[k].positions.size() == @N@);
      assert(command.points[k].velocities.size() == @N@);
      for (unsigned j = 0; j < @N@; ++j) {
        assert(command.points[k].positions[j] == plan.waypoints[k].q_rad[j]);
        assert(command.points[k].velocities[j] == 0.0);
      }
    }
    assert(command.points[0].time_from_start.sec == 0);
    assert(command.points[0].time_from_start.nanosec == 0);
    assert(command.points.back().time_from_start.sec == (count == 2 ? 0 : 25));
    assert(command.points.back().time_from_start.nanosec == (count == 2 ? 100000000 : 500000000));
  }
  for (unsigned fault = 0; fault < 4; ++fault) {
    reset_capture();
    auto plan = valid_plan(256);
    if (fault == 0) plan.waypoints.back().q_rad[0] = 1e6;
    if (fault == 1) plan.waypoints.back().dq_rad_s[0] = std::numeric_limits<double>::quiet_NaN();
    if (fault == 2) plan.waypoints[1].time_from_start_s = 1e-12;
    if (fault == 3) plan.waypoints.push_back(plan.waypoints.back());
    rclcpp::spin_hook = [&] { rclcpp::send(plan); };
    assert(generated_controller_bridge_main(0, nullptr) == 1);
    assert(rclcpp::publications == 0 && rclcpp::shutdowns == 1);
    assert(rclcpp::Publisher<Command>::last_message.points.empty());
  }
  reset_capture();
  auto node = std::make_shared<ControllerBridge>();
  const auto valid = valid_plan(256);
  rclcpp::send(valid);
  auto invalid = valid;
  invalid.waypoints.back().q_rad[0] = 1e6;
  for (const auto & plan : {invalid, valid}) {
    bool rejected = false;
    try { rclcpp::send(plan); }
    catch (const std::runtime_error &) { rejected = true; }
    assert(rejected && rclcpp::publications == 1);
  }
  assert(rclcpp::Publisher<Command>::last_message.points.size() == 256);
  rclcpp::shutdown();
  std::cout << "controller_callback_cases=7 joint_count=@N@\n";
}
