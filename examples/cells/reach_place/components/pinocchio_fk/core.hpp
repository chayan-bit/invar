#ifndef INVAR_REACH_PLACE_PINOCCHIO_FK_CORE_HPP
#define INVAR_REACH_PLACE_PINOCCHIO_FK_CORE_HPP

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

namespace invar_fixture {

// Private result for the vendor-library core. S1-11 owns the public C layout.
struct FkPose {
  std::array<double, 3> translation_m;
  std::array<double, 4> quaternion_wxyz;
};

struct JointInputError : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

class PinocchioFk {
public:
  PinocchioFk(const std::string &model_xml, const std::string &base_frame,
              const std::string &tool_frame,
              const std::vector<std::string> &arm_joint_names);

  FkPose step(const std::vector<double> &arm_q_rad);
  std::size_t joint_count() const { return joint_indices_.size(); }

private:
  pinocchio::Model model_;
  pinocchio::Data data_;
  pinocchio::FrameIndex base_frame_id_;
  pinocchio::FrameIndex tool_frame_id_;
  std::vector<pinocchio::JointIndex> joint_indices_;
};

} // namespace invar_fixture

#endif
