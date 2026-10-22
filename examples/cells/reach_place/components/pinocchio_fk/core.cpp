#include "core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Geometry>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/mjcf.hpp>

namespace invar_fixture {
namespace {

pinocchio::Model load_model(const std::string &path) {
  if (path.empty()) throw std::invalid_argument("MJCF path is empty");
  pinocchio::Model model;
  pinocchio::mjcf::buildModel(path, model, false);
  return model;
}

pinocchio::FrameIndex find_frame(const pinocchio::Model &model,
                                 const std::string &name) {
  for (pinocchio::FrameIndex i = 0; i < model.nframes; ++i) {
    if (model.frames[i].name == name) return i;
  }
  throw std::invalid_argument("MJCF frame not found: " + name);
}

pinocchio::JointIndex find_joint(const pinocchio::Model &model,
                                 const std::string &name) {
  for (pinocchio::JointIndex i = 1; i < model.njoints; ++i) {
    if (model.names[i] == name) return i;
  }
  throw std::invalid_argument("MJCF joint not found: " + name);
}

} // namespace

PinocchioFk::PinocchioFk(const std::string &model_xml,
                         const std::string &base_frame,
                         const std::string &tool_frame,
                         const std::vector<std::string> &arm_joint_names)
    : model_(load_model(model_xml)), data_(model_),
      base_frame_id_(find_frame(model_, base_frame)),
      tool_frame_id_(find_frame(model_, tool_frame)) {
  if (arm_joint_names.empty()) throw std::invalid_argument("joint list is empty");
  for (const auto &name : arm_joint_names) {
    const auto id = find_joint(model_, name);
    if (model_.nqs[id] != 1 ||
        std::find(joint_indices_.begin(), joint_indices_.end(), id) !=
            joint_indices_.end())
      throw std::invalid_argument("joint is repeated or non-scalar: " + name);
    joint_indices_.push_back(id);
  }
}

FkPose PinocchioFk::step(const std::vector<double> &arm_q_rad) {
  if (arm_q_rad.size() != joint_indices_.size())
    throw JointInputError("wrong joint count");

  for (std::size_t i = 0; i < arm_q_rad.size(); ++i) {
    const auto value = arm_q_rad[i];
    const auto index = model_.idx_qs[joint_indices_[i]];
    if (!std::isfinite(value) || value < model_.lowerPositionLimit[index] ||
        value > model_.upperPositionLimit[index])
      throw JointInputError("non-finite or out-of-limit joint");
  }

  Eigen::VectorXd q = pinocchio::neutral(model_);
  for (std::size_t i = 0; i < arm_q_rad.size(); ++i) {
    const auto value = arm_q_rad[i];
    const auto index = model_.idx_qs[joint_indices_[i]];
    q[index] = value;
  }

  pinocchio::forwardKinematics(model_, data_, q);
  pinocchio::updateFramePlacements(model_, data_);
  const auto tool_in_base =
      data_.oMf[base_frame_id_].inverse() * data_.oMf[tool_frame_id_];
  Eigen::Quaterniond rotation(tool_in_base.rotation());
  FkPose pose{{tool_in_base.translation().x(), tool_in_base.translation().y(),
               tool_in_base.translation().z()},
              {rotation.w(), rotation.x(), rotation.y(), rotation.z()}};
  const auto first_nonzero = std::find_if(
      pose.quaternion_wxyz.begin(), pose.quaternion_wxyz.end(),
      [](double value) { return std::abs(value) > 1e-15; });
  if (first_nonzero != pose.quaternion_wxyz.end() && *first_nonzero < 0.0) {
    for (auto &value : pose.quaternion_wxyz) value = -value;
  }
  if (!std::all_of(pose.translation_m.begin(), pose.translation_m.end(),
                   [](double value) { return std::isfinite(value); }) ||
      !std::all_of(pose.quaternion_wxyz.begin(), pose.quaternion_wxyz.end(),
                   [](double value) { return std::isfinite(value); }))
    throw std::runtime_error("Pinocchio returned non-finite pose");
  return pose;
}

} // namespace invar_fixture
