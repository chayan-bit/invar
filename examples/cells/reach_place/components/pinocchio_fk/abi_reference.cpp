#include "abi_reference.h"
#include "core.hpp"

#include <cmath>
#include <filesystem>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(sizeof(inv_joint6_f64) == 48 && alignof(inv_joint6_f64) == 8 &&
              offsetof(inv_joint6_f64, values) == 0);
static_assert(sizeof(inv_joint7_f64) == 56 && alignof(inv_joint7_f64) == 8 &&
              offsetof(inv_joint7_f64, values) == 0);
static_assert(sizeof(inv_pose_f64) == 56 && alignof(inv_pose_f64) == 8 &&
              offsetof(inv_pose_f64, translation_m) == 0 &&
              offsetof(inv_pose_f64, quaternion_xyzw) == 24);
static_assert(sizeof(inv_fk_config_v1) == 16 && alignof(inv_fk_config_v1) == 8 &&
              offsetof(inv_fk_config_v1, model_xml_path) == 0 &&
              offsetof(inv_fk_config_v1, model_xml_path_len) == 8 &&
              offsetof(inv_fk_config_v1, model_kind) == 12);

struct inv_fk_context_v1 {
  std::unique_ptr<invar_fixture::PinocchioFk> fk;
  uint32_t model_kind = 0;
};

namespace {

bool valid_utf8_path(const uint8_t *bytes, uint32_t length) {
  if (!bytes || length == 0 || length > 4096) return false;
  for (uint32_t i = 0; i < length;) {
    const uint8_t first = bytes[i++];
    if (first == 0) return false;
    if (first < 0x80) continue;
    uint32_t codepoint = 0;
    uint32_t minimum = 0;
    uint32_t remaining = 0;
    if (first >= 0xc2 && first <= 0xdf) {
      codepoint = first & 0x1f;
      minimum = 0x80;
      remaining = 1;
    } else if (first >= 0xe0 && first <= 0xef) {
      codepoint = first & 0x0f;
      minimum = 0x800;
      remaining = 2;
    } else if (first >= 0xf0 && first <= 0xf4) {
      codepoint = first & 0x07;
      minimum = 0x10000;
      remaining = 3;
    } else {
      return false;
    }
    if (length - i < remaining) return false;
    for (uint32_t j = 0; j < remaining; ++j) {
      const uint8_t next = bytes[i++];
      if ((next & 0xc0) != 0x80) return false;
      codepoint = (codepoint << 6) | (next & 0x3f);
    }
    if (codepoint < minimum || (codepoint >= 0xd800 && codepoint <= 0xdfff) ||
        codepoint > 0x10ffff)
      return false;
  }
  return true;
}

} // namespace

extern "C" const char *invar_pinocchio_v1_version(void) {
  return "invar-pinocchio-reference-abi-1/pinocchio-4.1.0";
}

extern "C" int32_t invar_pinocchio_v1_init(inv_fk_context_v1 **out_context) {
  if (!out_context) return INV_FK_BAD_ARGUMENT;
  if (*out_context) return INV_FK_BAD_STATE;
  try {
    *out_context = new inv_fk_context_v1;
    return INV_FK_OK;
  } catch (...) {
    return INV_FK_LIBRARY_FAILURE;
  }
}

extern "C" int32_t invar_pinocchio_v1_configure(
    inv_fk_context_v1 *context, const inv_fk_config_v1 *config) {
  if (!context) return INV_FK_BAD_STATE;
  if (context->fk) return INV_FK_BAD_STATE;
  if (!config || !valid_utf8_path(config->model_xml_path,
                                  config->model_xml_path_len))
    return INV_FK_BAD_ARGUMENT;
  if (config->model_kind != INV_FK_PANDA && config->model_kind != INV_FK_UR5E)
    return INV_FK_BAD_ARGUMENT;

  try {
    const auto *bytes = config->model_xml_path;
    const std::string path(reinterpret_cast<const char *>(bytes),
                           config->model_xml_path_len);
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error)
      return INV_FK_BAD_ARGUMENT;
    const bool panda = config->model_kind == INV_FK_PANDA;
    const std::vector<std::string> names = panda
        ? std::vector<std::string>{"joint1", "joint2", "joint3", "joint4",
                                   "joint5", "joint6", "joint7"}
        : std::vector<std::string>{"shoulder_pan_joint", "shoulder_lift_joint",
                                   "elbow_joint", "wrist_1_joint",
                                   "wrist_2_joint", "wrist_3_joint"};
    auto fk = std::make_unique<invar_fixture::PinocchioFk>(
        path, panda ? "link0" : "base", panda ? "hand" : "attachment_site",
        names);
    context->fk = std::move(fk);
    context->model_kind = config->model_kind;
    return INV_FK_OK;
  } catch (...) {
    return INV_FK_LIBRARY_FAILURE;
  }
}

extern "C" int32_t invar_pinocchio_v1_step(inv_fk_context_v1 *context,
                                            const double *joint_rad,
                                            uint32_t joint_count,
                                            inv_pose_f64 *out_pose) {
  if (!context || !context->fk) return INV_FK_BAD_STATE;
  if (!joint_rad || !out_pose) return INV_FK_BAD_ARGUMENT;
  if (joint_count != context->model_kind) return INV_FK_BAD_COUNT;
  for (uint32_t i = 0; i < joint_count; ++i) {
    if (!std::isfinite(joint_rad[i])) return INV_FK_NONFINITE;
  }
  try {
    const std::vector<double> q(joint_rad, joint_rad + joint_count);
    const auto result = context->fk->step(q);
    const auto &rotation = result.quaternion_wxyz;
    double norm_squared = 0.0;
    for (double value : rotation) norm_squared += value * value;
    if (!std::isfinite(norm_squared) ||
        std::abs(std::sqrt(norm_squared) - 1.0) > 1e-12)
      return INV_FK_LIBRARY_FAILURE;
    inv_pose_f64 pose{};
    for (int i = 0; i < 3; ++i) pose.translation_m[i] = result.translation_m[i];
    for (int i = 0; i < 4; ++i)
      pose.quaternion_xyzw[i] = rotation[(i + 1) % 4];
    *out_pose = pose;
    return INV_FK_OK;
  } catch (const invar_fixture::JointInputError &) {
    return INV_FK_BAD_ARGUMENT;
  } catch (...) {
    return INV_FK_LIBRARY_FAILURE;
  }
}

extern "C" int32_t invar_pinocchio_v1_shutdown(inv_fk_context_v1 **context) {
  if (!context || !*context) return INV_FK_BAD_STATE;
  try {
    delete *context;
    *context = nullptr;
    return INV_FK_OK;
  } catch (...) {
    return INV_FK_LIBRARY_FAILURE;
  }
}
