#include "invar_pinocchio_bridge_v1.h"
#include "abi_reference.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef INVAR_FK_UR5E
#define INVAR_JOINT invar_pinocchio_bridge_v1_joint6_f64
#define INV_JOINT inv_joint6_f64
#define JOINT_COUNT INV_FK_UR5E
#else
#define INVAR_JOINT invar_pinocchio_bridge_v1_joint7_f64
#define INV_JOINT inv_joint7_f64
#define JOINT_COUNT INV_FK_PANDA
#endif

_Static_assert(sizeof(INVAR_JOINT) == sizeof(INV_JOINT) &&
                   _Alignof(INVAR_JOINT) == _Alignof(INV_JOINT) &&
                   offsetof(INVAR_JOINT, values) == offsetof(INV_JOINT, values),
               "generated joint layout differs from reference");
_Static_assert(sizeof(invar_pinocchio_bridge_v1_pose_f64) ==
                   sizeof(inv_pose_f64) &&
                   _Alignof(invar_pinocchio_bridge_v1_pose_f64) ==
                       _Alignof(inv_pose_f64) &&
                   offsetof(invar_pinocchio_bridge_v1_pose_f64, translation_m) ==
                       offsetof(inv_pose_f64, translation_m) &&
                   offsetof(invar_pinocchio_bridge_v1_pose_f64, quaternion_xyzw) ==
                       offsetof(inv_pose_f64, quaternion_xyzw),
               "generated pose layout differs from reference");
_Static_assert(sizeof(invar_pinocchio_bridge_v1_config) ==
                   sizeof(inv_fk_config_v1) &&
                   _Alignof(invar_pinocchio_bridge_v1_config) ==
                       _Alignof(inv_fk_config_v1) &&
                   offsetof(invar_pinocchio_bridge_v1_config, model_xml_path) ==
                       offsetof(inv_fk_config_v1, model_xml_path) &&
                   offsetof(invar_pinocchio_bridge_v1_config, model_xml_path_len) ==
                       offsetof(inv_fk_config_v1, model_xml_path_len) &&
                   offsetof(invar_pinocchio_bridge_v1_config, model_kind) ==
                       offsetof(inv_fk_config_v1, model_kind),
               "generated config layout differs from reference");
_Static_assert(sizeof(invar_pinocchio_bridge_v1_inputs) ==
                   sizeof(INV_JOINT) &&
                   _Alignof(invar_pinocchio_bridge_v1_inputs) == 8 &&
                   offsetof(invar_pinocchio_bridge_v1_inputs, q) == 0 &&
                   sizeof(invar_pinocchio_bridge_v1_outputs) ==
                       sizeof(inv_pose_f64) &&
                   _Alignof(invar_pinocchio_bridge_v1_outputs) == 8 &&
                   offsetof(invar_pinocchio_bridge_v1_outputs, pose) == 0,
               "generated port layout differs from reference");

struct invar_pinocchio_bridge_v1_context {
  inv_fk_context_v1 *reference;
};

static int32_t map_status(int32_t status) {
  switch (status) {
    case INV_FK_OK: return INVAR_PINOCCHIO_BRIDGE_V1_OK;
    case INV_FK_BAD_ARGUMENT:
    case INV_FK_BAD_COUNT:
    case INV_FK_NONFINITE: return INVAR_PINOCCHIO_BRIDGE_V1_INVALID_INPUT;
    case INV_FK_BAD_STATE: return INVAR_PINOCCHIO_BRIDGE_V1_LIFECYCLE;
    default: return INVAR_PINOCCHIO_BRIDGE_V1_FAILURE;
  }
}

const char *invar_pinocchio_bridge_v1_version(void) {
  return "invar-component-v1/invar-pinocchio-reference-abi-1/pinocchio-4.1.0";
}

int32_t invar_pinocchio_bridge_v1_init(
    invar_pinocchio_bridge_v1_context **out_context,
    const uint8_t *launch_path, uint32_t launch_path_len) {
  if (!out_context || launch_path || launch_path_len)
    return INVAR_PINOCCHIO_BRIDGE_V1_INVALID_INPUT;
  if (*out_context) return INVAR_PINOCCHIO_BRIDGE_V1_LIFECYCLE;
  invar_pinocchio_bridge_v1_context *context = calloc(1, sizeof *context);
  if (!context) return INVAR_PINOCCHIO_BRIDGE_V1_FAILURE;
  int32_t status = invar_pinocchio_v1_init(&context->reference);
  if (status != INV_FK_OK) {
    free(context);
    return map_status(status);
  }
  *out_context = context;
  return INVAR_PINOCCHIO_BRIDGE_V1_OK;
}

int32_t invar_pinocchio_bridge_v1_configure(
    invar_pinocchio_bridge_v1_context *context,
    const invar_pinocchio_bridge_v1_config *config) {
  if (!context) return INVAR_PINOCCHIO_BRIDGE_V1_LIFECYCLE;
  if (!config || config->model_kind != JOINT_COUNT)
    return INVAR_PINOCCHIO_BRIDGE_V1_INVALID_INPUT;
  const inv_fk_config_v1 reference = {
      config->model_xml_path, config->model_xml_path_len, config->model_kind};
  return map_status(invar_pinocchio_v1_configure(context->reference, &reference));
}

int32_t invar_pinocchio_bridge_v1_step(
    invar_pinocchio_bridge_v1_context *context,
    const invar_pinocchio_bridge_v1_inputs *input,
    invar_pinocchio_bridge_v1_outputs *output) {
  if (!context) return INVAR_PINOCCHIO_BRIDGE_V1_LIFECYCLE;
  if (!input || !output) return INVAR_PINOCCHIO_BRIDGE_V1_INVALID_INPUT;
  for (size_t i = 0; i < JOINT_COUNT; ++i)
    if (!isfinite(input->q.values[i]))
      return INVAR_PINOCCHIO_BRIDGE_V1_INVALID_INPUT;
  inv_pose_f64 candidate;
  int32_t status = invar_pinocchio_v1_step(
      context->reference, input->q.values, JOINT_COUNT, &candidate);
  if (status != INV_FK_OK) return map_status(status);
  memcpy(&output->pose, &candidate, sizeof candidate);
  return INVAR_PINOCCHIO_BRIDGE_V1_OK;
}

int32_t invar_pinocchio_bridge_v1_shutdown(
    invar_pinocchio_bridge_v1_context **context) {
  if (!context || !*context) return INVAR_PINOCCHIO_BRIDGE_V1_LIFECYCLE;
  int32_t status = invar_pinocchio_v1_shutdown(&(*context)->reference);
  if (status != INV_FK_OK) return map_status(status);
  free(*context);
  *context = NULL;
  return INVAR_PINOCCHIO_BRIDGE_V1_OK;
}
