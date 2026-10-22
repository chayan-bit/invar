#include "invar_process_bridge_v1.h"
#include "process_reference.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(invar_process_bridge_v1_joint7_f64) ==
                   sizeof(inv_joint7_f64) &&
                   _Alignof(invar_process_bridge_v1_joint7_f64) ==
                       _Alignof(inv_joint7_f64) &&
                   offsetof(invar_process_bridge_v1_joint7_f64, values) ==
                       offsetof(inv_joint7_f64, values),
               "generated joint layout differs from reference");
_Static_assert(sizeof(invar_process_bridge_v1_pose_f64) ==
                   sizeof(inv_pose_f64) &&
                   _Alignof(invar_process_bridge_v1_pose_f64) ==
                       _Alignof(inv_pose_f64) &&
                   offsetof(invar_process_bridge_v1_pose_f64, translation_m) ==
                       offsetof(inv_pose_f64, translation_m) &&
                   offsetof(invar_process_bridge_v1_pose_f64, quaternion_xyzw) ==
                       offsetof(inv_pose_f64, quaternion_xyzw),
               "generated pose layout differs from reference");
_Static_assert(sizeof(invar_process_bridge_v1_config) ==
                   sizeof(inv_fk_config_v1) &&
                   _Alignof(invar_process_bridge_v1_config) ==
                       _Alignof(inv_fk_config_v1) &&
                   offsetof(invar_process_bridge_v1_config, model_xml_path) ==
                       offsetof(inv_fk_config_v1, model_xml_path) &&
                   offsetof(invar_process_bridge_v1_config, model_xml_path_len) ==
                       offsetof(inv_fk_config_v1, model_xml_path_len) &&
                   offsetof(invar_process_bridge_v1_config, model_kind) ==
                       offsetof(inv_fk_config_v1, model_kind),
               "generated config layout differs from reference");
_Static_assert(sizeof(invar_process_bridge_v1_inputs) ==
                   sizeof(inv_joint7_f64) &&
                   _Alignof(invar_process_bridge_v1_inputs) == 8 &&
                   offsetof(invar_process_bridge_v1_inputs, q) == 0 &&
                   sizeof(invar_process_bridge_v1_outputs) ==
                       sizeof(inv_pose_f64) &&
                   _Alignof(invar_process_bridge_v1_outputs) == 8 &&
                   offsetof(invar_process_bridge_v1_outputs, pose) == 0,
               "generated port layout differs from reference");

struct invar_process_bridge_v1_context {
  inv_process_context_v1 *reference;
};

static int32_t map_status(int32_t status) {
  switch (status) {
    case INV_PROCESS_OK: return INVAR_PROCESS_BRIDGE_V1_OK;
    case INV_PROCESS_INVALID_INPUT: return INVAR_PROCESS_BRIDGE_V1_INVALID_INPUT;
    case INV_PROCESS_LIFECYCLE: return INVAR_PROCESS_BRIDGE_V1_LIFECYCLE;
    case INV_PROCESS_COMPONENT_ERROR: return INVAR_PROCESS_BRIDGE_V1_FAILURE;
    case INV_PROCESS_CHILD_EXIT: return INVAR_PROCESS_BRIDGE_V1_CHILD_EXIT;
    case INV_PROCESS_TIMEOUT: return INVAR_PROCESS_BRIDGE_V1_TIMEOUT;
    case INV_PROCESS_MALFORMED: return INVAR_PROCESS_BRIDGE_V1_MALFORMED;
    default: return INVAR_PROCESS_BRIDGE_V1_FAILURE;
  }
}

const char *invar_process_bridge_v1_version(void) {
  return "invar-component-v1/invar-process-reference-abi-1";
}

int32_t invar_process_bridge_v1_init(
    invar_process_bridge_v1_context **out_context,
    const uint8_t *launch_path, uint32_t launch_path_len) {
  if (!out_context || !launch_path || !launch_path_len || launch_path[0] != '/')
    return INVAR_PROCESS_BRIDGE_V1_INVALID_INPUT;
  if (*out_context) return INVAR_PROCESS_BRIDGE_V1_LIFECYCLE;
  invar_process_bridge_v1_context *context = calloc(1, sizeof *context);
  if (!context) return INVAR_PROCESS_BRIDGE_V1_FAILURE;
  int32_t status = invar_process_v1_init(
      &context->reference, launch_path, launch_path_len);
  if (status != INV_PROCESS_OK || !context->reference) {
    if (context->reference) invar_process_v1_shutdown(&context->reference);
    free(context);
    return status == INV_PROCESS_OK ? INVAR_PROCESS_BRIDGE_V1_FAILURE
                                    : map_status(status);
  }
  *out_context = context;
  return INVAR_PROCESS_BRIDGE_V1_OK;
}

int32_t invar_process_bridge_v1_configure(
    invar_process_bridge_v1_context *context,
    const invar_process_bridge_v1_config *config) {
  if (!context) return INVAR_PROCESS_BRIDGE_V1_LIFECYCLE;
  if (!config || config->model_kind != INV_FK_PANDA)
    return INVAR_PROCESS_BRIDGE_V1_INVALID_INPUT;
  const inv_fk_config_v1 reference = {
      config->model_xml_path, config->model_xml_path_len, config->model_kind};
  return map_status(invar_process_v1_configure(context->reference, &reference));
}

int32_t invar_process_bridge_v1_step(
    invar_process_bridge_v1_context *context,
    const invar_process_bridge_v1_inputs *input,
    invar_process_bridge_v1_outputs *output) {
  if (!context) return INVAR_PROCESS_BRIDGE_V1_LIFECYCLE;
  if (!input || !output) return INVAR_PROCESS_BRIDGE_V1_INVALID_INPUT;
  for (size_t i = 0; i < 7; ++i)
    if (!isfinite(input->q.values[i]))
      return INVAR_PROCESS_BRIDGE_V1_INVALID_INPUT;
  inv_pose_f64 candidate;
  int32_t status = invar_process_v1_step(
      context->reference, input->q.values, 7, &candidate);
  if (status != INV_PROCESS_OK) return map_status(status);
  memcpy(&output->pose, &candidate, sizeof candidate);
  return INVAR_PROCESS_BRIDGE_V1_OK;
}

int32_t invar_process_bridge_v1_shutdown(
    invar_process_bridge_v1_context **context) {
  if (!context || !*context) return INVAR_PROCESS_BRIDGE_V1_LIFECYCLE;
  int32_t status = invar_process_v1_shutdown(&(*context)->reference);
  if ((*context)->reference) return INVAR_PROCESS_BRIDGE_V1_FAILURE;
  free(*context);
  *context = NULL;
  return map_status(status);
}
