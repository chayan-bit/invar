/* Typed generated ABI facade over the reviewed INVP v3 process adapter.
 * Build once per checked six/seven-joint header. No planner algorithm lives here.
 */
#include "invar_curobo_bridge_v1.h"
#include "planner_reference.h"
#include <stdlib.h>
#include <string.h>

struct invar_curobo_bridge_v1_context { inv_planner_context_v1 *reference; };

_Static_assert(sizeof(invar_curobo_bridge_v1_config) == sizeof(inv_planner_config_v1), "config ABI");
_Static_assert(offsetof(invar_curobo_bridge_v1_config, base_frame) == offsetof(inv_planner_config_v1, base_frame), "borrowed text ABI");
_Static_assert(offsetof(invar_curobo_bridge_v1_config, scene_config_path) == offsetof(inv_planner_config_v1, scene_config_path), "scene path ABI");
_Static_assert(sizeof(inv_planner_pose_wire_v1) == sizeof(inv_planner_pose_v1), "pose ABI");

const char *invar_curobo_bridge_v1_version(void) {
  return INVAR_CUROBO_BRIDGE_V1_JOINT_COUNT == 6
      ? "invar-planner-component-v2/6/INVP-v3"
      : "invar-planner-component-v2/7/INVP-v3";
}

int32_t invar_curobo_bridge_v1_init(invar_curobo_bridge_v1_context **out,
                                   const uint8_t *path, uint32_t length) {
  if (!out || *out) return INV_PLANNER_INVALID;
  invar_curobo_bridge_v1_context *context = calloc(1, sizeof(*context));
  if (!context) return INV_PLANNER_FAILED;
  int32_t status = invar_planner_v1_init(&context->reference, path, length);
  if (status != INV_PLANNER_OK) { free(context); return status; }
  *out = context;
  return INV_PLANNER_OK;
}

int32_t invar_curobo_bridge_v1_configure(invar_curobo_bridge_v1_context *context,
                                        const invar_curobo_bridge_v1_config *config) {
  if (!context || !context->reference) return INV_PLANNER_LIFECYCLE;
  if (!config || config->robot_kind != INVAR_CUROBO_BRIDGE_V1_JOINT_COUNT ||
      config->joint_count != INVAR_CUROBO_BRIDGE_V1_JOINT_COUNT) return INV_PLANNER_INVALID;
  inv_planner_config_v1 reference;
  /* Identical declared C member layouts, checked above; memcpy avoids aliasing
   * unrelated struct types. Pointees are borrowed only during configure. */
  memcpy(&reference, config, sizeof(reference));
  return invar_planner_v1_configure(context->reference, &reference);
}

int32_t invar_curobo_bridge_v1_step(invar_curobo_bridge_v1_context *context,
                                   const invar_curobo_bridge_v1_inputs *input,
                                   invar_curobo_bridge_v1_outputs *output) {
  if (!context || !context->reference) return INV_PLANNER_LIFECYCLE;
  if (!input || !output) return INV_PLANNER_INVALID;
  inv_planner_request_v2 request = {0};
  request.joint_count = INVAR_CUROBO_BRIDGE_V1_JOINT_COUNT;
  for (uint32_t i = 0; i < request.joint_count; ++i) {
    request.start_q_rad[i] = input->port_request.start_q_rad[i];
    request.goal_q_rad[i] = input->port_request.goal_q_rad[i];
  }
  memcpy(&request.target_base_from_tool, &input->port_request.target_base_from_tool, sizeof(request.target_base_from_tool));
  memcpy(&request.cube_base_pose, &input->port_request.cube_base_pose, sizeof(request.cube_base_pose));
  memcpy(&request.tool_from_cube, &input->port_request.tool_from_cube, sizeof(request.tool_from_cube));
  request.phase = input->port_request.phase;
  inv_planner_result_v1 planned = {0};
  int32_t status = invar_planner_v1_plan(context->reference, &request, &planned);
  if (status != INV_PLANNER_OK) return status;
  if (planned.count < 2 || planned.count > 256) return INV_PLANNER_MALFORMED;
  invar_curobo_bridge_v1_outputs candidate = {0};
  candidate.port_trajectory.dt_s = planned.dt_s;
  candidate.port_trajectory.count = planned.count;
  for (uint32_t k = 0; k < planned.count; ++k) {
    candidate.port_trajectory.waypoints[k].time_from_start_s = planned.waypoints[k].time_from_start_s;
    for (uint32_t i = 0; i < request.joint_count; ++i) {
      candidate.port_trajectory.waypoints[k].q_rad[i] = planned.waypoints[k].q_rad[i];
      candidate.port_trajectory.waypoints[k].dq_rad_s[i] = planned.waypoints[k].dq_rad_s[i];
    }
  }
  *output = candidate;
  return INV_PLANNER_OK;
}

int32_t invar_curobo_bridge_v1_shutdown(invar_curobo_bridge_v1_context **out) {
  if (!out || !*out) return INV_PLANNER_LIFECYCLE;
  invar_curobo_bridge_v1_context *context = *out;
  int32_t status = invar_planner_v1_shutdown(&context->reference);
  if (!context->reference) { free(context); *out = NULL; }
  return status;
}
