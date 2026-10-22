#ifndef INVAR_CUROBO_PLANNER_REFERENCE_H
#define INVAR_CUROBO_PLANNER_REFERENCE_H

/* Hand-authored S1-06 reference; main owns generated bindings and acceptance. */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  INV_PLANNER_OK = 0,
  INV_PLANNER_INVALID = 1,
  INV_PLANNER_LIFECYCLE = 2,
  INV_PLANNER_FAILED = 3,
  INV_PLANNER_CHILD_EXIT = 4,
  INV_PLANNER_TIMEOUT = 5,
  INV_PLANNER_MALFORMED = 6,
  INV_PLANNER_CANCELLED = 7,
  INV_PLANNER_MAX_WAYPOINTS = 256
};

typedef struct inv_planner_context_v1 inv_planner_context_v1;
typedef struct { double xyz_m[3], quaternion_xyzw[4]; } inv_planner_pose_v1;
typedef struct { const uint8_t *bytes; uint16_t length; } inv_planner_text_v1;

typedef struct {
  uint32_t robot_kind; /* 6 = UR5e, 7 = Panda */
  uint32_t joint_count;
  uint8_t model_sha256[32], robot_config_sha256[32], scene_sha256[32];
  double joint_limits_rad[7][2];
  inv_planner_pose_v1 table_pose;
  double table_half_extents_m[3];
  double cube_edge_m; /* exactly 0.05 */
  inv_planner_text_v1 base_frame, tool_frame, joint_names[7];
  inv_planner_text_v1 robot_config_path, scene_config_path;
} inv_planner_config_v1;

enum {
  INV_PLANNER_PHASE_FREE = 0,         /* cube is a world obstacle */
  INV_PLANNER_PHASE_APPROACH = 1,     /* cube obstacle disabled */
  INV_PLANNER_PHASE_HELD = 2,         /* cube attached, full cover */
  INV_PLANNER_PHASE_HELD_CONTACT = 3, /* cube attached, table-contact slab uncovered */
  INV_PLANNER_PHASE_HOME = 4          /* cube obstacle, joint goal goal_q_rad */
};

/* goal_q_rad equals start_q_rad bitwise except in INV_PLANNER_PHASE_HOME. */
typedef struct {
  uint32_t joint_count;
  double start_q_rad[7], goal_q_rad[7];
  inv_planner_pose_v1 target_base_from_tool, cube_base_pose, tool_from_cube;
  uint32_t phase;
} inv_planner_request_v2;

typedef struct {
  double time_from_start_s, q_rad[7], dq_rad_s[7];
} inv_planner_waypoint_v1;
typedef struct {
  double dt_s;
  uint32_t count;
  inv_planner_waypoint_v1 waypoints[INV_PLANNER_MAX_WAYPOINTS];
} inv_planner_result_v1;

const char *invar_planner_v1_version(void);
int32_t invar_planner_v1_init(inv_planner_context_v1 **out_context,
                              const uint8_t *absolute_executable_path, uint32_t path_len);
int32_t invar_planner_v1_configure(inv_planner_context_v1 *context,
                                   const inv_planner_config_v1 *config);
/* Serialized calls. Result remains byte-identical on failure. */
int32_t invar_planner_v1_plan(inv_planner_context_v1 *context,
                              const inv_planner_request_v2 *request,
                              inv_planner_result_v1 *out_result);
/* Concurrent with plan only; terminates its owned child and waits for plan to reap it. */
int32_t invar_planner_v1_cancel(inv_planner_context_v1 *context);
int32_t invar_planner_v1_shutdown(inv_planner_context_v1 **context);

#ifdef __cplusplus
}
#endif
#endif
