#ifndef INVAR_PROCESS_FIXTURE_REFERENCE_H
#define INVAR_PROCESS_FIXTURE_REFERENCE_H

/* HAND-AUTHORED S1-02 REFERENCE FIXTURE, not a generated binding or cuRobo. */
#include <stdint.h>
#include "../pinocchio_fk/abi_reference.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inv_process_context_v1 inv_process_context_v1;
enum {
  INV_PROCESS_OK = 0,
  INV_PROCESS_INVALID_INPUT = 1,
  INV_PROCESS_LIFECYCLE = 2,
  INV_PROCESS_COMPONENT_ERROR = 3,
  INV_PROCESS_CHILD_EXIT = 4,
  INV_PROCESS_TIMEOUT = 5,
  INV_PROCESS_MALFORMED = 6
};

const char *invar_process_v1_version(void);
/* Caller starts with *out_context == NULL and owns it until shutdown. */
int32_t invar_process_v1_init(inv_process_context_v1 **out_context,
                               const uint8_t *absolute_executable_path,
                               uint32_t path_len);
int32_t invar_process_v1_configure(inv_process_context_v1 *context,
                                    const inv_fk_config_v1 *config);
/* On any failure, *out_pose remains byte-identical. Calls are serialized. */
int32_t invar_process_v1_step(inv_process_context_v1 *context,
                               const double *joint_rad, uint32_t joint_count,
                               inv_pose_f64 *out_pose);
int32_t invar_process_v1_shutdown(inv_process_context_v1 **context);

#ifdef __cplusplus
}
#endif

#endif
