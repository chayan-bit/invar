#ifndef INVAR_PINOCCHIO_ABI_REFERENCE_H
#define INVAR_PINOCCHIO_ABI_REFERENCE_H

/* HAND-AUTHORED REFERENCE ABI. Not an Invar-generated header or S1 acceptance. */
#include <stddef.h>
#include <stdint.h>

#if UINTPTR_MAX != UINT64_MAX
#error "Pinocchio reference ABI requires a 64-bit C target"
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inv_fk_context_v1 inv_fk_context_v1;
/* Joint values are radians. Pose maps tool coordinates into the base frame. */
typedef struct { double values[6]; } inv_joint6_f64;
typedef struct { double values[7]; } inv_joint7_f64;
typedef struct {
  double translation_m[3];
  double quaternion_xyzw[4];
} inv_pose_f64;
typedef struct {
  /* Borrowed UTF-8 bytes for this configure call; no NUL terminator required. */
  const uint8_t *model_xml_path;
  uint32_t model_xml_path_len;
  uint32_t model_kind;
} inv_fk_config_v1;

enum {
  INV_FK_PANDA = 7,
  INV_FK_UR5E = 6,
  INV_FK_OK = 0,
  INV_FK_BAD_ARGUMENT = 1,
  INV_FK_BAD_STATE = 2,
  INV_FK_BAD_COUNT = 3,
  INV_FK_NONFINITE = 4,
  INV_FK_LIBRARY_FAILURE = 5
};

const char *invar_pinocchio_v1_version(void);
/* Caller initializes *out_context to NULL and releases it with shutdown. */
int32_t invar_pinocchio_v1_init(inv_fk_context_v1 **out_context);
int32_t invar_pinocchio_v1_configure(inv_fk_context_v1 *context,
                                     const inv_fk_config_v1 *config);
/* Every non-OK step leaves out_pose byte-identical. */
int32_t invar_pinocchio_v1_step(inv_fk_context_v1 *context,
                                const double *joint_rad, uint32_t joint_count,
                                inv_pose_f64 *out_pose);
int32_t invar_pinocchio_v1_shutdown(inv_fk_context_v1 **context);

#ifdef __cplusplus
}
#endif

#endif
