#include "../../../examples/cells/reach_place/components/pinocchio_fk/abi_reference.h"

#include <stddef.h>
#include <string.h>

_Static_assert(sizeof(inv_joint6_f64) == 48 && _Alignof(inv_joint6_f64) == 8 &&
                   offsetof(inv_joint6_f64, values) == 0,
               "joint6 C layout");
_Static_assert(sizeof(inv_joint7_f64) == 56 && _Alignof(inv_joint7_f64) == 8 &&
                   offsetof(inv_joint7_f64, values) == 0,
               "joint7 C layout");
_Static_assert(sizeof(inv_pose_f64) == 56 && _Alignof(inv_pose_f64) == 8 &&
                   offsetof(inv_pose_f64, translation_m) == 0 &&
                   offsetof(inv_pose_f64, quaternion_xyzw) == 24,
               "pose C layout");
_Static_assert(sizeof(inv_fk_config_v1) == 16 && _Alignof(inv_fk_config_v1) == 8 &&
                   offsetof(inv_fk_config_v1, model_xml_path) == 0 &&
                   offsetof(inv_fk_config_v1, model_xml_path_len) == 8 &&
                   offsetof(inv_fk_config_v1, model_kind) == 12,
               "config C layout");

int main(void) {
  const inv_joint6_f64 joint6 = {{0.0, 1.0, 2.0, 3.0, 4.0, 5.0}};
  const inv_joint7_f64 joint7 = {{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
  const inv_pose_f64 pose = {{1.0, 2.0, 3.0}, {0.0, 0.0, 0.0, 1.0}};
  inv_joint6_f64 joint6_roundtrip;
  inv_joint7_f64 joint7_roundtrip;
  inv_pose_f64 pose_roundtrip;
  unsigned char joint6_bytes[sizeof joint6];
  unsigned char joint7_bytes[sizeof joint7];
  unsigned char pose_bytes[sizeof pose];
  memcpy(joint6_bytes, &joint6, sizeof joint6);
  memcpy(joint7_bytes, &joint7, sizeof joint7);
  memcpy(pose_bytes, &pose, sizeof pose);
  memcpy(&joint6_roundtrip, joint6_bytes, sizeof joint6);
  memcpy(&joint7_roundtrip, joint7_bytes, sizeof joint7);
  memcpy(&pose_roundtrip, pose_bytes, sizeof pose);
  if (memcmp(&joint6, &joint6_roundtrip, sizeof joint6) ||
      memcmp(&joint7, &joint7_roundtrip, sizeof joint7) ||
      memcmp(&pose, &pose_roundtrip, sizeof pose))
    return 5;

  inv_fk_context_v1 *context = NULL;
  if (!invar_pinocchio_v1_version() ||
      strcmp(invar_pinocchio_v1_version(),
             "invar-pinocchio-reference-abi-1/pinocchio-4.1.0") != 0)
    return 1;
  if (invar_pinocchio_v1_init(&context) != INV_FK_OK || !context) return 2;
  if (invar_pinocchio_v1_shutdown(&context) != INV_FK_OK || context) return 3;
  if (invar_pinocchio_v1_shutdown(&context) != INV_FK_BAD_STATE) return 4;
  return 0;
}
