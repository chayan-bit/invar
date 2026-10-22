#include "invar_planar2_bridge_v1.h"
#include "planar2.h"

#include <math.h>
#include <stdlib.h>

#if defined(__GNUC__) || defined(__clang__)
#define PUBLIC __attribute__((visibility("default")))
#else
#define PUBLIC
#endif

struct invar_planar2_bridge_v1_context {
  FixturePlanar2 arm;
};

PUBLIC const char *invar_planar2_bridge_v1_version(void) {
  return "invar-component-v1/planar2-fixture-1";
}

PUBLIC int32_t invar_planar2_bridge_v1_init(
    invar_planar2_bridge_v1_context **out_context,
    const uint8_t *launch_path, uint32_t launch_path_len) {
  if (!out_context || *out_context || launch_path || launch_path_len)
    return INVAR_PLANAR2_BRIDGE_V1_INVALID_INPUT;
  invar_planar2_bridge_v1_context *context =
      (invar_planar2_bridge_v1_context *)calloc(1, sizeof(*context));
  if (!context) return INVAR_PLANAR2_BRIDGE_V1_FAILURE;
  if (fixture_planar2_init(&context->arm) != FIXTURE_PLANAR2_OK) {
    free(context);
    return INVAR_PLANAR2_BRIDGE_V1_FAILURE;
  }
  *out_context = context;
  return INVAR_PLANAR2_BRIDGE_V1_OK;
}

PUBLIC int32_t invar_planar2_bridge_v1_configure(
    invar_planar2_bridge_v1_context *context,
    const invar_planar2_bridge_v1_config *config) {
  if (!context) return INVAR_PLANAR2_BRIDGE_V1_LIFECYCLE;
  if (!config) return INVAR_PLANAR2_BRIDGE_V1_INVALID_INPUT;
  FixturePlanar2Status status = fixture_planar2_configure(
      &context->arm, config->proximal_m, config->distal_m);
  if (status == FIXTURE_PLANAR2_BAD_STATE)
    return INVAR_PLANAR2_BRIDGE_V1_LIFECYCLE;
  if (status == FIXTURE_PLANAR2_BAD_INPUT)
    return INVAR_PLANAR2_BRIDGE_V1_INVALID_INPUT;
  return status == FIXTURE_PLANAR2_OK ? INVAR_PLANAR2_BRIDGE_V1_OK
                                      : INVAR_PLANAR2_BRIDGE_V1_FAILURE;
}

PUBLIC int32_t invar_planar2_bridge_v1_step(
    invar_planar2_bridge_v1_context *context,
    const invar_planar2_bridge_v1_inputs *input,
    invar_planar2_bridge_v1_outputs *output) {
  if (!context) return INVAR_PLANAR2_BRIDGE_V1_LIFECYCLE;
  if (!input || !output) return INVAR_PLANAR2_BRIDGE_V1_INVALID_INPUT;
  double x = 0.0, y = 0.0, yaw = 0.0;
  FixturePlanar2Status status = fixture_planar2_step(
      &context->arm, input->q.values, 2, &x, &y, &yaw);
  if (status == FIXTURE_PLANAR2_BAD_STATE)
    return INVAR_PLANAR2_BRIDGE_V1_LIFECYCLE;
  if (status == FIXTURE_PLANAR2_BAD_INPUT)
    return INVAR_PLANAR2_BRIDGE_V1_INVALID_INPUT;
  if (status != FIXTURE_PLANAR2_OK || !isfinite(x) || !isfinite(y) || !isfinite(yaw))
    return INVAR_PLANAR2_BRIDGE_V1_FAILURE;
  invar_planar2_bridge_v1_outputs result = {0};
  result.pose.translation_m[0] = x;
  result.pose.translation_m[1] = y;
  result.pose.quaternion_xyzw[2] = sin(yaw / 2.0);
  result.pose.quaternion_xyzw[3] = cos(yaw / 2.0);
  *output = result;
  return INVAR_PLANAR2_BRIDGE_V1_OK;
}

PUBLIC int32_t invar_planar2_bridge_v1_shutdown(
    invar_planar2_bridge_v1_context **context) {
  if (!context || !*context) return INVAR_PLANAR2_BRIDGE_V1_LIFECYCLE;
  if ((*context)->arm.state == 2)
    (void)fixture_planar2_shutdown(&(*context)->arm);
  free(*context);
  *context = NULL;
  return INVAR_PLANAR2_BRIDGE_V1_OK;
}
