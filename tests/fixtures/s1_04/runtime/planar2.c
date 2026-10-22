#include "planar2.h"

#include <math.h>

enum { NEW = 0, INITIALIZED = 1, CONFIGURED = 2, CLOSED = 3 };

FixturePlanar2Status fixture_planar2_init(FixturePlanar2 *arm) {
  if (arm == NULL) return FIXTURE_PLANAR2_BAD_INPUT;
  if (arm->state != NEW) return FIXTURE_PLANAR2_BAD_STATE;
  arm->proximal_m = 0.0;
  arm->distal_m = 0.0;
  arm->state = INITIALIZED;
  return FIXTURE_PLANAR2_OK;
}

FixturePlanar2Status fixture_planar2_configure(FixturePlanar2 *arm,
                                                double proximal_m,
                                                double distal_m) {
  if (arm == NULL) return FIXTURE_PLANAR2_BAD_INPUT;
  if (arm->state != INITIALIZED) return FIXTURE_PLANAR2_BAD_STATE;
  if (!isfinite(proximal_m) || !isfinite(distal_m) || proximal_m <= 0.0 ||
      distal_m <= 0.0) return FIXTURE_PLANAR2_BAD_INPUT;
  arm->proximal_m = proximal_m;
  arm->distal_m = distal_m;
  arm->state = CONFIGURED;
  return FIXTURE_PLANAR2_OK;
}

FixturePlanar2Status fixture_planar2_step(const FixturePlanar2 *arm,
                                           const double *joint_rad,
                                           size_t joint_count, double *x_m,
                                           double *y_m, double *yaw_rad) {
  if (arm == NULL) return FIXTURE_PLANAR2_BAD_INPUT;
  if (arm->state != CONFIGURED) return FIXTURE_PLANAR2_BAD_STATE;
  if (joint_rad == NULL || x_m == NULL || y_m == NULL || yaw_rad == NULL ||
      joint_count != 2 || !isfinite(joint_rad[0]) || !isfinite(joint_rad[1]))
    return FIXTURE_PLANAR2_BAD_INPUT;

  const double yaw = joint_rad[0] + joint_rad[1];
  if (!isfinite(yaw)) return FIXTURE_PLANAR2_BAD_INPUT;
  const double x = arm->proximal_m * cos(joint_rad[0]) + arm->distal_m * cos(yaw);
  const double y = arm->proximal_m * sin(joint_rad[0]) + arm->distal_m * sin(yaw);
  if (!isfinite(x) || !isfinite(y)) return FIXTURE_PLANAR2_BAD_INPUT;
  *x_m = x;
  *y_m = y;
  *yaw_rad = yaw;
  return FIXTURE_PLANAR2_OK;
}

FixturePlanar2Status fixture_planar2_shutdown(FixturePlanar2 *arm) {
  if (arm == NULL) return FIXTURE_PLANAR2_BAD_INPUT;
  if (arm->state != CONFIGURED) return FIXTURE_PLANAR2_BAD_STATE;
  arm->proximal_m = 0.0;
  arm->distal_m = 0.0;
  arm->state = CLOSED;
  return FIXTURE_PLANAR2_OK;
}

const char *fixture_planar2_version(void) { return "planar2-fixture-1"; }
