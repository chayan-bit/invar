#ifndef INVAR_FIXTURE_PLANAR2_H
#define INVAR_FIXTURE_PLANAR2_H

#include <stddef.h>

/* Private fixture API. The generated S1-02 header will own the public ABI. */
typedef enum {
  FIXTURE_PLANAR2_OK = 0,
  FIXTURE_PLANAR2_BAD_STATE = 1,
  FIXTURE_PLANAR2_BAD_INPUT = 2
} FixturePlanar2Status;

typedef struct {
  unsigned state; /* Zero-initialize before init. */
  double proximal_m;
  double distal_m;
} FixturePlanar2;

FixturePlanar2Status fixture_planar2_init(FixturePlanar2 *arm);
FixturePlanar2Status fixture_planar2_configure(FixturePlanar2 *arm,
                                                double proximal_m,
                                                double distal_m);
FixturePlanar2Status fixture_planar2_step(const FixturePlanar2 *arm,
                                           const double *joint_rad,
                                           size_t joint_count, double *x_m,
                                           double *y_m, double *yaw_rad);
FixturePlanar2Status fixture_planar2_shutdown(FixturePlanar2 *arm);
const char *fixture_planar2_version(void);

#endif
