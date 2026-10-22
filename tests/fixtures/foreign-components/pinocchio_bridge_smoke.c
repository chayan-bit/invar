#include "invar_pinocchio_bridge_v1.h"
#include "../../../examples/cells/reach_place/components/pinocchio_fk/abi_reference.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
  fprintf(stderr, "bridge check failed at line %d\n", __LINE__); return 1; \
} } while (0)

struct inv_fk_context_v1 { int configured; };
static struct inv_fk_context_v1 reference_context;
static int step_calls;

const char *invar_pinocchio_v1_version(void) { return "test-reference"; }
int32_t invar_pinocchio_v1_init(inv_fk_context_v1 **out) {
  if (!out) return INV_FK_BAD_ARGUMENT;
  *out = &reference_context;
  reference_context.configured = 0;
  return INV_FK_OK;
}
int32_t invar_pinocchio_v1_configure(inv_fk_context_v1 *context,
                                     const inv_fk_config_v1 *config) {
  if (!context || context->configured) return INV_FK_BAD_STATE;
  if (!config || !config->model_xml_path || config->model_kind != 7)
    return INV_FK_BAD_ARGUMENT;
  context->configured = 1;
  return INV_FK_OK;
}
int32_t invar_pinocchio_v1_step(inv_fk_context_v1 *context,
                                const double *q, uint32_t count,
                                inv_pose_f64 *out) {
  ++step_calls;
  if (!context || !context->configured) return INV_FK_BAD_STATE;
  if (count != 7) return INV_FK_BAD_COUNT;
  if (!isfinite(q[0])) return INV_FK_NONFINITE;
  if (q[0] == 999.0) return INV_FK_LIBRARY_FAILURE;
  *out = (inv_pose_f64){{q[0], q[1], q[2]}, {0, 0, 0, 1}};
  return INV_FK_OK;
}
int32_t invar_pinocchio_v1_shutdown(inv_fk_context_v1 **context) {
  if (!context || !*context) return INV_FK_BAD_STATE;
  *context = NULL;
  return INV_FK_OK;
}

int main(void) {
  invar_pinocchio_bridge_v1_context *context = NULL;
  invar_pinocchio_bridge_v1_inputs input = {{{1, 2, 3, 4, 5, 6, 7}}};
  invar_pinocchio_bridge_v1_outputs output = {{{11, 22, 33}, {4, 5, 6, 7}}};
  unsigned char before[sizeof output];
  memcpy(before, &output, sizeof output);
  const uint8_t path[] = "model.xml";
  invar_pinocchio_bridge_v1_config config = {path, sizeof path - 1, 7};

  CHECK(strcmp(invar_pinocchio_bridge_v1_version(),
               "invar-component-v1/invar-pinocchio-reference-abi-1/pinocchio-4.1.0") == 0);
  CHECK(invar_pinocchio_bridge_v1_init(NULL, NULL, 0) == 1);
  CHECK(invar_pinocchio_bridge_v1_init(&context, path, sizeof path - 1) == 1);
  CHECK(invar_pinocchio_bridge_v1_init(&context, NULL, 0) == 0 && context);
  CHECK(invar_pinocchio_bridge_v1_init(&context, NULL, 0) == 2);
  CHECK(invar_pinocchio_bridge_v1_step(context, &input, &output) == 2);
  CHECK(memcmp(before, &output, sizeof output) == 0);
  config.model_kind = 6;
  CHECK(invar_pinocchio_bridge_v1_configure(context, &config) == 1);
  config.model_kind = 7;
  CHECK(invar_pinocchio_bridge_v1_configure(context, &config) == 0);
  CHECK(invar_pinocchio_bridge_v1_configure(context, &config) == 2);
  CHECK(invar_pinocchio_bridge_v1_step(context, NULL, &output) == 1);
  input.q.values[0] = NAN;
  int calls_before = step_calls;
  CHECK(invar_pinocchio_bridge_v1_step(context, &input, &output) == 1);
  CHECK(step_calls == calls_before);
  CHECK(memcmp(before, &output, sizeof output) == 0);
  input.q.values[0] = 999;
  CHECK(invar_pinocchio_bridge_v1_step(context, &input, &output) == 3);
  CHECK(memcmp(before, &output, sizeof output) == 0);
  input.q.values[0] = 1;
  CHECK(invar_pinocchio_bridge_v1_step(context, &input, &output) == 0);
  CHECK(output.pose.translation_m[0] == 1 && output.pose.translation_m[1] == 2 &&
        output.pose.quaternion_xyzw[3] == 1);
  CHECK(invar_pinocchio_bridge_v1_shutdown(&context) == 0 && !context);
  CHECK(invar_pinocchio_bridge_v1_shutdown(&context) == 2);
  return 0;
}
