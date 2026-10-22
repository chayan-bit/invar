#include "invar_process_bridge_v1.h"
#include "../../../examples/cells/reach_place/components/process_fixture/process_reference.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
  fprintf(stderr, "process bridge check failed at line %d\n", __LINE__); return 1; \
} } while (0)

struct inv_process_context_v1 { int configured, shutdown_status; };
static struct inv_process_context_v1 reference_context;
static int step_calls;

const char *invar_process_v1_version(void) { return "test-process-reference"; }
int32_t invar_process_v1_init(inv_process_context_v1 **out,
                               const uint8_t *path, uint32_t length) {
  if (!out || !path || !length || path[0] != '/') return INV_PROCESS_INVALID_INPUT;
  *out = &reference_context;
  reference_context.configured = 0;
  reference_context.shutdown_status = INV_PROCESS_OK;
  return INV_PROCESS_OK;
}
int32_t invar_process_v1_configure(inv_process_context_v1 *context,
                                    const inv_fk_config_v1 *config) {
  if (!context || context->configured) return INV_PROCESS_LIFECYCLE;
  if (!config || !config->model_xml_path || config->model_kind != 7)
    return INV_PROCESS_INVALID_INPUT;
  context->configured = 1;
  return INV_PROCESS_OK;
}
int32_t invar_process_v1_step(inv_process_context_v1 *context,
                               const double *q, uint32_t count,
                               inv_pose_f64 *out) {
  ++step_calls;
  if (!context || !context->configured) return INV_PROCESS_LIFECYCLE;
  if (count != 7) return INV_PROCESS_INVALID_INPUT;
  if (q[0] >= 3 && q[0] <= 6) {
    int status = (int)q[0];
    if (status == INV_PROCESS_CHILD_EXIT)
      context->shutdown_status = INV_PROCESS_CHILD_EXIT;
    return status;
  }
  *out = (inv_pose_f64){{q[0], q[1], q[2]}, {0, 0, 0, 1}};
  return INV_PROCESS_OK;
}
int32_t invar_process_v1_shutdown(inv_process_context_v1 **context) {
  if (!context || !*context) return INV_PROCESS_LIFECYCLE;
  int status = (*context)->shutdown_status;
  *context = NULL;
  return status;
}

int main(void) {
  invar_process_bridge_v1_context *context = NULL;
  invar_process_bridge_v1_inputs input = {{{1, 2, 3, 4, 5, 6, 7}}};
  invar_process_bridge_v1_outputs output = {{{11, 22, 33}, {4, 5, 6, 7}}};
  unsigned char before[sizeof output];
  memcpy(before, &output, sizeof output);
  const uint8_t child[] = "/fixture/child";
  const uint8_t model[] = "/models/fixture.xml";
  invar_process_bridge_v1_config config = {model, sizeof model - 1, 7};

  CHECK(strcmp(invar_process_bridge_v1_version(),
               "invar-component-v1/invar-process-reference-abi-1") == 0);
  CHECK(invar_process_bridge_v1_init(NULL, child, sizeof child - 1) == 1);
  CHECK(invar_process_bridge_v1_init(&context, NULL, 0) == 1);
  CHECK(invar_process_bridge_v1_init(&context, child, sizeof child - 1) == 0 && context);
  CHECK(invar_process_bridge_v1_init(&context, child, sizeof child - 1) == 2);
  CHECK(invar_process_bridge_v1_step(context, &input, &output) == 2);
  CHECK(memcmp(before, &output, sizeof output) == 0);
  config.model_kind = 6;
  CHECK(invar_process_bridge_v1_configure(context, &config) == 1);
  config.model_kind = 7;
  CHECK(invar_process_bridge_v1_configure(context, &config) == 0);
  CHECK(invar_process_bridge_v1_configure(context, &config) == 2);
  CHECK(invar_process_bridge_v1_step(context, NULL, &output) == 1);
  int calls_before = step_calls;
  input.q.values[0] = NAN;
  CHECK(invar_process_bridge_v1_step(context, &input, &output) == 1);
  CHECK(step_calls == calls_before);
  CHECK(memcmp(before, &output, sizeof output) == 0);
  for (int status = 3; status <= 6; ++status) {
    input.q.values[0] = status;
    CHECK(invar_process_bridge_v1_step(context, &input, &output) == status);
    CHECK(memcmp(before, &output, sizeof output) == 0);
  }
  input.q.values[0] = 1;
  CHECK(invar_process_bridge_v1_step(context, &input, &output) == 0);
  CHECK(output.pose.translation_m[0] == 1 && output.pose.translation_m[1] == 2 &&
        output.pose.quaternion_xyzw[3] == 1);
  CHECK(invar_process_bridge_v1_shutdown(&context) == 4 && !context);
  CHECK(invar_process_bridge_v1_shutdown(&context) == 2);
  return 0;
}
