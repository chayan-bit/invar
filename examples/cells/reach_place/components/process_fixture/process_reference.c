#define _GNU_SOURCE
#include "process_reference.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

_Static_assert(sizeof(double) == 8 && DBL_MANT_DIG == 53 && FLT_RADIX == 2,
               "INVP v1 requires IEEE-754 binary64");
_Static_assert(sizeof(inv_pose_f64) == 56 && offsetof(inv_pose_f64, quaternion_xyzw) == 24,
               "reference pose layout mismatch");

enum { HEADER_LEN = 16, FRAME_MAX = 65536, PATH_MAX_BYTES = 4096,
       DEADLINE_MS = 2000, CHILD_REAP_MS = 200 };

struct inv_process_context_v1 {
  pid_t pid;
  int fd;
  uint32_t next_sequence;
  uint32_t model_kind;
  int state; /* 1: initialized, 2: configured, 0: child gone */
};

extern char **environ;

static uint16_t get_u16(const uint8_t *p) {
  return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t get_u32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void put_u16(uint8_t *p, uint16_t value) {
  p[0] = (uint8_t)value;
  p[1] = (uint8_t)(value >> 8);
}

static void put_u32(uint8_t *p, uint32_t value) {
  for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(value >> (8 * i));
}

static void put_f64(uint8_t *p, double value) {
  uint64_t bits;
  memcpy(&bits, &value, sizeof bits);
  for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(bits >> (8 * i));
}

static double get_f64(const uint8_t *p) {
  uint64_t bits = 0;
  for (int i = 0; i < 8; ++i) bits |= (uint64_t)p[i] << (8 * i);
  double value;
  memcpy(&value, &bits, sizeof value);
  return value;
}

static int valid_utf8(const uint8_t *bytes, uint32_t length) {
  if (!bytes || !length || length > PATH_MAX_BYTES) return 0;
  for (uint32_t i = 0; i < length;) {
    uint8_t first = bytes[i++];
    if (!first) return 0;
    if (first < 0x80) continue;
    uint32_t codepoint, minimum, remaining;
    if (first >= 0xc2 && first <= 0xdf) {
      codepoint = first & 0x1f; minimum = 0x80; remaining = 1;
    } else if (first >= 0xe0 && first <= 0xef) {
      codepoint = first & 0x0f; minimum = 0x800; remaining = 2;
    } else if (first >= 0xf0 && first <= 0xf4) {
      codepoint = first & 0x07; minimum = 0x10000; remaining = 3;
    } else {
      return 0;
    }
    if (length - i < remaining) return 0;
    for (uint32_t j = 0; j < remaining; ++j) {
      uint8_t next = bytes[i++];
      if ((next & 0xc0) != 0x80) return 0;
      codepoint = (codepoint << 6) | (next & 0x3f);
    }
    if (codepoint < minimum || (codepoint >= 0xd800 && codepoint <= 0xdfff) ||
        codepoint > 0x10ffff) return 0;
  }
  return 1;
}

static int64_t monotonic_ms(void) {
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return -1;
  return (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

static int wait_ready(int fd, short events, int64_t deadline) {
  for (;;) {
    int64_t now = monotonic_ms();
    if (now < 0) return INV_PROCESS_COMPONENT_ERROR;
    if (now >= deadline) return INV_PROCESS_TIMEOUT;
    struct pollfd poll_fd = {.fd = fd, .events = events, .revents = 0};
    int result = poll(&poll_fd, 1, (int)(deadline - now));
    if (result > 0) return INV_PROCESS_OK;
    if (result == 0) return INV_PROCESS_TIMEOUT;
    if (errno != EINTR) return INV_PROCESS_COMPONENT_ERROR;
  }
}

static int send_all(int fd, const uint8_t *data, size_t length, int64_t deadline) {
  size_t sent = 0;
  while (sent < length) {
    ssize_t n = send(fd, data + sent, length - sent, MSG_NOSIGNAL | MSG_DONTWAIT);
    if (n > 0) { sent += (size_t)n; continue; }
    if (n == 0 || errno == EPIPE || errno == ECONNRESET) return INV_PROCESS_CHILD_EXIT;
    if (errno == EINTR) continue;
    if (errno != EAGAIN && errno != EWOULDBLOCK) return INV_PROCESS_COMPONENT_ERROR;
    int status = wait_ready(fd, POLLOUT, deadline);
    if (status) return status;
  }
  return INV_PROCESS_OK;
}

static int recv_all(int fd, uint8_t *data, size_t length, int64_t deadline,
                    int empty_status) {
  size_t received = 0;
  while (received < length) {
    ssize_t n = recv(fd, data + received, length - received, MSG_DONTWAIT);
    if (n > 0) { received += (size_t)n; continue; }
    if (n == 0) return received ? INV_PROCESS_MALFORMED : empty_status;
    if (errno == EINTR) continue;
    if (errno == ECONNRESET) return received ? INV_PROCESS_MALFORMED : empty_status;
    if (errno != EAGAIN && errno != EWOULDBLOCK) return INV_PROCESS_COMPONENT_ERROR;
    int status = wait_ready(fd, POLLIN, deadline);
    if (status) return status;
  }
  return INV_PROCESS_OK;
}

static int reap_child(inv_process_context_v1 *context) {
  if (context->fd >= 0) { close(context->fd); context->fd = -1; }
  if (context->pid <= 0) { context->state = 0; return 0; }
  int status;
  pid_t waited;
  int forced = 0;
  int64_t deadline = monotonic_ms() + CHILD_REAP_MS;
  do { waited = waitpid(context->pid, &status, WNOHANG); }
  while (waited < 0 && errno == EINTR);
  while (waited == 0 && monotonic_ms() < deadline) {
    struct timespec pause = {.tv_sec = 0, .tv_nsec = 10000000};
    nanosleep(&pause, NULL);
    do { waited = waitpid(context->pid, &status, WNOHANG); }
    while (waited < 0 && errno == EINTR);
  }
  if (waited == 0) {
    forced = 1;
    kill(context->pid, SIGTERM);
    deadline = monotonic_ms() + CHILD_REAP_MS;
    do { waited = waitpid(context->pid, &status, WNOHANG); }
    while (waited < 0 && errno == EINTR);
    while (waited == 0 && monotonic_ms() < deadline) {
      struct timespec pause = {.tv_sec = 0, .tv_nsec = 10000000};
      nanosleep(&pause, NULL);
      do { waited = waitpid(context->pid, &status, WNOHANG); }
      while (waited < 0 && errno == EINTR);
    }
    if (waited == 0) {
      kill(context->pid, SIGKILL);
      do { waited = waitpid(context->pid, &status, 0); }
      while (waited < 0 && errno == EINTR);
    }
  }
  int exit_state = forced ? 1 :
      (waited == context->pid && (!WIFEXITED(status) || WEXITSTATUS(status) != 0) ? 2 : 0);
  context->pid = 0;
  context->state = 0;
  return exit_state;
}

static int exchange(inv_process_context_v1 *context, uint8_t op,
                    const uint8_t *body, uint32_t body_len,
                    uint8_t *reply_body, uint32_t expected_reply_len) {
  if (!context->next_sequence) return INV_PROCESS_COMPONENT_ERROR;
  uint32_t sequence = context->next_sequence++;
  uint8_t request[4 + HEADER_LEN + 8 + PATH_MAX_BYTES];
  uint32_t payload_len = HEADER_LEN + body_len;
  if (payload_len > FRAME_MAX || 4 + payload_len > sizeof request)
    return INV_PROCESS_INVALID_INPUT;
  put_u32(request, payload_len);
  memcpy(request + 4, "INVP", 4);
  put_u16(request + 8, 1);
  request[10] = op;
  request[11] = 0;
  put_u32(request + 12, sequence);
  put_u32(request + 16, 0);
  if (body_len) memcpy(request + 20, body, body_len);
  int64_t start = monotonic_ms();
  if (start < 0) return INV_PROCESS_COMPONENT_ERROR;
  int64_t deadline = start + DEADLINE_MS;
  int result = send_all(context->fd, request, 4 + payload_len, deadline);
  if (result) return result;
  uint8_t prefix[4];
  result = recv_all(context->fd, prefix, sizeof prefix, deadline, INV_PROCESS_CHILD_EXIT);
  if (result) return result;
  uint32_t length = get_u32(prefix);
  if (length < HEADER_LEN || length > FRAME_MAX) return INV_PROCESS_MALFORMED;
  uint8_t response[FRAME_MAX];
  result = recv_all(context->fd, response, length, deadline, INV_PROCESS_MALFORMED);
  if (result) return result;
  if (memcmp(response, "INVP", 4) || get_u16(response + 4) != 1 ||
      response[6] != op || response[7] != 1 ||
      get_u32(response + 8) != sequence) return INV_PROCESS_MALFORMED;
  uint32_t status = get_u32(response + 12);
  if (status > INV_PROCESS_COMPONENT_ERROR) return INV_PROCESS_MALFORMED;
  uint32_t actual_body_len = length - HEADER_LEN;
  if ((status && actual_body_len) || (!status && actual_body_len != expected_reply_len))
    return INV_PROCESS_MALFORMED;
  if (status) return (int)status;
  if (actual_body_len) memcpy(reply_body, response + HEADER_LEN, actual_body_len);
  return INV_PROCESS_OK;
}

static int fail_and_reap(inv_process_context_v1 *context, int status) {
  if (status >= INV_PROCESS_COMPONENT_ERROR) reap_child(context);
  return status;
}

const char *invar_process_v1_version(void) {
  return "invar-process-reference-abi-1";
}

int32_t invar_process_v1_init(inv_process_context_v1 **out_context,
                               const uint8_t *absolute_executable_path,
                               uint32_t path_len) {
  if (!out_context || !valid_utf8(absolute_executable_path, path_len) ||
      absolute_executable_path[0] != '/') return INV_PROCESS_INVALID_INPUT;
  if (*out_context) return INV_PROCESS_LIFECYCLE;
  char path[PATH_MAX_BYTES + 1];
  memcpy(path, absolute_executable_path, path_len);
  path[path_len] = '\0';
  struct stat info;
  if (stat(path, &info) || !S_ISREG(info.st_mode) || !(info.st_mode & 0111))
    return INV_PROCESS_INVALID_INPUT;

  inv_process_context_v1 *context = calloc(1, sizeof *context);
  if (!context) return INV_PROCESS_COMPONENT_ERROR;
  context->fd = -1;
  int sockets[2];
  if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets) != 0) {
    free(context); return INV_PROCESS_COMPONENT_ERROR;
  }
  if (sockets[1] == 3) {
    int duplicate = fcntl(sockets[1], F_DUPFD_CLOEXEC, 4);
    if (duplicate < 0) { close(sockets[0]); close(sockets[1]); free(context); return INV_PROCESS_COMPONENT_ERROR; }
    close(sockets[1]); sockets[1] = duplicate;
  }
  posix_spawn_file_actions_t actions;
  if (posix_spawn_file_actions_init(&actions)) {
    close(sockets[0]); close(sockets[1]); free(context); return INV_PROCESS_COMPONENT_ERROR;
  }
  int action_error = posix_spawn_file_actions_adddup2(&actions, sockets[1], 3);
  if (!action_error && sockets[0] != 3)
    action_error = posix_spawn_file_actions_addclose(&actions, sockets[0]);
  if (!action_error) action_error = posix_spawn_file_actions_addclose(&actions, sockets[1]);
  char *const argv[] = {path, NULL};
  int spawn_error = action_error ? action_error :
      posix_spawn(&context->pid, path, &actions, NULL, argv, environ);
  posix_spawn_file_actions_destroy(&actions);
  close(sockets[1]);
  if (spawn_error) { close(sockets[0]); free(context); return INV_PROCESS_COMPONENT_ERROR; }
  context->fd = sockets[0];
  context->next_sequence = 1;
  int status = exchange(context, 1, NULL, 0, NULL, 0);
  if (status) { reap_child(context); free(context); return status; }
  context->state = 1;
  *out_context = context;
  return INV_PROCESS_OK;
}

int32_t invar_process_v1_configure(inv_process_context_v1 *context,
                                    const inv_fk_config_v1 *config) {
  if (!context || context->state != 1) return INV_PROCESS_LIFECYCLE;
  if (!config || (config->model_kind != 6 && config->model_kind != 7) ||
      !valid_utf8(config->model_xml_path, config->model_xml_path_len))
    return INV_PROCESS_INVALID_INPUT;
  uint8_t body[8 + PATH_MAX_BYTES];
  put_u32(body, config->model_kind);
  put_u32(body + 4, config->model_xml_path_len);
  memcpy(body + 8, config->model_xml_path, config->model_xml_path_len);
  int status = exchange(context, 2, body, 8 + config->model_xml_path_len, NULL, 0);
  if (status) return fail_and_reap(context, status);
  context->model_kind = config->model_kind;
  context->state = 2;
  return INV_PROCESS_OK;
}

int32_t invar_process_v1_step(inv_process_context_v1 *context,
                               const double *joint_rad, uint32_t joint_count,
                               inv_pose_f64 *out_pose) {
  if (!context || context->state != 2) return INV_PROCESS_LIFECYCLE;
  if (!joint_rad || !out_pose || joint_count != context->model_kind)
    return INV_PROCESS_INVALID_INPUT;
  for (uint32_t i = 0; i < joint_count; ++i)
    if (!isfinite(joint_rad[i])) return INV_PROCESS_INVALID_INPUT;
  uint8_t body[4 + 7 * 8];
  put_u32(body, joint_count);
  for (uint32_t i = 0; i < joint_count; ++i) put_f64(body + 4 + 8 * i, joint_rad[i]);
  uint8_t reply[56];
  int status = exchange(context, 3, body, 4 + 8 * joint_count, reply, sizeof reply);
  if (status) return fail_and_reap(context, status);
  inv_pose_f64 candidate;
  for (int i = 0; i < 3; ++i) candidate.translation_m[i] = get_f64(reply + 8 * i);
  double norm = 0;
  for (int i = 0; i < 4; ++i) {
    candidate.quaternion_xyzw[i] = get_f64(reply + 24 + 8 * i);
    norm += candidate.quaternion_xyzw[i] * candidate.quaternion_xyzw[i];
  }
  for (int i = 0; i < 3; ++i)
    if (!isfinite(candidate.translation_m[i])) return fail_and_reap(context, INV_PROCESS_MALFORMED);
  if (!isfinite(norm) || fabs(sqrt(norm) - 1.0) > 1e-12)
    return fail_and_reap(context, INV_PROCESS_MALFORMED);
  *out_pose = candidate;
  return INV_PROCESS_OK;
}

int32_t invar_process_v1_shutdown(inv_process_context_v1 **context) {
  if (!context || !*context) return INV_PROCESS_LIFECYCLE;
  inv_process_context_v1 *owned = *context;
  int status = owned->state ? exchange(owned, 4, NULL, 0, NULL, 0)
                            : INV_PROCESS_CHILD_EXIT;
  int forced = reap_child(owned);
  free(owned);
  *context = NULL;
  if (status != INV_PROCESS_OK) return status;
  return forced == 1 ? INV_PROCESS_TIMEOUT :
         forced == 2 ? INV_PROCESS_CHILD_EXIT : INV_PROCESS_OK;
}
