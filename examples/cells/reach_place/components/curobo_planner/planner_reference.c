#define _GNU_SOURCE
#include "planner_reference.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stddef.h>
#include <pthread.h>
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

enum { HEADER_LEN = 16, FRAME_MAX = 65536, PATH_MAX_BYTES = 4096,
       DEADLINE_MS = 5000, CONFIGURE_DEADLINE_MS = 120000, CHILD_REAP_MS = 200,
       /* Exit allowance after an acknowledged shutdown. A child holding a CUDA context
          spends 150-190 ms in exit_group on the qualification host (driver teardown,
          measured with strace), so 200 ms forced live GPU children; the ROS task
          process supervisor allows 2 s as well. Failure paths keep CHILD_REAP_MS. */
       CHILD_EXIT_MS = 2000 };

struct inv_planner_context_v1 {
  pid_t pid;
  int fd;
  uint32_t next_sequence;
  uint32_t model_kind;
  int state; /* 1: initialized, 2: configured, 0: child gone */
  double limits[7][2];
  pthread_mutex_t mutex;
  pthread_cond_t idle;
  int active, cancelled;
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

/* Pinned Panda/UR5e URDF velocity limits; robot_kind is the active joint count. */
static double velocity_limit(uint32_t robot_kind, uint32_t joint) {
  return robot_kind == 7 ? (joint < 4 ? 2.175 : 2.61)
                         : (joint < 3 ? 3.14 : 6.28);
}

static int valid_utf8(const uint8_t *bytes, uint32_t length, uint32_t maximum) {
  if (!bytes || !length || length > maximum) return 0;
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
    if (now < 0) return INV_PLANNER_FAILED;
    if (now >= deadline) return INV_PLANNER_TIMEOUT;
    struct pollfd poll_fd = {.fd = fd, .events = events, .revents = 0};
    int result = poll(&poll_fd, 1, (int)(deadline - now));
    if (result > 0) return INV_PLANNER_OK;
    if (result == 0) return INV_PLANNER_TIMEOUT;
    if (errno != EINTR) return INV_PLANNER_FAILED;
  }
}

static int send_all(int fd, const uint8_t *data, size_t length, int64_t deadline) {
  size_t sent = 0;
  while (sent < length) {
    ssize_t n = send(fd, data + sent, length - sent, MSG_NOSIGNAL | MSG_DONTWAIT);
    if (n > 0) { sent += (size_t)n; continue; }
    if (n == 0 || errno == EPIPE || errno == ECONNRESET) return INV_PLANNER_CHILD_EXIT;
    if (errno == EINTR) continue;
    if (errno != EAGAIN && errno != EWOULDBLOCK) return INV_PLANNER_FAILED;
    int status = wait_ready(fd, POLLOUT, deadline);
    if (status) return status;
  }
  return INV_PLANNER_OK;
}

static int recv_all(int fd, uint8_t *data, size_t length, int64_t deadline,
                    int empty_status) {
  size_t received = 0;
  while (received < length) {
    ssize_t n = recv(fd, data + received, length - received, MSG_DONTWAIT);
    if (n > 0) { received += (size_t)n; continue; }
    if (n == 0) return received ? INV_PLANNER_MALFORMED : empty_status;
    if (errno == EINTR) continue;
    if (errno == ECONNRESET) return received ? INV_PLANNER_MALFORMED : empty_status;
    if (errno != EAGAIN && errno != EWOULDBLOCK) return INV_PLANNER_FAILED;
    int status = wait_ready(fd, POLLIN, deadline);
    if (status) return status;
  }
  return INV_PLANNER_OK;
}

static int reap_child(inv_planner_context_v1 *context, int64_t grace_ms) {
  if (context->fd >= 0) { close(context->fd); context->fd = -1; }
  if (context->pid <= 0) { context->state = 0; return 0; }
  int status;
  pid_t waited;
  int forced = 0;
  int64_t deadline = monotonic_ms() + grace_ms;
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

static int exchange(inv_planner_context_v1 *context, uint8_t op,
                    const uint8_t *body, uint32_t body_len,
                    uint8_t *reply_body, uint32_t reply_capacity,
                    uint32_t *reply_len) {
  if (!context->next_sequence) return INV_PLANNER_FAILED;
  uint32_t sequence = context->next_sequence++;
  uint8_t request[4 + FRAME_MAX];
  uint32_t payload_len = HEADER_LEN + body_len;
  if (payload_len > FRAME_MAX || 4 + payload_len > sizeof request)
    return INV_PLANNER_INVALID;
  put_u32(request, payload_len);
  memcpy(request + 4, "INVP", 4);
  put_u16(request + 8, 3);
  request[10] = op;
  request[11] = 0;
  put_u32(request + 12, sequence);
  put_u32(request + 16, 0);
  if (body_len) memcpy(request + 20, body, body_len);
  int64_t start = monotonic_ms();
  if (start < 0) return INV_PLANNER_FAILED;
  int64_t deadline = start + (op == 2 ? CONFIGURE_DEADLINE_MS : DEADLINE_MS);
  int result = send_all(context->fd, request, 4 + payload_len, deadline);
  if (result) return result;
  uint8_t prefix[4];
  result = recv_all(context->fd, prefix, sizeof prefix, deadline, INV_PLANNER_CHILD_EXIT);
  if (result) return result;
  uint32_t length = get_u32(prefix);
  if (length < HEADER_LEN || length > FRAME_MAX) return INV_PLANNER_MALFORMED;
  uint8_t response[FRAME_MAX];
  result = recv_all(context->fd, response, length, deadline, INV_PLANNER_MALFORMED);
  if (result) return result;
  if (memcmp(response, "INVP", 4) || get_u16(response + 4) != 3 ||
      response[6] != op || response[7] != 1 ||
      get_u32(response + 8) != sequence) return INV_PLANNER_MALFORMED;
  uint32_t status = get_u32(response + 12);
  if (status > INV_PLANNER_CANCELLED) return INV_PLANNER_MALFORMED;
  uint32_t actual_body_len = length - HEADER_LEN;
  if ((status && actual_body_len) ||
      (!status && (actual_body_len > reply_capacity ||
                   (!reply_len && actual_body_len != reply_capacity))))
    return INV_PLANNER_MALFORMED;
  if (status) return (int)status;
  if (actual_body_len) memcpy(reply_body, response + HEADER_LEN, actual_body_len);
  if (reply_len) *reply_len = actual_body_len;
  return INV_PLANNER_OK;
}

static int fail_and_reap(inv_planner_context_v1 *context, int status) {
  if (status >= INV_PLANNER_FAILED) reap_child(context, CHILD_REAP_MS);
  return status;
}

static int valid_pose(const inv_planner_pose_v1 *pose) {
  double norm = 0;
  for (int i = 0; i < 3; ++i)
    if (!isfinite(pose->xyz_m[i])) return 0;
  for (int i = 0; i < 4; ++i) {
    if (!isfinite(pose->quaternion_xyzw[i])) return 0;
    norm += pose->quaternion_xyzw[i] * pose->quaternion_xyzw[i];
  }
  return isfinite(norm) && fabs(sqrt(norm) - 1.0) <= 1e-12;
}

static void put_pose(uint8_t *out, const inv_planner_pose_v1 *pose) {
  for (int i = 0; i < 3; ++i) put_f64(out + 8 * i, pose->xyz_m[i]);
  for (int i = 0; i < 4; ++i) put_f64(out + 24 + 8 * i, pose->quaternion_xyzw[i]);
}

static int valid_text(inv_planner_text_v1 text, uint32_t maximum, int absolute) {
  return valid_utf8(text.bytes, text.length, maximum) &&
         (!absolute || text.bytes[0] == '/');
}

static uint32_t put_text(uint8_t *out, uint32_t offset, inv_planner_text_v1 text) {
  put_u16(out + offset, text.length);
  memcpy(out + offset + 2, text.bytes, text.length);
  return offset + 2 + text.length;
}

const char *invar_planner_v1_version(void) {
  return "invar-curobo-planner-reference-abi-1";
}

int32_t invar_planner_v1_init(inv_planner_context_v1 **out_context,
                               const uint8_t *absolute_executable_path,
                               uint32_t path_len) {
  if (!out_context || !valid_utf8(absolute_executable_path, path_len, PATH_MAX_BYTES) ||
      absolute_executable_path[0] != '/') return INV_PLANNER_INVALID;
  if (*out_context) return INV_PLANNER_LIFECYCLE;
  char path[PATH_MAX_BYTES + 1];
  memcpy(path, absolute_executable_path, path_len);
  path[path_len] = '\0';
  struct stat info;
  if (stat(path, &info) || !S_ISREG(info.st_mode) || !(info.st_mode & 0111))
    return INV_PLANNER_INVALID;

  inv_planner_context_v1 *context = calloc(1, sizeof *context);
  if (!context) return INV_PLANNER_FAILED;
  context->fd = -1;
  if (pthread_mutex_init(&context->mutex, NULL)) { free(context); return INV_PLANNER_FAILED; }
  if (pthread_cond_init(&context->idle, NULL)) {
    pthread_mutex_destroy(&context->mutex); free(context); return INV_PLANNER_FAILED;
  }
  int sockets[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
    pthread_cond_destroy(&context->idle); pthread_mutex_destroy(&context->mutex);
    free(context); return INV_PLANNER_FAILED;
  }
  if (fcntl(sockets[0], F_SETFD, FD_CLOEXEC) < 0 ||
      fcntl(sockets[1], F_SETFD, FD_CLOEXEC) < 0) {
    close(sockets[0]); close(sockets[1]);
    pthread_cond_destroy(&context->idle); pthread_mutex_destroy(&context->mutex);
    free(context); return INV_PLANNER_FAILED;
  }
  if (sockets[1] == 3) {
    int duplicate = fcntl(sockets[1], F_DUPFD_CLOEXEC, 4);
    if (duplicate < 0) {
      close(sockets[0]); close(sockets[1]);
      pthread_cond_destroy(&context->idle); pthread_mutex_destroy(&context->mutex);
      free(context); return INV_PLANNER_FAILED;
    }
    close(sockets[1]); sockets[1] = duplicate;
  }
  posix_spawn_file_actions_t actions;
  if (posix_spawn_file_actions_init(&actions)) {
    close(sockets[0]); close(sockets[1]);
    pthread_cond_destroy(&context->idle); pthread_mutex_destroy(&context->mutex);
    free(context); return INV_PLANNER_FAILED;
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
  if (spawn_error) {
    close(sockets[0]); pthread_cond_destroy(&context->idle);
    pthread_mutex_destroy(&context->mutex); free(context); return INV_PLANNER_FAILED;
  }
  context->fd = sockets[0];
  context->next_sequence = 1;
  int status = exchange(context, 1, NULL, 0, NULL, 0, NULL);
  if (status) {
    reap_child(context, CHILD_REAP_MS); pthread_cond_destroy(&context->idle);
    pthread_mutex_destroy(&context->mutex); free(context); return status;
  }
  context->state = 1;
  *out_context = context;
  return INV_PLANNER_OK;
}

int32_t invar_planner_v1_configure(inv_planner_context_v1 *context,
                                    const inv_planner_config_v1 *config) {
  if (!context || context->state != 1) return INV_PLANNER_LIFECYCLE;
  if (!config || config->robot_kind != config->joint_count ||
      (config->joint_count != 6 && config->joint_count != 7) ||
      !valid_pose(&config->table_pose) || !isfinite(config->cube_edge_m) ||
      config->cube_edge_m != 0.05 ||
      !valid_text(config->base_frame, 64, 0) ||
      !valid_text(config->tool_frame, 64, 0) ||
      !valid_text(config->robot_config_path, PATH_MAX_BYTES, 1) ||
      !valid_text(config->scene_config_path, PATH_MAX_BYTES, 1)) return INV_PLANNER_INVALID;
  for (uint32_t i = 0; i < 3; ++i)
    if (!isfinite(config->table_half_extents_m[i]) ||
        config->table_half_extents_m[i] <= 0) return INV_PLANNER_INVALID;
  for (uint32_t i = 0; i < config->joint_count; ++i)
    if (!valid_text(config->joint_names[i], 64, 0) ||
        !isfinite(config->joint_limits_rad[i][0]) ||
        !isfinite(config->joint_limits_rad[i][1]) ||
        config->joint_limits_rad[i][0] >= config->joint_limits_rad[i][1])
      return INV_PLANNER_INVALID;
  uint8_t body[FRAME_MAX - HEADER_LEN];
  uint32_t offset = 0;
  put_u32(body + offset, config->robot_kind); offset += 4;
  put_u32(body + offset, config->joint_count); offset += 4;
  memcpy(body + offset, config->model_sha256, 32); offset += 32;
  memcpy(body + offset, config->robot_config_sha256, 32); offset += 32;
  memcpy(body + offset, config->scene_sha256, 32); offset += 32;
  for (uint32_t i = 0; i < config->joint_count; ++i)
    for (int k = 0; k < 2; ++k) {
      put_f64(body + offset, config->joint_limits_rad[i][k]); offset += 8;
    }
  put_pose(body + offset, &config->table_pose); offset += 56;
  for (int i = 0; i < 3; ++i) {
    put_f64(body + offset, config->table_half_extents_m[i]); offset += 8;
  }
  put_f64(body + offset, config->cube_edge_m); offset += 8;
  offset = put_text(body, offset, config->base_frame);
  offset = put_text(body, offset, config->tool_frame);
  for (uint32_t i = 0; i < config->joint_count; ++i)
    offset = put_text(body, offset, config->joint_names[i]);
  offset = put_text(body, offset, config->robot_config_path);
  offset = put_text(body, offset, config->scene_config_path);
  if (offset > sizeof body) return INV_PLANNER_INVALID;
  int status = exchange(context, 2, body, offset, NULL, 0, NULL);
  if (status) return fail_and_reap(context, status);
  context->model_kind = config->joint_count;
  for (uint32_t i = 0; i < config->joint_count; ++i)
    memcpy(context->limits[i], config->joint_limits_rad[i], sizeof context->limits[i]);
  context->state = 2;
  return INV_PLANNER_OK;
}

static int inside_limits(const inv_planner_context_v1 *context, uint32_t i, double q) {
  return isfinite(q) && q >= context->limits[i][0] && q <= context->limits[i][1];
}

static int valid_request(const inv_planner_context_v1 *context,
                         const inv_planner_request_v2 *request) {
  if (!request || request->joint_count != context->model_kind ||
      request->phase > INV_PLANNER_PHASE_HOME || !valid_pose(&request->target_base_from_tool) ||
      !valid_pose(&request->cube_base_pose) || !valid_pose(&request->tool_from_cube))
    return 0;
  for (uint32_t i = 0; i < request->joint_count; ++i)
    if (!inside_limits(context, i, request->start_q_rad[i]) ||
        !inside_limits(context, i, request->goal_q_rad[i]) ||
        (request->phase != INV_PLANNER_PHASE_HOME &&
         memcmp(&request->start_q_rad[i], &request->goal_q_rad[i], sizeof(double))))
      return 0;
  return 1;
}

int32_t invar_planner_v1_plan(inv_planner_context_v1 *context,
                               const inv_planner_request_v2 *request,
                               inv_planner_result_v1 *out_result) {
  if (!context || context->state != 2) return INV_PLANNER_LIFECYCLE;
  if (!out_result || !valid_request(context, request)) return INV_PLANNER_INVALID;
  pthread_mutex_lock(&context->mutex);
  if (context->active) { pthread_mutex_unlock(&context->mutex); return INV_PLANNER_LIFECYCLE; }
  context->active = 1;
  context->cancelled = 0;
  pthread_mutex_unlock(&context->mutex);

  uint8_t body[4 + 2 * 7 * 8 + 3 * 56 + 4];
  uint32_t offset = 0;
  put_u32(body + offset, request->joint_count); offset += 4;
  for (uint32_t i = 0; i < request->joint_count; ++i) {
    put_f64(body + offset, request->start_q_rad[i]); offset += 8;
  }
  for (uint32_t i = 0; i < request->joint_count; ++i) {
    put_f64(body + offset, request->goal_q_rad[i]); offset += 8;
  }
  put_pose(body + offset, &request->target_base_from_tool); offset += 56;
  put_pose(body + offset, &request->cube_base_pose); offset += 56;
  put_u32(body + offset, request->phase); offset += 4;
  put_pose(body + offset, &request->tool_from_cube); offset += 56;
  uint8_t reply[FRAME_MAX - HEADER_LEN];
  uint32_t reply_len = 0;
  int status = exchange(context, 3, body, offset, reply, sizeof reply, &reply_len);
  inv_planner_result_v1 candidate = {0};
  if (!status) {
    if (reply_len < 12) status = INV_PLANNER_MALFORMED;
    else {
      candidate.dt_s = get_f64(reply);
      candidate.count = get_u32(reply + 8);
      uint32_t count = candidate.count, n = request->joint_count;
      uint32_t stride = 8 * (1 + 2 * n);
      if (!isfinite(candidate.dt_s) || candidate.dt_s <= 0 ||
          count < 2 || count > INV_PLANNER_MAX_WAYPOINTS ||
          reply_len != 12 + count * stride) status = INV_PLANNER_MALFORMED;
      else {
        double previous = -1;
        uint32_t cursor = 12;
        for (uint32_t k = 0; k < count && !status; ++k) {
          inv_planner_waypoint_v1 *waypoint = &candidate.waypoints[k];
          waypoint->time_from_start_s = get_f64(reply + cursor); cursor += 8;
          if (!isfinite(waypoint->time_from_start_s) ||
              waypoint->time_from_start_s < 0 || waypoint->time_from_start_s <= previous)
            status = INV_PLANNER_MALFORMED;
          previous = waypoint->time_from_start_s;
          for (uint32_t i = 0; i < n; ++i) {
            waypoint->q_rad[i] = get_f64(reply + cursor); cursor += 8;
            if (!isfinite(waypoint->q_rad[i]) ||
                waypoint->q_rad[i] < context->limits[i][0] ||
                waypoint->q_rad[i] > context->limits[i][1] ||
                (k == 0 && fabs(waypoint->q_rad[i] - request->start_q_rad[i]) > 1e-6))
              status = INV_PLANNER_MALFORMED;
          }
          for (uint32_t i = 0; i < n; ++i) {
            waypoint->dq_rad_s[i] = get_f64(reply + cursor); cursor += 8;
            if (!isfinite(waypoint->dq_rad_s[i]) ||
                fabs(waypoint->dq_rad_s[i]) > velocity_limit(n, i))
              status = INV_PLANNER_MALFORMED;
          }
        }
      }
    }
  }
  pthread_mutex_lock(&context->mutex);
  if (context->cancelled) status = INV_PLANNER_CANCELLED;
  if (status >= INV_PLANNER_FAILED) reap_child(context, CHILD_REAP_MS);
  context->active = 0;
  pthread_cond_broadcast(&context->idle);
  pthread_mutex_unlock(&context->mutex);
  if (!status) *out_result = candidate;
  return status;
}

int32_t invar_planner_v1_cancel(inv_planner_context_v1 *context) {
  if (!context) return INV_PLANNER_LIFECYCLE;
  pthread_mutex_lock(&context->mutex);
  if (!context->active || context->pid <= 0) {
    pthread_mutex_unlock(&context->mutex); return INV_PLANNER_LIFECYCLE;
  }
  context->cancelled = 1;
  kill(context->pid, SIGTERM);
  while (context->active) pthread_cond_wait(&context->idle, &context->mutex);
  pthread_mutex_unlock(&context->mutex);
  return INV_PLANNER_CANCELLED;
}

int32_t invar_planner_v1_shutdown(inv_planner_context_v1 **context) {
  if (!context || !*context) return INV_PLANNER_LIFECYCLE;
  inv_planner_context_v1 *owned = *context;
  pthread_mutex_lock(&owned->mutex);
  if (owned->active) { pthread_mutex_unlock(&owned->mutex); return INV_PLANNER_LIFECYCLE; }
  pthread_mutex_unlock(&owned->mutex);
  int status = owned->state ? exchange(owned, 4, NULL, 0, NULL, 0, NULL)
                            : INV_PLANNER_CHILD_EXIT;
  int forced = reap_child(owned, status == INV_PLANNER_OK ? CHILD_EXIT_MS : CHILD_REAP_MS);
  pthread_cond_destroy(&owned->idle);
  pthread_mutex_destroy(&owned->mutex);
  free(owned);
  *context = NULL;
  if (status != INV_PLANNER_OK) return status;
  return forced == 1 ? INV_PLANNER_TIMEOUT :
         forced == 2 ? INV_PLANNER_CHILD_EXIT : INV_PLANNER_OK;
}
