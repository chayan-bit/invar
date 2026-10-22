#define _POSIX_C_SOURCE 200809L
#include "invar_foreign_v1.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint32_t calls = 0;
static uint32_t releases = 0;
uint32_t inv_abi_version_v1(void) { return 1; }

static int32_t scalar(inv_result_v1 *out, uint32_t tag, int64_t value) {
    ++calls;
    out->tag = tag;
    out->scalar = value;
    return 0;
}

int32_t c_count(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    (void)args; (void)argc;
    out->tag = INV_TAG_U32_V1; out->scalar = calls;
    return 0;
}

int32_t c_releases(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    (void)args; (void)argc;
    out->tag = INV_TAG_U32_V1; out->scalar = releases;
    return 0;
}

int32_t c_add(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    if (argc != 2 || args[0].tag != INV_TAG_I32_V1 || args[1].tag != INV_TAG_I32_V1) return 1;
    return scalar(out, INV_TAG_I32_V1, args[0].scalar + args[1].scalar);
}

int32_t c_bytes(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    if (argc != 1 || args[0].tag != INV_TAG_BYTES_V1 || args[0].mode != INV_MODE_READ_V1) return 1;
    uint32_t sum = 0;
    for (uint64_t i = 0; i < args[0].len; ++i) sum += args[0].data[i];
    return scalar(out, INV_TAG_U32_V1, sum);
}

int32_t c_address(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    if (argc != 3 || args[0].tag != INV_TAG_BYTES_V1) return 1;
    uintptr_t address = (uintptr_t)args[0].data;
    uint64_t expected = (uint64_t)args[1].scalar | ((uint64_t)args[2].scalar << 32);
    return scalar(out, INV_TAG_U32_V1, address == expected);
}

int32_t c_text_len(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    if (argc != 1 || args[0].tag != INV_TAG_STRING_V1) return 1;
    return scalar(out, INV_TAG_U32_V1, (int64_t)args[0].len);
}

int32_t c_hold(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    if (argc != 1 || args[0].tag != INV_TAG_BYTES_V1) return 1;
    struct timespec delay = { .tv_sec = 0, .tv_nsec = 150000000 };
    nanosleep(&delay, NULL);
    return c_bytes(args, argc, out);
}

static void release(uint8_t *data, uint64_t len) {
    (void)len; ++releases; free(data);
}

int32_t c_copy(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    if (argc != 1 || args[0].tag != INV_TAG_BYTES_V1) return 1;
    out->tag = INV_TAG_BYTES_V1;
    out->len = args[0].len;
    out->data = malloc(out->len ? (size_t)out->len : 1);
    if (!out->data) return 1;
    if (out->len) memcpy(out->data, args[0].data, (size_t)out->len);
    out->release = release;
    ++calls;
    return 0;
}

int32_t c_fail(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    (void)args; (void)argc;
    out->data = malloc(1); out->len = 1; out->release = release;
    ++calls; return 42;
}

int32_t c_bad_text(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    (void)args; (void)argc;
    out->data = malloc(1);
    if (!out->data) return 1;
    out->data[0] = 0xff; out->len = 1;
    out->tag = INV_TAG_STRING_V1; out->release = release;
    ++calls; return 0;
}

int32_t c_oversize(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    (void)args; (void)argc;
    out->data = malloc(1);
    if (!out->data) return 1;
    out->len = 1048577; out->tag = INV_TAG_BYTES_V1; out->release = release;
    ++calls; return 0;
}

int32_t c_bad_layout(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    (void)args; (void)argc; ++calls;
    out->tag = INV_TAG_BYTES_V1; out->len = 1; out->data = NULL; return 0;
}
