#include "invar_foreign_v1.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fenv.h>

static uint32_t release_calls;

__attribute__((constructor)) static void on_load(void) {
    const char *marker = getenv("INVAR_L104_CONSTRUCTOR_MARKER");
    if (!marker) return;
    FILE *file = fopen(marker, "ab");
    if (file) { fputc('1', file); fclose(file); }
}

static void release_buffer(uint8_t *data, uint64_t len) {
    (void)len;
    ++release_calls;
    free(data);
}

uint32_t inv_abi_version_v1(void) {
#ifdef INV_BAD_VERSION
    return 2u;
#else
    return INV_ABI_VERSION_V1;
#endif
}
uint32_t inv_release_count_v1(void) { return release_calls; }

int32_t c_add(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    if (argc != 2 || args[0].tag != INV_TAG_I32_V1 || args[1].tag != INV_TAG_I32_V1 ||
        args[0].mode != INV_MODE_VALUE_V1 || args[1].mode != INV_MODE_VALUE_V1) return 11;
    out->tag = INV_TAG_I32_V1;
    out->scalar = args[0].scalar + args[1].scalar;
    return 0;
}

int32_t c_f64_trunc(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    if (argc != 1 || args[0].tag != INV_TAG_F64_V1 || args[0].mode != INV_MODE_VALUE_V1) return 19;
    double value;
    memcpy(&value, &args[0].scalar, sizeof value);
    out->tag = INV_TAG_I32_V1;
    out->scalar = (int64_t)value;
    return 0;
}

/* Leaves the process in the upward rounding mode, as a careless library might. */
int32_t c_round_upward(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    if (argc != 1 || args[0].tag != INV_TAG_I32_V1 || args[0].mode != INV_MODE_VALUE_V1) return 21;
    if (fesetround(FE_UPWARD) != 0) return 22;
    out->tag = INV_TAG_I32_V1;
    out->scalar = args[0].scalar;
    return 0;
}

int32_t c_copy_take(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    if (argc != 1 || args[0].tag != INV_TAG_BYTES_V1 || args[0].mode != INV_MODE_TAKE_V1) return 12;
    uint8_t *data = malloc(args[0].len ? (size_t)args[0].len : 1u);
    if (!data) return 13;
    if (args[0].len) memcpy(data, args[0].data, (size_t)args[0].len);
    out->tag = INV_TAG_BYTES_V1;
    out->data = data;
    out->len = args[0].len;
    out->release = release_buffer;
    return 0;
}

int32_t c_copy_read(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    if (argc != 1 || args[0].tag != INV_TAG_BYTES_V1 || args[0].mode != INV_MODE_READ_V1) return 14;
    out->tag = INV_TAG_U32_V1;
    out->scalar = (int64_t)args[0].len;
    return 0;
}

int32_t c_text(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    (void)args;
    if (argc != 0) return 16;
    uint8_t *data = malloc(2);
    if (!data) return 13;
    data[0] = 0xc3;
    data[1] = 0xa9;
    out->tag = INV_TAG_STRING_V1;
    out->data = data;
    out->len = 2;
    out->release = release_buffer;
    return 0;
}

int32_t c_status(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    (void)args;
    if (argc != 0) return 15;
    out->tag = INV_TAG_BYTES_V1;
    out->data = malloc(1);
    out->len = 1;
    out->release = release_buffer;
    return 42;
}

int32_t c_wrong_tag(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    (void)args; (void)argc;
    out->tag = INV_TAG_STRING_V1;
    return 0;
}

int32_t c_bad_utf8(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    (void)args; (void)argc;
    out->tag = INV_TAG_STRING_V1;
    out->data = malloc(1);
    if (!out->data) return 13;
    out->data[0] = 0xff;
    out->len = 1;
    out->release = release_buffer;
    return 0;
}

int32_t c_overflow_len(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    (void)args; (void)argc;
    out->tag = INV_TAG_BYTES_V1;
    out->data = malloc(1);
    if (!out->data) return 13;
    out->len = UINT64_MAX;
    out->release = release_buffer;
    return 0;
}

int32_t c_borrowed_return(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    (void)args; (void)argc;
    static uint8_t borrowed = 7;
    out->tag = INV_TAG_BYTES_V1;
    out->data = &borrowed;
    out->len = 1;
    return 0;
}

int32_t c_null_data(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out) {
    (void)args; (void)argc;
    out->tag = INV_TAG_BYTES_V1;
    out->len = 1;
    return 0;
}
