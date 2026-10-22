#ifndef INVAR_FOREIGN_V1_H
#define INVAR_FOREIGN_V1_H

#include <stddef.h>
#include <stdint.h>

#define INV_ABI_VERSION_V1 1u
#define INV_TAG_I32_V1 1u
#define INV_TAG_U32_V1 2u
#define INV_TAG_BYTES_V1 3u
#define INV_TAG_STRING_V1 4u
/* Value argument whose scalar holds the binary64 encoding (memcpy to double). */
#define INV_TAG_F64_V1 5u
#define INV_MODE_VALUE_V1 0u
#define INV_MODE_READ_V1 1u
#define INV_MODE_TAKE_V1 2u

typedef struct inv_arg_v1 {
    uint32_t tag;
    uint32_t mode;
    int64_t scalar;
    const uint8_t *data;
    uint64_t len;
} inv_arg_v1;

typedef void (*inv_release_v1)(uint8_t *data, uint64_t len);

typedef struct inv_result_v1 {
    uint32_t abi_version;
    uint32_t tag;
    int64_t scalar;
    uint8_t *data;
    uint64_t len;
    inv_release_v1 release;
} inv_result_v1;

typedef int32_t (*inv_call_v1)(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out);

/* Every selected library exports this exact version symbol. */
uint32_t inv_abi_version_v1(void);

_Static_assert(sizeof(inv_arg_v1) == 32, "inv_arg_v1 size");
_Static_assert(_Alignof(inv_arg_v1) == 8, "inv_arg_v1 alignment");
_Static_assert(offsetof(inv_arg_v1, tag) == 0, "arg tag offset");
_Static_assert(offsetof(inv_arg_v1, mode) == 4, "arg mode offset");
_Static_assert(offsetof(inv_arg_v1, scalar) == 8, "arg scalar offset");
_Static_assert(offsetof(inv_arg_v1, data) == 16, "arg data offset");
_Static_assert(offsetof(inv_arg_v1, len) == 24, "arg length offset");
_Static_assert(sizeof(inv_result_v1) == 40, "inv_result_v1 size");
_Static_assert(_Alignof(inv_result_v1) == 8, "inv_result_v1 alignment");
_Static_assert(offsetof(inv_result_v1, abi_version) == 0, "result version offset");
_Static_assert(offsetof(inv_result_v1, tag) == 4, "result tag offset");
_Static_assert(offsetof(inv_result_v1, scalar) == 8, "result scalar offset");
_Static_assert(offsetof(inv_result_v1, data) == 16, "result data offset");
_Static_assert(offsetof(inv_result_v1, len) == 24, "result length offset");
_Static_assert(offsetof(inv_result_v1, release) == 32, "result release offset");

#endif
