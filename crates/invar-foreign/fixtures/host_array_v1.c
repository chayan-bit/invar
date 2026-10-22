/* C-FIXTURE EVIDENCE ONLY for the private host-array protocol v1 (HA-1 §12).
 * This is not compiled-source support: no Invar program is compiled, loaded
 * or run. It checks the record layout from C and exercises the Rust
 * rollback guard against real writes at the caller's addresses. */
#include <stddef.h>
#include <stdint.h>

struct inv_private_host_array_v1 {
    uint32_t element; /* 1 = int32_t, 2 = uint32_t */
    uint32_t mode;    /* 1 = read, 3 = mut */
    void *data;       /* NULL iff len == 0 */
    uint64_t len;     /* element count */
};

_Static_assert(sizeof(void *) == 8, "64-bit host ABI");
_Static_assert(sizeof(struct inv_private_host_array_v1) == 24, "size");
_Static_assert(_Alignof(struct inv_private_host_array_v1) == 8, "alignment");
_Static_assert(offsetof(struct inv_private_host_array_v1, element) == 0, "element");
_Static_assert(offsetof(struct inv_private_host_array_v1, mode) == 4, "mode");
_Static_assert(offsetof(struct inv_private_host_array_v1, data) == 8, "data");
_Static_assert(offsetof(struct inv_private_host_array_v1, len) == 16, "len");

void ha1_fixture_layout_v1(uint64_t out[6]) {
    out[0] = sizeof(struct inv_private_host_array_v1);
    out[1] = _Alignof(struct inv_private_host_array_v1);
    out[2] = offsetof(struct inv_private_host_array_v1, element);
    out[3] = offsetof(struct inv_private_host_array_v1, mode);
    out[4] = offsetof(struct inv_private_host_array_v1, data);
    out[5] = offsetof(struct inv_private_host_array_v1, len);
}

/* For each record: check tags, length and the NULL-iff-empty rule; report
 * the address received; sum read arrays; write the first `writes` elements
 * of each mut array with a deterministic bit pattern. Returns 0 (normal),
 * 1 (the fixture reports a non-normal outcome after its writes) or
 * 2 (an invalid record). */
int32_t ha1_fixture_call_v1(const struct inv_private_host_array_v1 *records,
                            uint32_t count, uint64_t writes, uint32_t fail,
                            uint64_t *seen, int64_t *read_sum) {
    int64_t sum = 0;
    for (uint32_t k = 0; k < count; ++k) {
        const struct inv_private_host_array_v1 *r = &records[k];
        if ((r->element != 1 && r->element != 2) || (r->mode != 1 && r->mode != 3) ||
            r->len > 1000000 || ((r->len == 0) != (r->data == NULL))) {
            return 2;
        }
        seen[k] = (uint64_t)(uintptr_t)r->data;
        if (r->mode == 1) {
            for (uint64_t i = 0; i < r->len; ++i) {
                sum += r->element == 1 ? (int64_t)((const int32_t *)r->data)[i]
                                       : (int64_t)((const uint32_t *)r->data)[i];
            }
        } else {
            uint64_t n = writes < r->len ? writes : r->len;
            for (uint64_t i = 0; i < n; ++i) {
                uint32_t bits = (r->element == 1 ? 0x7fa00001u : 0xffc12345u) ^ (uint32_t)(i * 7u + k);
                if (r->element == 1) {
                    ((int32_t *)r->data)[i] = (int32_t)bits;
                } else {
                    ((uint32_t *)r->data)[i] = bits;
                }
            }
        }
    }
    *read_sum = sum;
    return fail ? 1 : 0;
}
