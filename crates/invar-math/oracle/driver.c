/* Differential oracle for invar-math-cr-v1.
 *
 * Linked against an independent build of upstream CORE-MATH (not the vendored
 * archive). For one function it evaluates, in order: the upstream worst-case
 * file, special values, `random` pseudo-random encodings and, for binary32
 * univariate functions, every `stride`-th encoding. It writes one record per
 * input to stdout: three little-endian u64 words (first operand, second
 * operand, result), binary32 values in the low 32 bits.
 *
 * usage: driver NAME 32|64 WCFILE|- RANDOM SEED STRIDE
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define F32(n) float cr_##n##f(float);
#define F32B(n) float cr_##n##f(float, float);
#define F64(n) double cr_##n(double);
#define F64B(n) double cr_##n(double, double);
#define UNARY(X) X(cbrt) X(sin) X(cos) X(tan) X(asin) X(acos) X(atan) X(sinh) X(cosh) \
    X(tanh) X(exp) X(exp2) X(expm1) X(log) X(log2) X(log10) X(log1p)
#define BINARY(X) X(hypot) X(atan2) X(pow)
UNARY(F32) BINARY(F32B) UNARY(F64) BINARY(F64B)

typedef struct {
    const char *name;
    float (*f1)(float);
    float (*f2)(float, float);
    double (*d1)(double);
    double (*d2)(double, double);
} entry;

#define E1(n) {#n, cr_##n##f, 0, cr_##n, 0},
#define E2(n) {#n, 0, cr_##n##f, 0, cr_##n},
static const entry table[] = {UNARY(E1) BINARY(E2)};

static const entry *selected;
static int binary64;

static void emit(uint64_t x, uint64_t y) {
    uint64_t record[3] = {x, y, 0};
    if (binary64) {
        double a, b, r;
        memcpy(&a, &x, 8);
        memcpy(&b, &y, 8);
        r = selected->d1 ? selected->d1(a) : selected->d2(a, b);
        memcpy(&record[2], &r, 8);
    } else {
        uint32_t xa = (uint32_t)x, yb = (uint32_t)y, rb;
        float a, b, r;
        memcpy(&a, &xa, 4);
        memcpy(&b, &yb, 4);
        r = selected->f1 ? selected->f1(a) : selected->f2(a, b);
        memcpy(&rb, &r, 4);
        record[2] = rb;
    }
    fwrite(record, 8, 3, stdout);
}

static uint64_t parse(const char *text) {
    if (binary64) {
        double value = strtod(text, 0);
        uint64_t bits;
        memcpy(&bits, &value, 8);
        return bits;
    }
    float value = strtof(text, 0);
    uint32_t bits;
    memcpy(&bits, &value, 4);
    return bits;
}

static void worst_cases(const char *path) {
    FILE *file = fopen(path, "r");
    if (!file) {
        perror(path);
        exit(2);
    }
    char line[512];
    while (fgets(line, sizeof line, file)) {
        char *hash = strchr(line, '#');
        if (hash) *hash = 0;
        if (!strpbrk(line, "0123456789")) continue;
        char *comma = strchr(line, ',');
        emit(parse(line), comma ? parse(comma + 1) : 0);
    }
    fclose(file);
}

static uint64_t next(uint64_t *state) {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    return *state;
}

/* Full-width encodings half the time, moderate exponents otherwise. */
static uint64_t sample(uint64_t *state) {
    uint64_t raw = next(state);
    if (binary64) {
        if (raw & 1) return raw;
        uint64_t exponent = 1023 - 40 + (raw >> 1) % 80;
        return (raw & 0x800fffffffffffffull) | (exponent << 52);
    }
    uint32_t bits = (uint32_t)(raw >> 16);
    if (raw & 1) return bits;
    uint32_t exponent = 127 - 20 + (uint32_t)((raw >> 1) % 40);
    return (bits & 0x807fffffu) | (exponent << 23);
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: driver NAME 32|64 WCFILE|- RANDOM SEED STRIDE\n");
        return 2;
    }
    for (size_t i = 0; i < sizeof table / sizeof table[0]; i++)
        if (!strcmp(table[i].name, argv[1])) selected = &table[i];
    if (!selected) return 2;
    binary64 = !strcmp(argv[2], "64");
    if (strcmp(argv[3], "-")) worst_cases(argv[3]);
    static const char *specials[] = {"0", "-0", "1", "-1", "0.5", "-0.5", "2", "-2", "inf", "-inf",
        "nan", "-nan", "0x1p-1074", "-0x1p-1074", "0x1p-149", "0x1p-126", "0x1.fffffffffffffp+1023",
        "0x1.fffffep+127", "3.14159265358979323846", "1e-300", "1e300", "710", "-746", "89", "-104"};
    size_t count = sizeof specials / sizeof specials[0];
    int bivariate = binary64 ? !selected->d1 : !selected->f1;
    for (size_t i = 0; i < count; i++)
        for (size_t j = 0; j < (bivariate ? count : 1); j++)
            emit(parse(specials[i]), bivariate ? parse(specials[j]) : 0);
    uint64_t state = strtoull(argv[5], 0, 10) | 1;
    for (long i = atol(argv[4]); i > 0; i--) {
        uint64_t x = sample(&state);
        emit(x, bivariate ? sample(&state) : 0);
    }
    uint64_t stride = strtoull(argv[6], 0, 10);
    if (!binary64 && !bivariate && stride)
        for (uint64_t bits = 0; bits <= 0xffffffffull; bits += stride) emit(bits, 0);
    return fflush(stdout) != 0;
}
