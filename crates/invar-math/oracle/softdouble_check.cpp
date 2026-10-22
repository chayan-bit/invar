// The MSL prelude's binary64 unit against the host FPU (built with
// -ffp-contract=off): every operation on random, edge-biased encodings must
// give the same encoding, NaN as one class. Usage: softdouble_check [cases].
#include "msl_prelude.h"
#undef double
#undef float
#undef __builtin_fma
#undef __builtin_sqrt
#undef __builtin_fmaf
#undef __builtin_sqrtf
#undef __builtin_expect
#undef __builtin_fabs
#undef __builtin_roundeven
#undef __builtin_round
#undef __builtin_floor
#undef __builtin_ceil
#undef __builtin_trunc
extern "C" int printf(const char *, ...);
extern "C" long strtol(const char *, char **, int);
static ulong state = 0x9e3779b97f4a7c15ul;
static ulong next() { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state; }
static ulong pick() {
  ulong r = next();
  switch (next() % 8) {
    case 0: return r;                                  // anything
    case 1: return (r & 0x800ffffffffffffful) | ((ulong)(next() % 64) << 52);   // subnormal/tiny
    case 2: return (r & 0x800ffffffffffffful) | ((ulong)(2047 - 1 - next() % 64) << 52); // huge
    case 3: return (r & 0x800ffffffffffffful) | ((ulong)(1023 - 30 + next() % 60) << 52); // moderate
    case 4: { ulong s[] = {0, INVAR_SIGN, INVAR_INF, INVAR_INF|INVAR_SIGN, INVAR_QNAN, 0x3ff0000000000000ul, 1, 0x000ffffffffffffful, 0x0010000000000000ul, 0x7fefffffffffffful};
              return s[next() % 10] ^ (next() & INVAR_SIGN); }
    case 5: return (r & 0xfff0000000000000ul) | (next() % 8) | ((next() & 1) ? INVAR_FRAC & ~7ul : 0); // near powers of two
    case 6: return 0x4330000000000000ul + (r & 0xfffffff) ; // integers near 2^52
    default: return (r & 0x800ffffffffffffful) | ((ulong)(1023 + (long)(next() % 7) - 3) << 52) ; // near 1
  }
}
static double f(ulong b) { return __builtin_bit_cast(double, b); }
static ulong u(double d) { return __builtin_bit_cast(ulong, d); }
static bool same(ulong a, ulong b) { return a == b || (invar_isnan(a) && invar_isnan(b)); }
static long fails = 0;
static void check32(const char *op, uint a, uint b, uint got, uint want) {
  bool nan = (got & 0x7fffffffu) > 0x7f800000u && (want & 0x7fffffffu) > 0x7f800000u;
  if (got != want && !nan && fails++ < 20) printf("%s %08x %08x got %08x want %08x\n", op, a, b, got, want);
}
static void check32(const char *op, uint a, uint b, bool got, bool want) { check32(op, a, b, (uint)got, (uint)want); }
static void check(const char *op, ulong a, ulong b, ulong c, ulong got, ulong want) {
  if (!same(got, want) && fails++ < 20) printf("%s %016lx %016lx %016lx got %016lx want %016lx\n", op, a, b, c, got, want);
}
static void check32(const char *op, uint a, uint b, uint got, float want) {
  check32(op, a, b, got, __builtin_bit_cast(uint, want));
}
int main(int argc, char **argv) {
  long n = argc > 1 ? strtol(argv[1], 0, 10) : 1000000;
  for (long i = 0; i < n; i++) {
    ulong a = pick(), b = pick(), c = pick();
    if (next() & 1) b = (b & ~0x7ff0000000000000ul) | ((a & 0x7ff0000000000000ul) + ((next() % 5) << 52) - (2ul << 52)) % 0x7ff0000000000000ul; // close exponents
    check("add", a, b, 0, invar_add(a, b), u(f(a) + f(b)));
    check("sub", a, b, 0, invar_add(a, b ^ INVAR_SIGN), u(f(a) - f(b)));
    check("mul", a, b, 0, invar_mul(a, b), u(f(a) * f(b)));
    check("div", a, b, 0, invar_div(a, b), u(f(a) / f(b)));
    check("fma", a, b, c, invar_fma(a, b, c), u(__builtin_fma(f(a), f(b), f(c))));
    ulong p = invar_mul(a, b);   // fma near cancellation
    check("fmacancel", a, b, p ^ INVAR_SIGN, invar_fma(a, b, p ^ INVAR_SIGN), u(__builtin_fma(f(a), f(b), -f(p))));
    check("sqrt", a, 0, 0, invar_sqrt(a), u(__builtin_sqrt(f(a))));
    check("lt", a, b, 0, invar_lt(a, b), f(a) < f(b));
    check("eq", a, b, 0, invar_eq(a, b), f(a) == f(b));
    uint fa = (uint)a;
    check("fromf32", fa, 0, 0, invar_from_f32(fa), u((double)__builtin_bit_cast(float, fa)));
    ulong tf = invar_to_f32(a), wf = __builtin_bit_cast(uint, (float)f(a));
    if (!(tf == wf || ((tf & 0x7fffffff) > 0x7f800000 && (wf & 0x7fffffff) > 0x7f800000))) check("tof32", a, 0, 0, tf, wf);
    check("fromi64", a, 0, 0, invar_from_i64((long)a), u((double)(long)a));
    check("fromu64", a, 0, 0, invar_from_u64(0, a), u((double)a));
    if (__builtin_fabs(f(a)) < 9.2e18) check("toi64", a, 0, 0, (ulong)invar_to_i64(a), (ulong)(long)f(a));
    if (f(a) > -1 && f(a) < 1.8e19) check("tou64", a, 0, 0, invar_to_u64(a), (ulong)f(a));
    check("rint", a, 0, 0, invar_integral(a, 0), u(__builtin_roundeven(f(a))));
    check("round", a, 0, 0, invar_integral(a, 1), u(__builtin_round(f(a))));
    check("floor", a, 0, 0, invar_integral(a, 2), u(__builtin_floor(f(a))));
    check("ceil", a, 0, 0, invar_integral(a, 3), u(__builtin_ceil(f(a))));
    check("trunc", a, 0, 0, invar_integral(a, 4), u(__builtin_trunc(f(a))));
    // binary32 through F (the host does not flush subnormals).
    uint x = (uint)(next() & 1 ? a : a >> 32), y = (uint)b, z = (uint)c;
    if (next() & 1) y = (y & 0x807fffffu) | (x & 0x7f800000u);  // close exponents
    if (next() & 1) x &= 0x80ffffffu;                         // subnormal or tiny
    float fx = __builtin_bit_cast(float, x), fy = __builtin_bit_cast(float, y), fz = __builtin_bit_cast(float, z);
    F px = F::raw(x), py = F::raw(y), pz = F::raw(z);
    check32("fadd", x, y, (px + py).u, fx + fy);
    check32("fsub", x, y, (px - py).u, fx - fy);
    check32("fmul", x, y, (px * py).u, fx * fy);
    check32("fdiv", x, y, (px / py).u, fx / fy);
    check32("fsqrt", x, 0, invar_f_sqrt(px).u, __builtin_sqrtf(fx));
    check32("ffma", x, y, invar_fmaf(px, py, pz).u, __builtin_fmaf(fx, fy, fz));
    F prod = px * py;  // fma near cancellation
    check32("ffmacancel", x, y, invar_fmaf(px, py, -prod).u, __builtin_fmaf(fx, fy, -__builtin_bit_cast(float, prod.u)));
    check32("flt", x, y, px < py, fx < fy);
    check32("feq", x, y, px == py, fx == fy);
    check32("fle", x, y, px <= py, fx <= fy);
    check32("fromi64", x, 0, F((long)a).u, (float)(long)a);
    check32("fromu64", x, 0, F(a).u, (float)a);
    check32("fromi32", x, 0, F((int)a).u, (float)(int)a);
  }
  printf("cases=%ld fails=%ld\n", n, fails);
  return fails != 0;
}
