// invar-math-cr-v1 MSL prelude: exact arithmetic for the binary32 library.
//
// Metal has no binary64, but CORE-MATH's binary32 functions compute their
// intermediates in double. D holds an IEEE 754 binary64 encoding and every
// operation on it is computed with integer arithmetic, rounding to nearest
// even exactly once with gradual underflow: the result a correct binary64
// unit gives. F does the same for binary32 (see below), because Metal flushes
// binary32 subnormals. They serve only the internals of the f32 library;
// Invar's f64 programs never run on Metal (explain reason metal-no-f64).
// Packing and rounding follow Berkeley SoftFloat 3e's roundPackToF64 and
// roundPackToF32.
//
// The same text compiles as host C++ (no __METAL_VERSION__) so the unit and
// the transformed CORE-MATH sources can be checked off the GPU.
#ifndef INVAR_MSL_PRELUDE_H
#define INVAR_MSL_PRELUDE_H
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#pragma METAL fp contract(off)
#undef INVAR_CONSTANT
#undef INVAR_THREAD
#define INVAR_CONSTANT constant
#define INVAR_THREAD thread
static inline ulong invar_mulhi(ulong a, ulong b) { return metal::mulhi(a, b); }
// Table initializers convert at compile time, where the metal:: builtins
// are unavailable; the loops never run on the GPU.
static constexpr int invar_clz64(ulong x) {
  if (__builtin_is_constant_evaluated()) {
    int n = 0;
    while (n < 64 && !((x >> (63 - n)) & 1)) n++;
    return n;
  }
  return (int)metal::clz(x);
}
static constexpr int invar_clz32(uint x) { return x ? invar_clz64(x) - 32 : 32; }
static inline int invar_ctz64(ulong x) { return (int)metal::ctz(x); }
static inline int invar_ctz32(uint x) { return (int)metal::ctz(x); }

#else
#undef __x86_64__
#undef __i386__
#undef __aarch64__
#undef __arm64__
#undef __AVX__
#undef __SSE4_1__
#undef __ARM_ARCH
#undef __FMA__
#undef INVAR_CONSTANT
#undef INVAR_THREAD
#define INVAR_CONSTANT const
#define INVAR_THREAD
typedef unsigned long ulong;
typedef unsigned int uint;
static_assert(sizeof(ulong) == 8 && sizeof(long) == 8, "LP64 host required");
static inline ulong invar_mulhi(ulong a, ulong b) { return (ulong)(((unsigned __int128)a * b) >> 64); }
static constexpr int invar_clz64(ulong x) { return x ? __builtin_clzl(x) : 64; }
static constexpr int invar_clz32(uint x) { return x ? __builtin_clz(x) : 32; }
static inline int invar_ctz64(ulong x) { return x ? __builtin_ctzl(x) : 64; }
static inline int invar_ctz32(uint x) { return x ? __builtin_ctz(x) : 32; }
#endif
static constexpr uint invar_bits(float f) { return __builtin_bit_cast(uint, f); }
static constexpr float invar_float(uint b) { return __builtin_bit_cast(float, b); }
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef uint uint32_t;
typedef long int64_t;
typedef ulong uint64_t;
#define INT64_C(c) c##L
#define UINT64_C(c) c##UL

#define INVAR_SIGN 0x8000000000000000ul
#define INVAR_FRAC 0x000ffffffffffffful
#define INVAR_INF 0x7ff0000000000000ul
#define INVAR_QNAN 0x7ff8000000000000ul

static constexpr bool invar_isnan(ulong a) { return (a & ~INVAR_SIGN) > INVAR_INF; }

// a >> d with every shifted-out bit ORed into bit 0 (SoftFloat shiftRightJam64).
static constexpr ulong invar_jam(ulong a, int d) {
  if (d <= 0) return a;
  if (d >= 64) return (ulong)(a != 0);
  return (a >> d) | (ulong)((a << (64 - d)) != 0);
}

// The binary64 nearest to sig * 2^(exp - 1084), where sig < 2^63 carries its
// leading one at bit 62 unless the value is below the normal range. `odd`
// rounds to odd instead (truncate, then set the last bit when inexact), which
// a later rounding to binary32 turns into one correct rounding (53 >= 24 + 2).
static constexpr ulong invar_pack(ulong sign, int exp, ulong sig, bool odd = false) {
  if (exp >= 0x7fd && (exp > 0x7fd || sig + 0x200 >= INVAR_SIGN)) return sign | INVAR_INF;
  if (exp < 0) {
    sig = invar_jam(sig, -exp);
    exp = 0;
  }
  ulong bits = sig & 0x3ff;
  if (odd) return sign + ((ulong)exp << 52) + ((sig >> 10) | (ulong)(bits != 0));
  sig = (sig + 0x200) >> 10;
  if (bits == 0x200) sig &= ~1ul;
  if (sig == 0) exp = 0;
  return sign + ((ulong)exp << 52) + sig;
}

// The binary64 nearest to (hi * 2^64 + lo) * 2^q, which must be nonzero.
static constexpr ulong invar_round128(ulong sign, ulong hi, ulong lo, int q, bool odd = false) {
  int lead = hi ? 127 - invar_clz64(hi) : 63 - invar_clz64(lo);
  int shift = lead - 62;
  ulong sig;
  if (shift <= 0) {
    sig = lo << -shift;
  } else if (shift < 64) {
    sig = (hi << (64 - shift)) | (lo >> shift) | (ulong)((lo << (64 - shift)) != 0);
  } else {
    bool sticky = lo != 0 || (shift > 64 && (hi << (128 - shift)) != 0);
    sig = (hi >> (shift - 64)) | (ulong)sticky;
  }
  return invar_pack(sign, q + shift + 1084, sig, odd);
}

// A finite nonzero encoding as sig * 2^(exp - 1075) with sig in [2^52, 2^53).
struct InvarUnpacked {
  ulong sign;
  int exp;
  ulong sig;
};

static constexpr InvarUnpacked invar_unpack(ulong a) {
  InvarUnpacked u = {0, 0, 0};
  u.sign = a & INVAR_SIGN;
  int e = (int)((a >> 52) & 0x7ff);
  ulong f = a & INVAR_FRAC;
  if (e == 0) {
    int s = invar_clz64(f) - 11;
    f <<= s;
    e = 1 - s;
  } else {
    f |= 1ul << 52;
  }
  u.exp = e;
  u.sig = f;
  return u;
}

// (h, l) >>= d, ORing every shifted-out bit into bit 0.
static inline void invar_jam128(INVAR_THREAD ulong &h, INVAR_THREAD ulong &l, int d) {
  if (d <= 0) return;
  if (d < 64) {
    l = (h << (64 - d)) | (l >> d) | (ulong)((l << (64 - d)) != 0);
    h >>= d;
  } else if (d < 128) {
    bool sticky = l != 0 || (d > 64 && (h << (128 - d)) != 0);
    l = (d == 64 ? h : h >> (d - 64)) | (ulong)sticky;
    h = 0;
  } else {
    l = (ulong)((h | l) != 0);
    h = 0;
  }
}

// The binary64 nearest to x + y for x = (xh, xl) * 2^qx and y likewise, both
// nonzero with their leading one at bit 125. Shifting the smaller operand
// right by d >= 2 can only lose bits far below the rounding position, and by
// d <= 1 loses none because every operand's low bits are zero.
static ulong invar_sum(ulong sx, ulong xh, ulong xl, int qx, ulong sy, ulong yh, ulong yl, int qy,
                       bool odd = false) {
  if (qx < qy || (qx == qy && (xh < yh || (xh == yh && xl < yl)))) {
    ulong t = sx; sx = sy; sy = t;
    t = xh; xh = yh; yh = t;
    t = xl; xl = yl; yl = t;
    int u = qx; qx = qy; qy = u;
  }
  invar_jam128(yh, yl, qx - qy);
  ulong h, l;
  if (sx == sy) {
    l = xl + yl;
    h = xh + yh + (ulong)(l < xl);
  } else {
    l = xl - yl;
    h = xh - yh - (ulong)(xl < yl);
  }
  if ((h | l) == 0) return 0;
  return invar_round128(sx, h, l, qx, odd);
}

static ulong invar_add(ulong a, ulong b) {
  if (invar_isnan(a) || invar_isnan(b)) return INVAR_QNAN;
  ulong ma = a & ~INVAR_SIGN, mb = b & ~INVAR_SIGN;
  if (ma == INVAR_INF || mb == INVAR_INF) {
    if (ma == mb && ((a ^ b) & INVAR_SIGN)) return INVAR_QNAN;
    return ma == INVAR_INF ? a : b;
  }
  if (mb == 0) return ma == 0 ? (a & b) : a;
  if (ma == 0) return b;
  InvarUnpacked x = invar_unpack(a), y = invar_unpack(b);
  return invar_sum(x.sign, x.sig << 9, 0, x.exp - 1148, y.sign, y.sig << 9, 0, y.exp - 1148);
}

static ulong invar_mul(ulong a, ulong b) {
  ulong sign = (a ^ b) & INVAR_SIGN;
  if (invar_isnan(a) || invar_isnan(b)) return INVAR_QNAN;
  ulong ma = a & ~INVAR_SIGN, mb = b & ~INVAR_SIGN;
  if (ma == INVAR_INF || mb == INVAR_INF) return (ma == 0 || mb == 0) ? INVAR_QNAN : sign | INVAR_INF;
  if (ma == 0 || mb == 0) return sign;
  InvarUnpacked x = invar_unpack(a), y = invar_unpack(b);
  return invar_round128(sign, invar_mulhi(x.sig, y.sig), x.sig * y.sig, x.exp + y.exp - 2150);
}

static ulong invar_fma(ulong a, ulong b, ulong c, bool odd = false) {
  ulong sign = (a ^ b) & INVAR_SIGN;
  if (invar_isnan(a) || invar_isnan(b) || invar_isnan(c)) return INVAR_QNAN;
  ulong ma = a & ~INVAR_SIGN, mb = b & ~INVAR_SIGN, mc = c & ~INVAR_SIGN;
  if (ma == INVAR_INF || mb == INVAR_INF) {
    if (ma == 0 || mb == 0) return INVAR_QNAN;
    if (mc == INVAR_INF && (c & INVAR_SIGN) != sign) return INVAR_QNAN;
    return sign | INVAR_INF;
  }
  if (mc == INVAR_INF) return c;
  if (ma == 0 || mb == 0) return mc == 0 ? (sign & c) : c;
  if (mc == 0) return invar_mul(a, b);
  InvarUnpacked x = invar_unpack(a), y = invar_unpack(b), z = invar_unpack(c);
  ulong ph = invar_mulhi(x.sig, y.sig), pl = x.sig * y.sig;
  int q = x.exp + y.exp - 2150;
  int up = 125 - (127 - invar_clz64(ph));
  ph = (ph << up) | (pl >> (64 - up));
  pl <<= up;
  return invar_sum(sign, ph, pl, q - up, z.sign, z.sig << 9, 0, z.exp - 1148, odd);
}

static ulong invar_div(ulong a, ulong b) {
  ulong sign = (a ^ b) & INVAR_SIGN;
  if (invar_isnan(a) || invar_isnan(b)) return INVAR_QNAN;
  ulong ma = a & ~INVAR_SIGN, mb = b & ~INVAR_SIGN;
  if (ma == INVAR_INF) return mb == INVAR_INF ? INVAR_QNAN : sign | INVAR_INF;
  if (mb == INVAR_INF) return sign;
  if (mb == 0) return ma == 0 ? INVAR_QNAN : sign | INVAR_INF;
  if (ma == 0) return sign;
  InvarUnpacked x = invar_unpack(a), y = invar_unpack(b);
  ulong r = x.sig, d = y.sig, quotient = 0;
  int q = x.exp - y.exp;
  if (r < d) {
    r <<= 1;
    q -= 1;
  }
  for (int i = 0; i < 63; i++) {
    quotient <<= 1;
    if (r >= d) {
      r -= d;
      quotient |= 1;
    }
    r <<= 1;
  }
  return invar_pack(sign, q + 1022, quotient | (ulong)(r != 0));
}

static ulong invar_sqrt(ulong a) {
  if (invar_isnan(a)) return INVAR_QNAN;
  if ((a & ~INVAR_SIGN) == 0) return a;
  if (a & INVAR_SIGN) return INVAR_QNAN;
  if (a == INVAR_INF) return a;
  InvarUnpacked x = invar_unpack(a);
  ulong sig = x.sig;
  int e = x.exp - 1075;
  if (e & 1) {
    sig <<= 1;
    e -= 1;
  }
  // Digit-by-digit square root of sig * 2^56, 55 result bits.
  ulong r = 0, s = 0;
  for (int i = 54; i >= 0; i--) {
    int sh = 2 * i - 56;
    r = (r << 2) | (sh >= 0 ? (sig >> sh) & 3 : 0);
    ulong t = (s << 2) | 1;
    s <<= 1;
    if (r >= t) {
      r -= t;
      s |= 1;
    }
  }
  return invar_pack(0, (e - 56) / 2 + 1076, (s << 8) | (ulong)(r != 0));
}

// The ordering key of a non-NaN encoding, with -0 equal to +0.
static inline ulong invar_key(ulong a) {
  if ((a & ~INVAR_SIGN) == 0) return INVAR_SIGN;
  return (a & INVAR_SIGN) ? ~a : (a | INVAR_SIGN);
}

static inline bool invar_lt(ulong a, ulong b) {
  return !invar_isnan(a) && !invar_isnan(b) && invar_key(a) < invar_key(b);
}

static inline bool invar_eq(ulong a, ulong b) {
  return !invar_isnan(a) && !invar_isnan(b) && invar_key(a) == invar_key(b);
}

static constexpr ulong invar_from_u64(ulong sign, ulong v, bool odd = false) {
  return v == 0 ? 0 : invar_round128(sign, 0, v, 0, odd);
}

static constexpr ulong invar_from_i64(long v) {
  return v < 0 ? invar_from_u64(INVAR_SIGN, 0ul - (ulong)v) : invar_from_u64(0, (ulong)v);
}

static constexpr ulong invar_from_f32(uint f) {
  ulong sign = (ulong)(f >> 31) << 63;
  int e = (int)((f >> 23) & 0xff);
  uint m = f & 0x7fffff;
  if (e == 0xff) return m ? INVAR_QNAN : sign | INVAR_INF;
  if (e == 0) {
    if (m == 0) return sign;
    int s = invar_clz32(m) - 8;
    m = (m << s) & 0x7fffff;
    e = 1 - s;
  }
  return sign | ((ulong)(e + 896) << 52) | ((ulong)m << 29);
}

// The binary32 nearest to a binary64 encoding (SoftFloat roundPackToF32).
// Kept out of line: inlined after a mask that clears the low word (cbrtf's
// final rounding), it crashes the macOS 27 AGX backend compiler
// (XPC_ERROR_CONNECTION_INTERRUPTED).
__attribute__((noinline)) static constexpr uint invar_to_f32(ulong a) {
  uint sign = (uint)(a >> 63) << 31;
  if (invar_isnan(a)) return 0x7fc00000u;
  if ((a & ~INVAR_SIGN) == INVAR_INF) return sign | 0x7f800000u;
  if ((a & ~INVAR_SIGN) == 0) return sign;
  InvarUnpacked x = invar_unpack(a);
  ulong wide = x.sig << 10;
  uint sig = (uint)(wide >> 32) | (uint)((wide & 0xffffffffu) != 0);
  int exp = x.exp - 897;
  if (exp >= 0xfd && (exp > 0xfd || sig + 0x40 >= 0x80000000u)) return sign | 0x7f800000u;
  if (exp < 0) {
    int d = -exp;
    sig = d >= 32 ? (uint)(sig != 0) : (sig >> d) | (uint)((sig << (32 - d)) != 0);
    exp = 0;
  }
  uint bits = sig & 0x7f;
  sig = (sig + 0x40) >> 7;
  if (bits == 0x40) sig &= ~1u;
  if (sig == 0) exp = 0;
  return sign + ((uint)exp << 23) + sig;
}

// Truncation toward zero; out of range is undefined in C and unused upstream.
static inline long invar_to_i64(ulong a) {
  int e = (int)((a >> 52) & 0x7ff);
  if (e < 1023) return 0;
  if (e >= 1023 + 63) return (long)INVAR_SIGN;
  ulong sig = (a & INVAR_FRAC) | (1ul << 52);
  ulong v = e >= 1075 ? sig << (e - 1075) : sig >> (1075 - e);
  return (a & INVAR_SIGN) ? -(long)v : (long)v;
}

static inline ulong invar_to_u64(ulong a) {
  int e = (int)((a >> 52) & 0x7ff);
  if (e < 1023 || (a & INVAR_SIGN)) return (ulong)invar_to_i64(a);
  if (e >= 1023 + 64) return 0;
  ulong sig = (a & INVAR_FRAC) | (1ul << 52);
  return e >= 1075 ? sig << (e - 1075) : sig >> (1075 - e);
}

// Round to an integral value: 0 nearest-even, 1 nearest-away, 2 floor,
// 3 ceil, 4 trunc. Exact, so no rounding of the result is involved.
static inline ulong invar_integral(ulong a, int mode) {
  int e = (int)((a >> 52) & 0x7ff);
  ulong sign = a & INVAR_SIGN;
  if (e >= 1075) return invar_isnan(a) ? INVAR_QNAN : a;
  if (e < 1023) {
    if ((a & ~INVAR_SIGN) == 0) return a;
    bool one = (mode == 0 && e == 1022 && (a & INVAR_FRAC) != 0) || (mode == 1 && e == 1022) ||
               (mode == 2 && sign) || (mode == 3 && !sign);
    return sign | (one ? 0x3ff0000000000000ul : 0);
  }
  int k = 1075 - e;
  ulong mask = (1ul << k) - 1, frac = a & mask, midpoint = 1ul << (k - 1), t = a & ~mask;
  if (frac == 0) return a;
  bool up = (mode == 0 && (frac > midpoint || (frac == midpoint && ((t >> k) & 1)))) || (mode == 1 && frac >= midpoint) ||
            (mode == 2 && sign) || (mode == 3 && !sign);
  return up ? t + (1ul << k) : t;
}

struct D {
  ulong b;
  D() = default;
  constexpr D(ulong bits, int) : b(bits) {}
  static constexpr D raw(ulong bits) { return D(bits, 0); }
  constexpr D(float f) : b(invar_from_f32(invar_bits(f))) {}
  constexpr D(int v) : b(invar_from_i64(v)) {}
  constexpr D(uint v) : b(invar_from_u64(0, v)) {}
  constexpr D(long v) : b(invar_from_i64(v)) {}
  constexpr D(ulong v) : b(invar_from_u64(0, v)) {}
  operator float() const { return invar_float(invar_to_f32(b)); }
  explicit operator int() const { return (int)invar_to_i64(b); }
  explicit operator uint() const { return (uint)invar_to_u64(b); }
  explicit operator long() const { return invar_to_i64(b); }
  explicit operator ulong() const { return invar_to_u64(b); }
  explicit operator bool() const { return (b & ~INVAR_SIGN) != 0; }
#ifdef __METAL_VERSION__
  // Tables live in the constant address space; convert them in place.
  operator float() const constant { return invar_float(invar_to_f32(b)); }
  explicit operator int() const constant { return (int)invar_to_i64(b); }
  explicit operator uint() const constant { return (uint)invar_to_u64(b); }
  explicit operator long() const constant { return invar_to_i64(b); }
  explicit operator ulong() const constant { return invar_to_u64(b); }
#endif
};

constexpr D operator-(D a) { return D::raw(a.b ^ INVAR_SIGN); }
constexpr D operator+(D a) { return a; }
inline D operator+(D a, D b) { return D::raw(invar_add(a.b, b.b)); }
inline D operator-(D a, D b) { return D::raw(invar_add(a.b, b.b ^ INVAR_SIGN)); }
inline D operator*(D a, D b) { return D::raw(invar_mul(a.b, b.b)); }
inline D operator/(D a, D b) { return D::raw(invar_div(a.b, b.b)); }
inline bool operator<(D a, D b) { return invar_lt(a.b, b.b); }
inline bool operator>(D a, D b) { return invar_lt(b.b, a.b); }
inline bool operator<=(D a, D b) { return invar_lt(a.b, b.b) || invar_eq(a.b, b.b); }
inline bool operator>=(D a, D b) { return invar_lt(b.b, a.b) || invar_eq(a.b, b.b); }
inline bool operator==(D a, D b) { return invar_eq(a.b, b.b); }
inline bool operator!=(D a, D b) { return !invar_eq(a.b, b.b); }
// C converts the other operand of a mixed expression to double.
#define INVAR_MIXED(op, R)                                                                         \
  template <class T> inline R operator op(D a, T b) { return a op D(b); }                          \
  template <class T> inline R operator op(T a, D b) { return D(a) op b; }
INVAR_MIXED(+, D)
INVAR_MIXED(-, D)
INVAR_MIXED(*, D)
INVAR_MIXED(/, D)
INVAR_MIXED(<, bool)
INVAR_MIXED(>, bool)
INVAR_MIXED(<=, bool)
INVAR_MIXED(>=, bool)
INVAR_MIXED(==, bool)
INVAR_MIXED(!=, bool)
#undef INVAR_MIXED
template <class T> inline INVAR_THREAD D &operator+=(INVAR_THREAD D &a, T b) { return a = a + D(b); }
template <class T> inline INVAR_THREAD D &operator-=(INVAR_THREAD D &a, T b) { return a = a - D(b); }
template <class T> inline INVAR_THREAD D &operator*=(INVAR_THREAD D &a, T b) { return a = a * D(b); }
template <class T> inline INVAR_THREAD D &operator/=(INVAR_THREAD D &a, T b) { return a = a / D(b); }

// unsigned __int128 for the Payne-Hanek reductions (modulo 2^128, as in C).
struct invar_u128 {
  ulong lo, hi;
  invar_u128() = default;
  constexpr invar_u128(ulong v) : lo(v), hi(0) {}
  constexpr invar_u128(ulong h, ulong l, int) : lo(l), hi(h) {}
  operator ulong() const { return lo; }
};
inline invar_u128 operator*(invar_u128 a, invar_u128 b) {
  return invar_u128(invar_mulhi(a.lo, b.lo) + a.lo * b.hi + a.hi * b.lo, a.lo * b.lo, 0);
}
inline invar_u128 operator*(invar_u128 a, ulong b) { return a * invar_u128(b); }
inline invar_u128 operator+(invar_u128 a, invar_u128 b) {
  ulong lo = a.lo + b.lo;
  return invar_u128(a.hi + b.hi + (ulong)(lo < a.lo), lo, 0);
}
inline INVAR_THREAD invar_u128 &operator+=(INVAR_THREAD invar_u128 &a, invar_u128 b) { return a = a + b; }
inline invar_u128 operator>>(invar_u128 a, int d) {
  if (d == 0) return a;
  if (d >= 64) return invar_u128(0, d >= 128 ? 0 : a.hi >> (d - 64), 0);
  return invar_u128(a.hi >> d, (a.lo >> d) | (a.hi << (64 - d)), 0);
}
inline invar_u128 operator<<(invar_u128 a, int d) {
  if (d == 0) return a;
  if (d >= 64) return invar_u128(d >= 128 ? 0 : a.lo << (d - 64), 0, 0);
  return invar_u128((a.hi << d) | (a.lo >> (64 - d)), a.lo << d, 0);
}

static inline D invar_d_fabs(D a) { return D::raw(a.b & ~INVAR_SIGN); }
static inline D invar_d_copysign(D a, D b) { return D::raw((a.b & ~INVAR_SIGN) | (b.b & INVAR_SIGN)); }
static inline D invar_d_fma(D a, D b, D c) { return D::raw(invar_fma(a.b, b.b, c.b)); }
static inline D invar_d_sqrt(D a) { return D::raw(invar_sqrt(a.b)); }
static inline D invar_d_roundeven(D a) { return D::raw(invar_integral(a.b, 0)); }
static inline D invar_d_round(D a) { return D::raw(invar_integral(a.b, 1)); }
static inline D invar_d_floor(D a) { return D::raw(invar_integral(a.b, 2)); }
static inline D invar_d_ceil(D a) { return D::raw(invar_integral(a.b, 3)); }
static inline D invar_d_trunc(D a) { return D::raw(invar_integral(a.b, 4)); }
static inline D invar_d_fmax(D a, D b) { return invar_isnan(a.b) ? b : invar_isnan(b.b) ? a : (a < b ? b : a); }
static inline D invar_d_fmin(D a, D b) { return invar_isnan(a.b) ? b : invar_isnan(b.b) ? a : (b < a ? b : a); }
// F holds a binary32 encoding. Metal flushes subnormal binary32 operands and
// results even with fast math disabled, so every binary32 operation upstream
// performs goes through D: + - * / and sqrt rounded twice through binary64
// are still correctly rounded (53 >= 2 * 24 + 2), fma rounds to odd in
// binary64 first (53 >= 24 + 2), and comparisons use the encodings.
static constexpr bool invar_f_isnan(uint u) { return (u & 0x7fffffffu) > 0x7f800000u; }
// Total order key (-0 before +0) and the IEEE comparison key (-0 equals +0).
static constexpr uint invar_f_key(uint u) { return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
static constexpr uint invar_f_cmp(uint u) { return (u & 0x7fffffffu) == 0 ? 0x80000000u : invar_f_key(u); }

struct F {
  uint u;
  F() = default;
  constexpr F(uint bits, int) : u(bits) {}
  static constexpr F raw(uint bits) { return F(bits, 0); }
  constexpr F(float f) : u(invar_bits(f)) {}
  constexpr F(D d) : u(invar_to_f32(d.b)) {}
  constexpr F(int v) : u(invar_to_f32(invar_from_i64(v))) {}
  constexpr F(uint v) : u(invar_to_f32(invar_from_u64(0, v))) {}
  constexpr F(long v)
      : u(invar_to_f32(v < 0 ? invar_from_u64(INVAR_SIGN, 0ul - (ulong)v, true) : invar_from_u64(0, (ulong)v, true))) {}
  constexpr F(ulong v) : u(invar_to_f32(invar_from_u64(0, v, true))) {}
  constexpr operator D() const { return D::raw(invar_from_f32(u)); }
  explicit operator int() const { return (int)invar_to_i64(invar_from_f32(u)); }
  explicit operator uint() const { return (uint)invar_to_u64(invar_from_f32(u)); }
  explicit operator long() const { return invar_to_i64(invar_from_f32(u)); }
  explicit operator ulong() const { return invar_to_u64(invar_from_f32(u)); }
  explicit operator bool() const { return (u & 0x7fffffffu) != 0; }
#ifdef __METAL_VERSION__
  operator D() const constant { return D::raw(invar_from_f32(u)); }
  explicit operator int() const constant { return (int)invar_to_i64(invar_from_f32(u)); }
  explicit operator uint() const constant { return (uint)invar_to_u64(invar_from_f32(u)); }
#endif
};

constexpr F operator-(F a) { return F::raw(a.u ^ 0x80000000u); }
constexpr F operator+(F a) { return a; }
inline F operator+(F a, F b) { return F(D(a) + D(b)); }
inline F operator-(F a, F b) { return F(D(a) - D(b)); }
inline F operator*(F a, F b) { return F(D(a) * D(b)); }
inline F operator/(F a, F b) { return F(D(a) / D(b)); }
inline bool operator<(F a, F b) { return !invar_f_isnan(a.u) && !invar_f_isnan(b.u) && invar_f_cmp(a.u) < invar_f_cmp(b.u); }
inline bool operator>(F a, F b) { return b < a; }
inline bool operator<=(F a, F b) { return !invar_f_isnan(a.u) && !invar_f_isnan(b.u) && invar_f_cmp(a.u) <= invar_f_cmp(b.u); }
inline bool operator>=(F a, F b) { return b <= a; }
inline bool operator==(F a, F b) { return !invar_f_isnan(a.u) && !invar_f_isnan(b.u) && invar_f_cmp(a.u) == invar_f_cmp(b.u); }
inline bool operator!=(F a, F b) { return !(a == b); }
// C converts an integer operand to float and a float operand meeting a
// double to double.
#define INVAR_MIXED_F(op, R)                                                                       \
  template <class T> inline R operator op(F a, T b) { return a op F(b); }                          \
  template <class T> inline R operator op(T a, F b) { return F(a) op b; }                          \
  inline auto operator op(F a, D b) { return D(a) op b; }                                          \
  inline auto operator op(D a, F b) { return a op D(b); }
INVAR_MIXED_F(+, F)
INVAR_MIXED_F(-, F)
INVAR_MIXED_F(*, F)
INVAR_MIXED_F(/, F)
INVAR_MIXED_F(<, bool)
INVAR_MIXED_F(>, bool)
INVAR_MIXED_F(<=, bool)
INVAR_MIXED_F(>=, bool)
INVAR_MIXED_F(==, bool)
INVAR_MIXED_F(!=, bool)
#undef INVAR_MIXED_F
template <class T> inline INVAR_THREAD F &operator+=(INVAR_THREAD F &a, T b) { return a = F(a + b); }
template <class T> inline INVAR_THREAD F &operator-=(INVAR_THREAD F &a, T b) { return a = F(a - b); }
template <class T> inline INVAR_THREAD F &operator*=(INVAR_THREAD F &a, T b) { return a = F(a * b); }
template <class T> inline INVAR_THREAD F &operator/=(INVAR_THREAD F &a, T b) { return a = F(a / b); }

static inline F invar_f_sqrt(F a) { return F(invar_d_sqrt(D(a))); }
static inline F invar_f_fabs(F a) { return F::raw(a.u & 0x7fffffffu); }
static inline F invar_f_copysign(F a, F b) { return F::raw((a.u & 0x7fffffffu) | (b.u & 0x80000000u)); }
static inline F invar_f_round(F a) { return F(invar_d_round(D(a))); }
static inline F invar_f_floor(F a) { return F(invar_d_floor(D(a))); }
static inline F invar_f_ceil(F a) { return F(invar_d_ceil(D(a))); }
static inline F invar_f_trunc(F a) { return F(invar_d_trunc(D(a))); }
static inline F invar_fmaf(F a, F b, F c) {
  return F::raw(invar_to_f32(invar_fma(invar_from_f32(a.u), invar_from_f32(b.u), invar_from_f32(c.u), true)));
}
// IEEE 754-2019 minimumNumber and maximumNumber (-0 < +0), and C's fmin and
// fmax (either zero on a tie).
static inline F invar_f_minimum_number(F a, F b) {
  if (invar_f_isnan(a.u)) return b;
  if (invar_f_isnan(b.u)) return a;
  return invar_f_key(a.u) <= invar_f_key(b.u) ? a : b;
}
static inline F invar_f_maximum_number(F a, F b) {
  if (invar_f_isnan(a.u)) return b;
  if (invar_f_isnan(b.u)) return a;
  return invar_f_key(a.u) >= invar_f_key(b.u) ? a : b;
}
static inline F invar_f_fmax(F a, F b) { return invar_f_maximum_number(a, b); }
static inline F invar_f_fmin(F a, F b) { return invar_f_minimum_number(a, b); }

#define double D
#define float F
#define __builtin_expect(e, v) (e)
#define __builtin_fabs invar_d_fabs
#define __builtin_copysign invar_d_copysign
#define __builtin_fma invar_d_fma
#define __builtin_sqrt invar_d_sqrt
#define __builtin_roundeven invar_d_roundeven
#define __builtin_round invar_d_round
#define __builtin_floor invar_d_floor
#define __builtin_ceil invar_d_ceil
#define __builtin_trunc invar_d_trunc
#define __builtin_fmax invar_d_fmax
#define __builtin_fmin invar_d_fmin
#define __builtin_fabsf invar_f_fabs
#define __builtin_copysignf invar_f_copysign
#define __builtin_fmaf invar_fmaf
#define __builtin_sqrtf invar_f_sqrt
#define __builtin_roundf invar_f_round
#define __builtin_fmaxf invar_f_fmax
#define __builtin_fminf invar_f_fmin
#define __builtin_inff() invar_float(0x7f800000u)
#define __builtin_nanf(tag) invar_float(0x7fc00000u)
#define __builtin_clz(x) invar_clz32((uint)(x))
#define __builtin_clzll(x) invar_clz64((ulong)(x))
#define __builtin_ctz(x) invar_ctz32((uint)(x))
#define __builtin_ctzll(x) invar_ctz64((ulong)(x))
#define FE_INVALID 1
#define FE_DIVBYZERO 4
#define FE_OVERFLOW 8
#define FE_UNDERFLOW 16
#define FE_INEXACT 32
#define FE_ALL_EXCEPT 61
#define FE_TONEAREST 0
typedef uint fexcept_t;
static inline int fegetround() { return FE_TONEAREST; }
#define feraiseexcept(e) ((void)0)
#define feclearexcept(e) ((void)0)
#define fetestexcept(e) 0
#define fegetexcept() 0
#define fesetexcept(e) ((void)0)
#define fegetexceptflag(p, e) ((void)(p))
#define fesetexceptflag(p, e) ((void)(p))
#endif // INVAR_MSL_PRELUDE_H
