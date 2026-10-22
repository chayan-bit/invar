// Handwritten B-form baselines of the C1 fork-join family (GNET-05 corpus request C1), after the
// P4a spike's B kernels: digit replay of the fork tree to a fixed depth D,
// one lane per digit string (radix R = 2^RB), a checked serial frame machine
// below depth D, in-group joins (kernel hw_b) and the remaining top joins
// (kernel hw_top). Every + - * is checked; a trap carries no span (the bench
// asserts that every timed input returns a value equal to the oracle's).
//
// The host prepends a dialect header defining HW_FN, HW_TH, HW_DEV, HW_I64,
// HW_SHARED, HW_KERNEL_B, HW_KERNEL_TOP, HW_IDS (lane and group index),
// HW_BARRIER and PROGRAM.

struct O { int v; uint trap; };
struct A { int x, y, z; };
struct N { uint leaf; uint k; uint sel; int v; A c[3]; };

HW_FN int ck(HW_I64 w, HW_TH O& o) {
  if (w < (HW_I64)(-2147483647 - 1) || w > (HW_I64)2147483647) o.trap = 1u;
  return (int)w;
}
#define K(e) ck((HW_I64)(e), o)
HW_FN A a3(int x, int y, int z) { A a; a.x = x; a.y = y; a.z = z; return a; }
HW_FN N leaf_of(int v) { N n; n.leaf = 1u; n.k = 0u; n.sel = 0u; n.v = v; return n; }
HW_FN N fork_of(uint sel, uint k, A c0, A c1, A c2) {
  N n; n.leaf = 0u; n.k = k; n.sel = sel; n.v = 0; n.c[0] = c0; n.c[1] = c1; n.c[2] = c2; return n;
}

// ---- per-program node, extra-call and combine -------------------------
// t family: A = (n, s, 0); dc family: A = (lo, hi, d).

#if PROGRAM == 0 || PROGRAM == 2 || PROGRAM == 4 || PROGRAM == 6 || PROGRAM == 11
#define HW_COLLATZ_LEAF 1
#endif

#if PROGRAM == 0
#define RB 2
HW_FN int w(int x, HW_TH O& o) { if (x % 2 == 0) return K(x / 2 + 38) % 10007; return K(K(x * 3) + 1) % 10007; }
HW_FN N node(A a, HW_TH O& o) {
  int n = a.x, s = a.y;
  if (n <= 0) return leaf_of(w(s, o));
  int sel = K(K(s * 7) + K(n * 2)) % 4;
  if (sel == 0) return fork_of(0u, 3u, a3(K(n - 2), K(K(s * 22) + 38) % 1009, 0), a3(K(n - 1), K(K(s * 27) + 96) % 1009, 0), a3(K(n - 2), K(K(s * 24) + 56) % 1009, 0));
  if (sel == 1) return fork_of(9u, 1u, a3(K(n - 1), K(K(s * 7) + 88) % 1009, 0), a3(0, 0, 0), a3(0, 0, 0));
  if (sel == 2) return fork_of(2u, 3u, a3(K(n - 2), K(K(s * 4) + 43) % 1009, 0), a3(K(n - 1), K(K(s * 22) + 43) % 1009, 0), a3(K(n - 2), K(K(s * 16) + 79) % 1009, 0));
  return fork_of(9u, 1u, a3(K(n - 1), K(K(s * 17) + 95) % 1009, 0), a3(0, 0, 0), a3(0, 0, 0));
}
HW_FN bool extra(A a, uint sel, HW_TH int* v, HW_TH A& e, HW_TH O& o) { return false; }
HW_FN int combine(A a, uint sel, HW_TH int* v, int e, HW_TH O& o) {
  if (sel == 9u) return v[0];
  int m = sel == 0u ? 3 : 2;
  if (v[0] > v[1]) return K(K(K(v[0] * m) + v[1]) + v[2]) % 10007;
  return K(K(K(K(v[1] * v[1]) % 10007 + v[0]) + v[1]) + v[2]) % 10007;
}
HW_FN A root(int a, int b, HW_TH O& o) { return a3(a, b % 1009, 0); }
#endif

#if PROGRAM == 1
#define RB 2
HW_FN int w(int x, HW_TH O& o) { int s = x % 101; int i = 0; while (i < x % 6) { s = K(K(s * 8) + i) % 10007; i = K(i + 1); } return s; }
HW_FN N node(A a, HW_TH O& o) {
  int n = a.x, s = a.y;
  if (n <= 0) return leaf_of(w(s, o));
  int sel = K(K(s * 4) + K(n * 3)) % 3;
  if (sel == 0) return fork_of(0u, 3u, a3(K(n - 1), K(K(s * 18) + 1) % 1009, 0), a3(K(n - 1), K(K(s * 21) + 49) % 1009, 0), a3(K(n - 2), K(K(s * 27) + 25) % 1009, 0));
  if (sel == 1) return fork_of(9u, 1u, a3(K(n - 1), K(K(s * 27) + 47) % 1009, 0), a3(0, 0, 0), a3(0, 0, 0));
  return fork_of(9u, 1u, a3(K(n - 1), K(K(s * 22) + 80) % 1009, 0), a3(0, 0, 0), a3(0, 0, 0));
}
HW_FN bool extra(A a, uint sel, HW_TH int* v, HW_TH A& e, HW_TH O& o) { return false; }
HW_FN int combine(A a, uint sel, HW_TH int* v, int e, HW_TH O& o) {
  if (sel == 9u) return v[0];
  return K(K(K(K(v[0] * v[1]) + K(v[0] * 5)) + v[1]) + v[2]) % 10007;
}
HW_FN A root(int a, int b, HW_TH O& o) { return a3(a, b % 1009, 0); }
#endif

#if PROGRAM == 2
#define RB 2
HW_FN int w(int x, HW_TH O& o) { if (x % 2 == 0) return K(x / 2 + 9) % 10007; return K(K(x * 3) + 1) % 10007; }
HW_FN N node(A a, HW_TH O& o) {
  int n = a.x, s = a.y;
  if (n <= 0) return leaf_of(w(s, o));
  int sel = K(K(s * 22) + K(n * 7)) % 3;
  if (sel == 0) return fork_of(0u, 3u, a3(K(n - 1), K(K(s * 23) + 50) % 1009, 0), a3(K(n - 2), K(K(s * 12) + 8) % 1009, 0), a3(K(n - 1), K(K(s * 16) + 88) % 1009, 0));
  if (sel == 1) return fork_of(9u, 1u, a3(K(n - 1), K(K(s * 3) + 7) % 1009, 0), a3(0, 0, 0), a3(0, 0, 0));
  return fork_of(9u, 1u, a3(K(n - 1), K(K(s * 21) + 44) % 1009, 0), a3(0, 0, 0), a3(0, 0, 0));
}
HW_FN bool extra(A a, uint sel, HW_TH int* v, HW_TH A& e, HW_TH O& o) {
  if (sel != 0u || K(v[0] + v[1]) % 2 != 0) return false;
  e = a3(K(a.x - 2), K(K(a.y * 16) + 84) % 1009, 0); return true;
}
HW_FN int combine(A a, uint sel, HW_TH int* v, int e, HW_TH O& o) {
  if (sel == 9u) return v[0];
  if (K(v[0] + v[1]) % 2 == 0) return K(K(K(e * v[0]) + v[1]) + v[2]) % 10007;
  return K(K(K(v[0] * 4) + v[1]) + v[2]) % 10007;
}
HW_FN A root(int a, int b, HW_TH O& o) { return a3(a, b % 1009, 0); }
#endif

#if PROGRAM == 4
#define RB 1
HW_FN int w(int x, HW_TH O& o) { if (x % 2 == 0) return K(x / 2 + 19) % 10007; return K(K(x * 3) + 1) % 10007; }
HW_FN N node(A a, HW_TH O& o) {
  int n = a.x, s = a.y;
  if (n <= 0) return leaf_of(w(s, o));
  int sel = K(K(s * 10) + K(n * 6)) % 2;
  if (sel == 0) return fork_of(0u, 2u, a3(K(n - 1), K(K(s * 26) + 41) % 1009, 0), a3(K(n - 1), K(K(s * 29) + 12) % 1009, 0), a3(0, 0, 0));
  return leaf_of(K(w(s, o) + n) % 10007);
}
HW_FN bool extra(A a, uint sel, HW_TH int* v, HW_TH A& e, HW_TH O& o) {
  if (K(v[0] + v[1]) % 5 != 0) return false;
  e = a3(K(a.x - 2), K(K(a.y * 3) + 69) % 1009, 0); return true;
}
HW_FN int combine(A a, uint sel, HW_TH int* v, int e, HW_TH O& o) {
  if (K(v[0] + v[1]) % 5 == 0) return K(K(e * v[0]) + v[1]) % 10007;
  return K(K(v[0] * 9) + v[1]) % 10007;
}
HW_FN A root(int a, int b, HW_TH O& o) { return a3(a, b % 1009, 0); }
#endif

#if PROGRAM == 5
#define RB 2
HW_FN int w(int x, HW_TH O& o) { int s = x % 101; int i = 0; while (i < x % 9) { s = K(K(s * 11) + i) % 10007; i = K(i + 1); } return s; }
HW_FN N node(A a, HW_TH O& o) {
  int n = a.x, s = a.y;
  if (n <= 0) return leaf_of(w(s, o));
  int sel = K(K(s * 4) + K(n * 7)) % 3;
  if (sel == 0) return fork_of(0u, 2u, a3(K(n - 2), K(K(s * 15) + 50) % 1009, 0), a3(K(n - 2), K(K(s * 5) + 30) % 1009, 0), a3(0, 0, 0));
  if (sel == 1) return fork_of(1u, 3u, a3(K(n - 1), K(K(s * 6) + 67) % 1009, 0), a3(K(n - 2), K(K(s * 29) + 52) % 1009, 0), a3(K(n - 2), K(K(s * 19) + 32) % 1009, 0));
  return leaf_of(K(w(s, o) + n) % 10007);
}
HW_FN bool extra(A a, uint sel, HW_TH int* v, HW_TH A& e, HW_TH O& o) {
  if (K(v[0] + v[1]) % 2 != 0) return false;
  e = sel == 0u ? a3(K(a.x - 2), K(K(a.y * 16) + 82) % 1009, 0) : a3(K(a.x - 2), K(K(a.y * 20) + 65) % 1009, 0);
  return true;
}
HW_FN int combine(A a, uint sel, HW_TH int* v, int e, HW_TH O& o) {
  bool even = K(v[0] + v[1]) % 2 == 0;
  if (sel == 0u) return even ? K(K(e * v[0]) + v[1]) % 10007 : K(K(v[0] * 8) + v[1]) % 10007;
  return even ? K(K(K(e * v[0]) + v[1]) + v[2]) % 10007 : K(K(K(v[0] * 3) + v[1]) + v[2]) % 10007;
}
HW_FN A root(int a, int b, HW_TH O& o) { return a3(a, b % 1009, 0); }
#endif

#if PROGRAM == 6
#define RB 2
HW_FN int w(int x, HW_TH O& o) { if (x % 2 == 0) return K(x / 2 + 26) % 10007; return K(K(x * 3) + 1) % 10007; }
HW_FN N node(A a, HW_TH O& o) {
  int n = a.x, s = a.y;
  if (n <= 0) return leaf_of(w(s, o));
  int sel = K(K(s * 5) + K(n * 3)) % 4;
  if (sel == 0) return fork_of(0u, 2u, a3(K(n - 1), K(K(s * 29) + 6) % 1009, 0), a3(K(n - 2), K(K(s * 18) + 54) % 1009, 0), a3(0, 0, 0));
  if (sel == 1) return fork_of(1u, 3u, a3(K(n - 1), K(K(s * 19) + 85) % 1009, 0), a3(K(n - 2), K(K(s * 2) + 97) % 1009, 0), a3(K(n - 1), K(K(s * 22) + 54) % 1009, 0));
  if (sel == 2) return fork_of(2u, 3u, a3(K(n - 1), K(K(s * 16) + 84) % 1009, 0), a3(K(n - 1), K(K(s * 21) + 89) % 1009, 0), a3(K(n - 2), K(K(s * 7) + 48) % 1009, 0));
  return fork_of(9u, 1u, a3(K(n - 1), K(K(s * 21) + 43) % 1009, 0), a3(0, 0, 0), a3(0, 0, 0));
}
HW_FN bool extra(A a, uint sel, HW_TH int* v, HW_TH A& e, HW_TH O& o) { return false; }
HW_FN int combine(A a, uint sel, HW_TH int* v, int e, HW_TH O& o) {
  if (sel == 9u) return v[0];
  if (sel == 0u) return K(K(K(v[0] * v[1]) + K(v[0] * 5)) + v[1]) % 10007;
  int m = sel == 1u ? 9 : 6;
  return K(K(K(K(v[0] * v[1]) + K(v[0] * m)) + v[1]) + v[2]) % 10007;
}
HW_FN A root(int a, int b, HW_TH O& o) { return a3(a, b % 1009, 0); }
#endif

#if PROGRAM == 3 || PROGRAM == 7 || PROGRAM == 8 || PROGRAM == 9 || PROGRAM == 10 || PROGRAM == 11
// dc family.
#if PROGRAM == 3
#define RB 2
#define BASE 6
#define LO0 6
HW_FN int leaf(int x, HW_TH O& o) {
  int v = K(x % 997 + 1); int k = 0;
  while (v != 1) { if (k == 38) return k; if (v % 2 == 0) v = v / 2; else v = K(K(v * 3) + 1); k = K(k + 1); }
  return k;
}
HW_FN int small_leaf(int lo, int hi, HW_TH O& o) { return leaf(K(K(lo * 6) + hi), o); }
HW_FN int deep_leaf(int lo, int hi, HW_TH O& o) { return leaf(K(K(hi - lo) + 1), o); }
HW_FN N split(int lo, int hi, int d, HW_TH O& o) {
  int m1 = K(K(lo + 1) + K(K(K(hi - lo) - 2) * K(3 + K(lo % 3 + 3) % 3)) / 6);
  int m2 = K(K(m1 + 1) + K(K(K(hi - m1) - 2) * 3) / 7);
  return fork_of(0u, 3u, a3(lo, m1, K(d - 1)), a3(m1, m2, K(d - 1)), a3(m2, hi, K(d - 2)));
}
HW_FN bool extra(A a, uint sel, HW_TH int* v, HW_TH A& e, HW_TH O& o) { return false; }
HW_FN int combine(A a, uint sel, HW_TH int* v, int e, HW_TH O& o) {
  if (v[0] > v[1]) return K(K(K(v[0] * 8) + v[1]) + v[2]) % 10007;
  return K(K(K(K(v[1] * v[1]) % 10007 + v[0]) + v[1]) + v[2]) % 10007;
}
#endif
#if PROGRAM == 7
#define RB 1
#define BASE 5
#define LO0 40
HW_FN int leaf(int x, HW_TH O& o) { int s = x % 101; int i = 0; while (i < x % 4) { s = K(K(s * 13) + i) % 10007; i = K(i + 1); } return s; }
HW_FN int small_leaf(int lo, int hi, HW_TH O& o) { return leaf(K(K(lo * 8) + hi), o); }
HW_FN int deep_leaf(int lo, int hi, HW_TH O& o) { return leaf(K(K(hi - lo) + 9), o); }
HW_FN N split(int lo, int hi, int d, HW_TH O& o) {
  int m1 = K(K(lo + 1) + K(K(K(hi - lo) - 2) * K(2 + K(lo % 1 + 1) % 1)) / 7);
  return fork_of(0u, 2u, a3(lo, m1, K(d - 1)), a3(m1, hi, K(d - 1)), a3(0, 0, 0));
}
HW_FN bool extra(A a, uint sel, HW_TH int* v, HW_TH A& e, HW_TH O& o) { return false; }
HW_FN int combine(A a, uint sel, HW_TH int* v, int e, HW_TH O& o) {
  return K(K(K(v[0] * v[1]) + K(v[0] * 5)) + v[1]) % 10007;
}
#endif
#if PROGRAM == 8
#define RB 2
#define BASE 2
#define LO0 10
HW_FN int leaf(int x, HW_TH O& o) { int s = x % 101; int i = 0; while (i < x % 3) { s = K(K(s * 8) + i) % 10007; i = K(i + 1); } return s; }
HW_FN int small_leaf(int lo, int hi, HW_TH O& o) { return leaf(K(K(lo * 5) + hi), o); }
HW_FN int deep_leaf(int lo, int hi, HW_TH O& o) { return leaf(K(K(hi - lo) + 2), o); }
HW_FN N split(int lo, int hi, int d, HW_TH O& o) {
  int m1 = K(K(lo + 1) + K(K(K(hi - lo) - 2) * K(3 + K(lo % 1 + 1) % 1)) / 5);
  int m2 = K(K(m1 + 1) + K(K(K(hi - m1) - 2) * 1) / 6);
  return fork_of(0u, 3u, a3(lo, m1, K(d - 1)), a3(m1, m2, K(d - 1)), a3(m2, hi, K(d - 1)));
}
HW_FN bool extra(A a, uint sel, HW_TH int* v, HW_TH A& e, HW_TH O& o) {
  if (K(v[0] + v[1]) % 2 != 0) return false;
  int m1 = K(K(a.x + 1) + K(K(K(a.y - a.x) - 2) * K(3 + K(a.x % 1 + 1) % 1)) / 5);
  e = a3(a.x, m1, K(a.z - 2)); return true;
}
HW_FN int combine(A a, uint sel, HW_TH int* v, int e, HW_TH O& o) {
  if (K(v[0] + v[1]) % 2 == 0) return K(K(K(e * v[0]) + v[1]) + v[2]) % 10007;
  return K(K(K(v[0] * 6) + v[1]) + v[2]) % 10007;
}
#endif
#if PROGRAM == 9
#define RB 1
#define BASE 2
#define LO0 12
HW_FN int leaf(int x, HW_TH O& o) { int s = x % 101; int i = 0; while (i < x % 16) { s = K(K(s * 12) + i) % 10007; i = K(i + 1); } return s; }
HW_FN int small_leaf(int lo, int hi, HW_TH O& o) { return leaf(K(K(lo * 2) + hi), o); }
HW_FN int deep_leaf(int lo, int hi, HW_TH O& o) { return leaf(K(K(hi - lo) + 1), o); }
HW_FN N split(int lo, int hi, int d, HW_TH O& o) {
  int m1 = K(K(lo + 1) + K(K(K(hi - lo) - 2) * K(1 + K(lo % 2 + 2) % 2)) / 5);
  return fork_of(0u, 2u, a3(lo, m1, K(d - 1)), a3(m1, hi, K(d - 1)), a3(0, 0, 0));
}
HW_FN bool extra(A a, uint sel, HW_TH int* v, HW_TH A& e, HW_TH O& o) {
  if (K(v[0] + v[1]) % 4 != 0) return false;
  int m1 = K(K(a.x + 1) + K(K(K(a.y - a.x) - 2) * K(1 + K(a.x % 2 + 2) % 2)) / 5);
  e = a3(a.x, m1, K(a.z - 2)); return true;
}
HW_FN int combine(A a, uint sel, HW_TH int* v, int e, HW_TH O& o) {
  if (K(v[0] + v[1]) % 4 == 0) return K(K(e * v[0]) + v[1]) % 10007;
  return K(K(v[0] * 9) + v[1]) % 10007;
}
#endif
#if PROGRAM == 10
#define RB 2
#define BASE 4
#define LO0 24
HW_FN int leaf(int x, HW_TH O& o) { int s = x % 101; int i = 0; while (i < x % 9) { s = K(K(s * 3) + i) % 10007; i = K(i + 1); } return s; }
HW_FN int small_leaf(int lo, int hi, HW_TH O& o) { return leaf(K(K(lo * 7) + hi), o); }
HW_FN int deep_leaf(int lo, int hi, HW_TH O& o) { return leaf(K(K(hi - lo) + 6), o); }
HW_FN N split(int lo, int hi, int d, HW_TH O& o) {
  int m1 = K(K(lo + 1) + K(K(K(hi - lo) - 2) * K(1 + K(lo % 3 + 3) % 3)) / 5);
  int m2 = K(K(m1 + 1) + K(K(K(hi - m1) - 2) * 2) / 5);
  return fork_of(0u, 3u, a3(lo, m1, K(d - 1)), a3(m1, m2, K(d - 1)), a3(m2, hi, K(d - 1)));
}
HW_FN bool extra(A a, uint sel, HW_TH int* v, HW_TH A& e, HW_TH O& o) { return false; }
HW_FN int combine(A a, uint sel, HW_TH int* v, int e, HW_TH O& o) {
  if (v[0] > v[1]) return K(K(K(v[0] * 3) + v[1]) + v[2]) % 10007;
  return K(K(K(K(v[1] * v[1]) % 10007 + v[0]) + v[1]) + v[2]) % 10007;
}
#endif
#if PROGRAM == 11
#define RB 2
#define BASE 6
#define LO0 28
HW_FN int leaf(int x, HW_TH O& o) { if (x % 2 == 0) return K(x / 2 + 49) % 10007; return K(K(x * 3) + 1) % 10007; }
HW_FN int small_leaf(int lo, int hi, HW_TH O& o) { return leaf(K(K(lo * 9) + hi), o); }
HW_FN int deep_leaf(int lo, int hi, HW_TH O& o) { return leaf(K(K(hi - lo) + 6), o); }
HW_FN N split(int lo, int hi, int d, HW_TH O& o) {
  int m1 = K(K(lo + 1) + K(K(K(hi - lo) - 2) * K(4 + K(lo % 2 + 2) % 2)) / 6);
  int m2 = K(K(m1 + 1) + K(K(K(hi - m1) - 2) * 4) / 8);
  return fork_of(0u, 3u, a3(lo, m1, K(d - 1)), a3(m1, m2, K(d - 1)), a3(m2, hi, K(d - 2)));
}
HW_FN bool extra(A a, uint sel, HW_TH int* v, HW_TH A& e, HW_TH O& o) { return false; }
HW_FN int combine(A a, uint sel, HW_TH int* v, int e, HW_TH O& o) {
  if (v[0] > v[1]) return K(K(K(v[0] * 3) + v[1]) + v[2]) % 10007;
  return K(K(K(K(v[1] * v[1]) % 10007 + v[0]) + v[1]) + v[2]) % 10007;
}
#endif
HW_FN N node(A a, HW_TH O& o) {
  int lo = a.x, hi = a.y, d = a.z;
  if (K(hi - lo) <= BASE) return leaf_of(small_leaf(lo, hi, o));
  if (d <= 0) return leaf_of(deep_leaf(lo, hi, o));
  return split(lo, hi, d, o);
}
HW_FN A root(int a, int b, HW_TH O& o) { return a3(LO0, K(LO0 + a), b); }
#endif

#define R (1u << RB)

// ---- checked serial frame machine (below the replay depth) -------------
// `base` is the oracle call depth of the starting frame (`run` is depth 1).
struct F { A a; uint sel, k, next; A c[3]; int v[3]; };

HW_FN O serial(A start, uint base) {
  F f[64];
  O o; o.v = 0; o.trap = 0u;
  uint sp = 0u;
  f[0].a = start; f[0].next = 0u;
  int r = 0;
  for (;;) {
    if (f[sp].next == 0u) {
      N n = node(f[sp].a, o);
      if (o.trap != 0u) return o;
      if (n.leaf != 0u) { r = n.v; }
      else {
        f[sp].sel = n.sel; f[sp].k = n.k; f[sp].c[0] = n.c[0]; f[sp].c[1] = n.c[1]; f[sp].c[2] = n.c[2];
        f[sp].next = 1u;
        if (base + sp >= 64u) { o.trap = 4u; return o; }
        f[sp + 1u].a = n.c[0]; f[sp + 1u].next = 0u; ++sp; continue;
      }
    }
    // r is the result of frame sp.
    for (;;) {
      if (sp == 0u) { o.v = r; return o; }
      --sp;
      uint j = f[sp].next - 1u;
      if (j < f[sp].k) f[sp].v[j] = r;
      if (f[sp].next < f[sp].k) {
        if (base + sp >= 64u) { o.trap = 4u; return o; }
        f[sp + 1u].a = f[sp].c[f[sp].next]; f[sp + 1u].next = 0u; f[sp].next += 1u; ++sp; break;
      }
      if (f[sp].next == f[sp].k) {
        A e;
        bool more = extra(f[sp].a, f[sp].sel, f[sp].v, e, o);
        if (o.trap != 0u) return o;
        if (more) {
          f[sp].next += 1u;
          if (base + sp >= 64u) { o.trap = 4u; return o; }
          f[sp + 1u].a = e; f[sp + 1u].next = 0u; ++sp; break;
        }
        r = combine(f[sp].a, f[sp].sel, f[sp].v, 0, o);
      } else {
        r = combine(f[sp].a, f[sp].sel, f[sp].v, r, o);
      }
      if (o.trap != 0u) return o;
    }
  }
}

// Join a fork node (oracle call depth `depth`) from its child observations
// (first trapped child wins), running an extra call serially below it.
HW_FN O join(A a, uint sel, uint k, HW_TH O* c, uint depth) {
  O o; o.v = 0; o.trap = 0u;
  for (uint j = 0u; j < k; ++j) if (c[j].trap != 0u) return c[j];
  int v[3] = {c[0].v, k > 1u ? c[1].v : 0, k > 2u ? c[2].v : 0};
  A e;
  int ev = 0;
  if (extra(a, sel, v, e, o)) {
    if (o.trap != 0u) return o;
    O x = serial(e, depth + 1u);
    if (x.trap != 0u) return x;
    ev = x.v;
  }
  if (o.trap != 0u) return o;
  o.v = combine(a, sel, v, ev, o);
  return o;
}

// Per-lane path record of the replay.
struct P { A a[16]; uint sel[16]; uint k[16]; };

HW_FN O replay(int pa, int pb, uint D, uint gid, HW_TH uint& stop, HW_TH uint& valid, HW_TH P& p) {
  O o; o.v = 0; o.trap = 0u;
  A a = root(pa, pb, o);
  stop = 0u; valid = 1u;
  if (o.trap != 0u) return o;
  for (uint depth = 0u; depth < D; ++depth) {
    p.a[depth] = a;
    N n = node(a, o);
    if (o.trap != 0u) { stop = depth; return o; }
    if (n.leaf != 0u) { o.v = n.v; stop = depth; return o; }
    p.sel[depth] = n.sel; p.k[depth] = n.k;
    uint digit = (gid >> ((D - 1u - depth) * RB)) & (R - 1u);
    if (digit >= n.k) { valid = 0u; stop = depth + 1u; return o; }
    a = n.c[digit];
  }
  stop = D;
  return serial(a, D + 2u);
}

// ---- kernels -------------------------------------------------------------
// Buffers: 0 params {a, b, D, groups}, 1 part[O], 2 stops[2 per group],
// 3 paths[P per group], 4 out[O]. 256 lanes per group.
#define LEVELS_IN_GROUP (8u / RB)

HW_KERNEL_B {
  HW_IDS
  HW_SHARED O v[256];
  HW_SHARED uint stop[256];
  HW_SHARED uint valid[256];
  uint D = (uint)params[2];
  uint gid = GROUP * 256u + TID;
  P p;
  uint st = 0u, va = 0u;
  O r; r.v = 0; r.trap = 0u;
  if (gid < (1u << (D * RB))) r = replay(params[0], params[1], D, gid, st, va, p);
  v[TID] = r; stop[TID] = st; valid[TID] = va;
  HW_BARRIER;
  uint levels = D < LEVELS_IN_GROUP ? D : LEVELS_IN_GROUP;
  uint stride = 1u;
  for (uint level = 0u; level < levels; ++level, stride *= R) {
    uint depth = D - 1u - level;
    if (TID % (stride * R) == 0u && valid[TID] != 0u && stop[TID] > depth) {
      O c[3];
      for (uint j = 0u; j < p.k[depth]; ++j) c[j] = v[TID + j * stride];
      v[TID] = join(p.a[depth], p.sel[depth], p.k[depth], c, depth + 2u);
    }
    HW_BARRIER;
  }
  if (TID == 0u) {
    part[GROUP] = v[0];
    stops[2u * GROUP] = stop[0];
    stops[2u * GROUP + 1u] = valid[0];
    paths[GROUP] = p;
  }
}

HW_KERNEL_TOP {
  HW_IDS
  HW_SHARED O v[256];
  uint D = (uint)params[2];
  uint groups = (uint)params[3];
  uint levels = D > LEVELS_IN_GROUP ? D - LEVELS_IN_GROUP : 0u;
  if (TID < groups) v[TID] = part[TID];
  HW_BARRIER;
  uint stride = 1u;
  for (uint level = 0u; level < levels; ++level, stride *= R) {
    uint depth = levels - 1u - level;
    if (TID < groups && TID % (stride * R) == 0u && stops[2u * TID + 1u] != 0u && stops[2u * TID] > depth) {
      O c[3];
      uint k = paths[TID].k[depth];
      for (uint j = 0u; j < k; ++j) c[j] = v[TID + j * stride];
      v[TID] = join(paths[TID].a[depth], paths[TID].sel[depth], k, c, depth + 2u);
    }
    HW_BARRIER;
  }
  if (TID == 0u) out[0] = v[0];
}
