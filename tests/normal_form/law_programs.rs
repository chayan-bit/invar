//! The P1b law programs: one source per law case, shared by the law harness and the `invar run` test.

pub const L1_MAP: &str = "fn f(xs: read Array<i32>, ys: read Array<i32>, n: u32) -> Array<i32> { var out = filled_i32(n, 0); for i in 0u32 .. n { out[i] = xs[i] / ys[i]; out[i] = out[i] + 1; } return move out; }";
pub const L1_STRIDE: &str = "fn f(xs: read Array<i32>, n: u32, k: u32) -> Array<i32> { var out = filled_i32(n * 3u32 + 1u32, 0); for i in 0u32 .. n { out[i * k] = out[i * k] + xs[i]; } return move out; }";
pub const L2_SEALED: &str = "fn f(xs: read Array<u32>, n: u32) -> u32 { var s = 7u32; var p = 3u32; for i in 0u32 .. n { s = wrapping_add(s, xs[i]); p = wrapping_mul(p, xs[i]); } return wrapping_add(s, p); }";
pub const L3_SUM: &str = "fn f(xs: read Array<i32>, n: u32, s0: i32) -> i32 { var s = s0; for i in 0u32 .. n { s = s + xs[i]; } return s; }";
pub const L3_PRODUCT: &str = "fn f(xs: read Array<i32>, n: u32, s0: i32) -> i32 { var p = s0; for i in 0u32 .. n { p = p * xs[i]; } return p; }";
pub const L4_CHECKED: &str = "fn f(xs: read Array<i32>, ys: read Array<i32>, n: u32, s0: i32) -> i32 { var s = s0; for i in 0u32 .. n { s = s * xs[i] + ys[i]; } return s; }";
pub const L4_WRAPPING: &str = "fn f(xs: read Array<u32>, ys: read Array<u32>, n: u32, s0: u32) -> u32 { var s = s0; for i in 0u32 .. n { s = wrapping_add(wrapping_mul(s, xs[i]), ys[i]); } return s; }";
pub const L4_RESIDUE: &str = "fn f(xs: read Array<i32>, ys: read Array<i32>, n: u32, s0: i32) -> i32 { var s = s0; for i in 0u32 .. n { s = (s * xs[i] + ys[i]) % 10007; } return s; }";
pub const L4_WHILE: &str = "fn w(x: i32) -> i32 { var s = x % 101; var i = 0; while i < x % 11 { s = (s * 2 + i * x) % 10007; i = i + 1; } return s; }";
pub const L5_RESIDUE: &str = "fn f(cs: read Array<i32>, n: u32, a0: i32, b0: i32) -> i32 { var a = a0; var b = b0; for i in 0u32 .. n { let na = (a * 2 + b * cs[i]) % 10007; let nb = (a + b) % 10007; a = na; b = nb; } return a * 3 + b; }";
pub const L5_CHECKED: &str = "fn f(cs: read Array<i32>, n: u32, a0: i32, b0: i32) -> i32 { var a = a0; var b = b0; for i in 0u32 .. n { let na = a + b * cs[i]; let nb = a - b; a = na; b = nb; } return a - b; }";
pub const L6_KADANE: &str = "fn f(xs: read Array<i32>, n: u32) -> i32 { var best = 0 - 2147483647; var cur = 0; for i in 0u32 .. n { cur = cur + xs[i]; if cur < 0 { cur = 0; } if cur > best { best = cur; } } return best; }";
pub const L7_ARGMIN: &str = "fn f(xs: read Array<f32>, n: u32, m0: f32) -> u32 { var m = m0; var at = 4294967295u32; for i in 0u32 .. n { if xs[i] < m { m = xs[i]; at = i; } } if m != m { return at + 1u32; } return at; }";
pub const L7_ARGMAX: &str = "fn f(xs: read Array<i32>, n: u32) -> i32 { var m = 0 - 1000; var at = 0u32; for i in 0u32 .. n { if xs[i] >= m { m = xs[i]; at = i; } } if at % 2u32 == 0u32 { return m; } return m + 1; }";
pub const L9_PATHSUM: &str = "fn leaf(x: i32) -> i32 { if x % 2 == 0 { return (x / 2 + 16) % 10007; } return (x * 3 + 1) % 10007; } fn dc(lo: i32, hi: i32, d: i32) -> i32 { if hi - lo <= 5 { return leaf(lo * 4 + hi); } if d <= 0 { return leaf(hi - lo + 7); } let m1 = lo + 1 + ((hi - lo - 2) * 3) / 9; let m2 = m1 + 1 + ((hi - m1 - 2) * 3) / 6; let a = dc(lo, m1, d - 1); let b = dc(m1, m2, d - 1); let c = dc(m2, hi, d - 1); return (a * 5 + b + c * 2) % 100003; } fn run(lo: i32, a: i32, b: i32) -> i32 { return dc(lo, lo + a, b); }";
pub const L9_TAIL: &str = "fn w(x: i32) -> i32 { return (x * 7 + 3) % 1009; } fn t(n: i32, s: i32) -> i32 { if n <= 0 { return w(s); } let sel = (s * 29 + n * 6) % 4; if sel == 0 { let v0 = t(n - 1, (s * 14 + 60) % 1009); let v1 = t(n - 1, (s * 12 + 66) % 1009); return (v0 * 7 + v1 + s) % 10007; } if sel == 1 { return (w(s) + n) % 10007; } return t(n - 1, (s * 4 + 22) % 1009); }";
pub const L10_RANGE: &str = "fn leaf(x: i32) -> i32 { if x % 2 == 0 { return x * x - 7; } return x * x * 3 + 1; } fn sum(lo: i32, hi: i32) -> i32 { if hi - lo <= 1 { return leaf(lo); } let m = lo + (hi - lo) / 2; let a = sum(lo, m); let b = sum(m, hi); return a + b; }";
pub const L11_CHECKED: &str = "fn f(h: mut Array<i32>, ix: read Array<u32>, w: read Array<i32>, n: u32, m: u32) -> Unit { for i in 0u32 .. n { h[ix[i] % m] = h[ix[i] % m] + w[i]; } }";
pub const L11_WRAPPING: &str = "fn f(h: mut Array<u32>, ix: read Array<u32>, w: read Array<u32>, n: u32, m: u32) -> Unit { for i in 0u32 .. n { h[ix[i] % m] = wrapping_add(h[ix[i] % m], w[i]); } }";
pub const L12_ORDERED: &str = "fn f(out: mut Array<i32>, ix: read Array<u32>, xs: read Array<i32>, n: u32) -> i32 { var last = 0 - 1; for i in 0u32 .. n { out[ix[i]] = xs[i] * 2; if xs[i] > 0 { last = xs[i]; } } return last; }";
pub const L13_COMPACT: &str = "fn f(out: mut Array<i32>, xs: read Array<i32>, n: u32, t: i32) -> u32 { var c = 0u32; for i in 0u32 .. n { if xs[i] > t { out[c] = xs[i]; c = c + 1u32; } } return c; }";
pub const L14_FRONTIER: &str = "fn succ(v: u32, k: u32, n: u32) -> u32 { return (v * 13u32 + k * 2u32 + 5u32) % n; } fn expand_round(frontier: read Array<u32>, seen: mut Array<u32>, next: mut Array<u32>, n: u32, c: u32) -> i32 { var count = 0; for u in 0u32 .. n { if frontier[u] == 1u32 and seen[u] == 0u32 { seen[u] = 1u32; count = count + 1; let degree = u % 4u32 + 1u32; for k in 0u32 .. degree { if seen[succ(u, k * c, n)] == 0u32 { next[succ(u, k * c, n)] = 1u32; } } } } return count; }";
pub const L15_FIRST: &str = "fn f(xs: read Array<i32>, n: u32, key: i32) -> u32 { for i in 0u32 .. n { if 100 / xs[i] == key { return i; } } return n; }";
pub const L15_SCAN: &str = "fn f(xs: read Array<i32>, n: u32, key: i32) -> u32 { var s = 0; for i in 0u32 .. n { s = s + xs[i]; if s > key { return i; } } return n + 1u32; }";
pub const L19_FUSED: &str = "fn f(xs: read Array<i32>, n: u32) -> Array<i32> { var a = filled_i32(n, 0); for i in 0u32 .. n { a[i] = xs[i] * 3; } var b = filled_i32(n, 0); for j in 0u32 .. n { b[j] = a[j] + xs[j] / 2; } return move b; }";
pub const L20_FORK: &str = "fn g(x: i32) -> i32 { return 1000 / x + x * x; } fn h(x: i32, y: i32) -> i32 { return x * y - 3; } fn f(a: i32, b: i32) -> i32 { let p = g(a); let q = h(a, b); let r = g(b); return p + q + r; }";

/// Every law program as (`L<id>` label, source), in declaration order.
pub const LAW_PROGRAMS: &[(&str, &str)] = &[
    ("L1_MAP", L1_MAP),
    ("L1_STRIDE", L1_STRIDE),
    ("L2_SEALED", L2_SEALED),
    ("L3_SUM", L3_SUM),
    ("L3_PRODUCT", L3_PRODUCT),
    ("L4_CHECKED", L4_CHECKED),
    ("L4_WRAPPING", L4_WRAPPING),
    ("L4_RESIDUE", L4_RESIDUE),
    ("L4_WHILE", L4_WHILE),
    ("L5_RESIDUE", L5_RESIDUE),
    ("L5_CHECKED", L5_CHECKED),
    ("L6_KADANE", L6_KADANE),
    ("L7_ARGMIN", L7_ARGMIN),
    ("L7_ARGMAX", L7_ARGMAX),
    ("L9_PATHSUM", L9_PATHSUM),
    ("L9_TAIL", L9_TAIL),
    ("L10_RANGE", L10_RANGE),
    ("L11_CHECKED", L11_CHECKED),
    ("L11_WRAPPING", L11_WRAPPING),
    ("L12_ORDERED", L12_ORDERED),
    ("L13_COMPACT", L13_COMPACT),
    ("L14_FRONTIER", L14_FRONTIER),
    ("L15_FIRST", L15_FIRST),
    ("L15_SCAN", L15_SCAN),
    ("L19_FUSED", L19_FUSED),
    ("L20_FORK", L20_FORK),
];
