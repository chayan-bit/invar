//! Native CPU probes: the placement map, then M5-01 probes for the nine
//! frozen-table families beyond `map`.
//! Each kernel is ordinary source run by the native serial engine, and every
//! timed call's output must equal an independent Rust reference before its
//! time is kept. Rows use their own family IDs, so no auto decision prices
//! them; GNET-04 owns any policy built on them.
use super::*;

/// Every probe exports `k(xs, a, b, out)`; `a` and `b` carry its shape.
const HEAD: &str = "fn k(xs: read Array<u32>, a: u32, b: u32, out: mut Array<u32>) -> u32 ";
const ROUNDS: u32 = 16;
/// Bounded, and far above the largest probe (128^3 multiply-adds).
pub(super) const FUEL: u64 = 1 << 40;

pub(super) struct Family {
    pub table: &'static str,
    body: &'static str,
    /// Largest input; every size is a square so grids stay exact.
    pub large: usize,
}
pub(super) const FAMILIES_BEYOND_MAP: [Family; 9] = [
    Family {
        table: "reduction",
        body: "{ var acc = 17u32; for i in 0u32..len(xs) { acc = wrapping_add(acc, xs[i]); } out[0u32] = acc; return acc; }",
        large: 262144,
    },
    Family {
        table: "scan",
        body: "{ var s = 17u32; for i in 0u32..len(xs) { s = wrapping_add(wrapping_mul(s, 1664525u32), xs[i]); out[i] = s; } return s; }",
        large: 262144,
    },
    Family {
        table: "compaction",
        body: "{ var pos = 0u32; for i in 0u32..len(xs) { if xs[i] < 2147483648u32 { out[pos] = xs[i]; pos = pos + 1u32; } } return pos; }",
        large: 262144,
    },
    Family {
        table: "transpose",
        body: "{ for r in 0u32..a { for c in 0u32..b { out[c * a + r] = xs[r * b + c]; } } return a; }",
        large: 262144,
    },
    Family {
        table: "stencil",
        body: "{ for r in 0u32..a { for c in 0u32..b { var v = xs[r * b + c]; if r > 0u32 { v = wrapping_add(v, xs[(r - 1u32) * b + c]); } if r + 1u32 < a { v = wrapping_add(v, xs[(r + 1u32) * b + c]); } if c > 0u32 { v = wrapping_add(v, xs[r * b + c - 1u32]); } if c + 1u32 < b { v = wrapping_add(v, xs[r * b + c + 1u32]); } out[r * b + c] = v; } } return a; }",
        large: 262144,
    },
    Family {
        table: "matrix",
        body: "{ for i in 0u32..a { for j in 0u32..a { var s = 0u32; for q in 0u32..a { s = wrapping_add(s, wrapping_mul(xs[i * a + q], xs[q * a + j])); } out[i * a + j] = s; } } return a; }",
        large: 16384,
    },
    Family {
        table: "segmented",
        body: "{ for s in 0u32..a { var acc = 0u32; for i in s * 16u32..s * 16u32 + 16u32 { acc = wrapping_add(acc, xs[i]); } out[s] = acc; } return a; }",
        large: 262144,
    },
    Family {
        table: "gather_scatter",
        body: "{ for i in 0u32..len(xs) { out[i] = xs[wrapping_mul(i, 2654435761u32) % len(xs)]; } return len(xs); }",
        large: 262144,
    },
    Family {
        table: "frontier",
        body: "{ if a > 0u32 { out[a] = 1u32; } for step in 0u32..b { for u in 0u32..a { if out[a + u] == 1u32 and out[u] == 0u32 { out[u] = step + 1u32; for w in 0u32..a { if wrapping_mul(xs[u * a + w], 2654435761u32) / 134217728u32 == 0u32 and out[w] == 0u32 { out[2u32 * a + w] = 1u32; } } } } for c in 0u32..a { out[a + c] = out[2u32 * a + c]; out[2u32 * a + c] = 0u32; } } return b; }",
        large: 65536,
    },
];

pub(super) fn input(n: usize) -> Vec<u32> {
    (0..n as u32)
        .map(|i| i.wrapping_mul(1_664_525).wrapping_add(1_013_904_223))
        .collect()
}
impl Family {
    /// The profile family ID registered for this table family.
    pub(super) fn id(&self) -> &'static str {
        PROBES
            .iter()
            .find(|(_, table)| *table == self.table)
            .map_or("", |(id, _)| id)
    }
    pub(super) fn source(&self) -> String {
        format!("{HEAD}{}", self.body)
    }
    /// The `(a, b)` shape arguments for an input of `n` elements.
    pub(super) fn shape(&self, n: usize) -> (u32, u32) {
        let side = n.isqrt() as u32;
        match self.table {
            "transpose" | "stencil" => (side, side),
            "matrix" => (side, 0),
            "segmented" => ((n / 16) as u32, 0),
            "frontier" => (side, ROUNDS),
            _ => (0, 0),
        }
    }
    /// The same kernel as an ordinary-source `main` over its own input of `n`
    /// elements, for forced-device runs: `main` returns 42 only when every
    /// output word equals the Rust reference's, embedded as a literal. Also
    /// returns the byte range of the kernel body, where the device must run.
    pub(super) fn device_source(&self, n: usize) -> (String, std::ops::Range<usize>) {
        let (a, b) = self.shape(n);
        let expected = self.reference(&input(n));
        let body = self.body.trim();
        let kernel = &body[1..body.rfind("return ").unwrap_or(body.len() - 1)];
        let head = format!(
            "fn main()->i32 {{ var xs = filled_u32({n}u32, 0u32); for p in 0u32..{n}u32 {{ xs[p] = wrapping_add(wrapping_mul(p, 1664525u32), 1013904223u32); }} var out = filled_u32({}u32, 0u32); let a = {a}u32; let b = {b}u32;",
            expected.len()
        );
        let words = expected
            .iter()
            .map(|x| format!("{x}u32"))
            .collect::<Vec<_>>()
            .join(",");
        let source = format!(
            "{head}{kernel} let expect: Array<u32> = [{words}]; for hi in 0u32..{}u32 {{ if out[hi] != expect[hi] {{ return 9; }} }} return 42; }}",
            expected.len()
        );
        (source, head.len()..head.len() + kernel.len())
    }
    /// Independent Rust reference for `k`, including the output length.
    pub(super) fn reference(&self, xs: &[u32]) -> Vec<u32> {
        let (a, b) = self.shape(xs.len());
        let (a, b) = (a as usize, b as usize);
        let n = xs.len();
        match self.table {
            "reduction" => vec![xs.iter().fold(17u32, |s, x| s.wrapping_add(*x))],
            "scan" => xs
                .iter()
                .scan(17u32, |s, x| {
                    *s = s.wrapping_mul(1_664_525).wrapping_add(*x);
                    Some(*s)
                })
                .collect(),
            "compaction" => {
                let mut out = xs
                    .iter()
                    .copied()
                    .filter(|x| *x < 1 << 31)
                    .collect::<Vec<_>>();
                out.resize(n, 0);
                out
            }
            "transpose" => (0..n).map(|i| xs[(i % a) * b + i / a]).collect(),
            "stencil" => (0..n)
                .map(|i| {
                    let (r, c) = (i / b, i % b);
                    let mut v = xs[i];
                    for (ok, j) in [
                        (r > 0, i.wrapping_sub(b)),
                        (r + 1 < a, i + b),
                        (c > 0, i.wrapping_sub(1)),
                        (c + 1 < b, i + 1),
                    ] {
                        if ok {
                            v = v.wrapping_add(xs[j]);
                        }
                    }
                    v
                })
                .collect(),
            "matrix" => (0..n)
                .map(|i| {
                    (0..a).fold(0u32, |s, q| {
                        s.wrapping_add(xs[(i / a) * a + q].wrapping_mul(xs[q * a + i % a]))
                    })
                })
                .collect(),
            "segmented" => xs
                .chunks_exact(16)
                .map(|c| c.iter().fold(0u32, |s, x| s.wrapping_add(*x)))
                .collect(),
            "gather_scatter" => (0..n as u32)
                .map(|i| xs[(i.wrapping_mul(2_654_435_761) % n as u32) as usize])
                .collect(),
            // Exported kernels cannot allocate, so `out` holds the levels,
            // the frontier and the next frontier, `a` words each.
            _ => {
                let mut seen = vec![0u32; a];
                let mut frontier = (0..a).map(|u| u32::from(u == 0)).collect::<Vec<_>>();
                for step in 0..b {
                    let mut next = vec![0u32; a];
                    for u in 0..a {
                        if frontier[u] == 1 && seen[u] == 0 {
                            seen[u] = step as u32 + 1;
                            for w in 0..a {
                                if xs[u * a + w].wrapping_mul(2_654_435_761) >> 27 == 0
                                    && seen[w] == 0
                                {
                                    next[w] = 1;
                                }
                            }
                        }
                    }
                    frontier = next;
                }
                seen.extend(frontier);
                seen.resize(3 * a, 0);
                seen
            }
        }
    }
}

#[cfg(feature = "native")]
pub(super) fn cpu(start: Instant) -> Result<Value, String> {
    use crate::eval::host_read::{HostArgument as A, HostMutArray as M, HostReadArray as R};
    use crate::exports::{ExportEngine, ScalarValue, compile_exports};
    let cold = Instant::now();
    let mut library = compile_exports(
        MAP,
        "calibration-map.inv",
        &["map"],
        ExportEngine::NativeSerial,
    )
    .map_err(|e| format!("{e:?}"))?;
    let compile_ns = nanos(cold);
    let sizes = [0, SIZES[0], SIZES[1]];
    let mut timed = Vec::new();
    let mut points = Vec::new();
    for n in sizes {
        let input = (0..n)
            .map(|i| (i as u32).wrapping_mul(65537))
            .collect::<Vec<_>>();
        let mut output = vec![0u32; n];
        let (first, samples) = sample(start, || {
            resident(&input)?;
            output.fill(0);
            let now = Instant::now();
            let result = library
                .invoke_arrays(
                    "map",
                    &mut [A::Read(R::U32(&input)), A::Mut(M::U32(&mut output))],
                    100_000_000,
                )
                .map_err(|e| format!("{e:?}"))?;
            let ns = nanos(now);
            if result.value != ScalarValue::U32(n as u32)
                || !result.zero_copy
                || input
                    .iter()
                    .zip(&output)
                    .any(|(x, y)| *y != x.wrapping_add(1))
            {
                return Err("compiled CPU output mismatch".into());
            }
            Ok((ns, ()))
        })?;
        if n == 0 {
            points.push(point(
                None,
                FIRST_CALL,
                &[compile_ns.saturating_add(first)],
            )?);
        }
        let ns = samples.iter().map(|s| s.0).collect::<Vec<_>>();
        points.push(point(Some(n), "llvm-host-array-v1", &ns)?);
        timed.push(samples);
    }
    let mut passes = Vec::new();
    let mut per_pass = Vec::new();
    for pass in 0..5 {
        let times = timed
            .iter()
            .map(|s| group(s, pass).0 as f64 / 1e6)
            .collect::<Vec<_>>();
        let (fixed, work) = fit(&times, SIZES)?;
        passes.push(coefficients(fixed, work, None, 0.0, 0.0));
        per_pass.push(json!(times));
    }
    row(
        Engine::Cpu,
        "llvm-host-array-v1",
        passes,
        per_pass,
        Some(compile_ns as f64 / 1e6),
        &sizes,
        points,
    )
}
/// One family under the M5-01 protocol at sizes 0, 4096 and its large size.
#[cfg(feature = "native")]
pub(super) fn probe(start: Instant, family: &Family) -> Result<Value, String> {
    use crate::eval::host_read::{HostArgument as A, HostMutArray as M, HostReadArray as R};
    use crate::exports::{ExportEngine, compile_exports};
    let cold = Instant::now();
    let mut library = compile_exports(
        family.source().as_bytes(),
        "calibration-family.inv",
        &["k"],
        ExportEngine::NativeSerial,
    )
    .map_err(|e| format!("{e:?}"))?;
    let compile_ns = nanos(cold);
    let sizes = [0, SIZES[0], family.large];
    let mut timed = Vec::new();
    let mut points = Vec::new();
    for n in sizes {
        let xs = input(n);
        let (a, b) = family.shape(n);
        let expected = family.reference(&xs);
        let mut out = vec![0u32; expected.len()];
        let (first, samples) = sample(start, || {
            resident(&xs)?;
            out.fill(0);
            let now = Instant::now();
            library
                .invoke_arrays(
                    "k",
                    &mut [
                        A::Read(R::U32(&xs)),
                        A::U32(a),
                        A::U32(b),
                        A::Mut(M::U32(&mut out)),
                    ],
                    FUEL,
                )
                .map_err(|e| format!("{e:?}"))?;
            let ns = nanos(now);
            if out != expected {
                return Err(format!("{} output mismatch at n={n}", family.table));
            }
            Ok((ns, ()))
        })?;
        if n == 0 {
            points.push(point(
                None,
                FIRST_CALL,
                &[compile_ns.saturating_add(first)],
            )?);
        }
        let ns = samples.iter().map(|s| s.0).collect::<Vec<_>>();
        points.push(point(Some(n), "llvm-host-array-v1", &ns)?);
        timed.push(samples);
    }
    let mut passes = Vec::new();
    let mut per_pass = Vec::new();
    for pass in 0..5 {
        let times = timed
            .iter()
            .map(|s| group(s, pass).0 as f64 / 1e6)
            .collect::<Vec<_>>();
        let (fixed, work) = fit(&times, [sizes[1], sizes[2]])?;
        passes.push(coefficients(fixed, work, None, 0.0, 0.0));
        per_pass.push(json!(times));
    }
    let mut row = row(
        Engine::Cpu,
        "llvm-host-array-v1",
        passes,
        per_pass,
        Some(compile_ns as f64 / 1e6),
        &sizes,
        points,
    )?;
    row["family"] = json!(family.id());
    row["table_family"] = json!(family.table);
    Ok(row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::host_read::{HostArgument as A, HostMutArray as M, HostReadArray as R};
    use crate::exports::{ExportEngine, compile_exports};

    /// Each forced-device source returns 42 on the reference evaluator, so it
    /// embeds the Rust reference exactly, and one changed word returns 9. Its
    /// kernel range covers exactly the kernel's loops.
    #[test]
    fn device_sources_compare_every_output_word() {
        let run = |source: &str, table: &str| {
            let program = crate::front::parse(source.as_bytes())
                .and_then(|p| crate::check::check(p, false))
                .unwrap_or_else(|e| panic!("{table}: {}", e.message));
            let program = crate::mir::lower(&program).unwrap();
            crate::mir_eval::observe(&program, FUEL).result
        };
        for family in &FAMILIES_BEYOND_MAP {
            for n in [1, 256] {
                let (source, kernel) = family.device_source(n);
                assert!(family.body.contains(source[kernel.clone()].trim()));
                assert!(source[kernel].contains("for "), "{}", family.table);
                let ok = Ok(crate::eval::Value::I32(42));
                assert_eq!(run(&source, family.table), ok, "{} n={n}", family.table);
                // Segments of 16 leave no output at n = 1.
                if let Some(first) = family.reference(&input(n)).first() {
                    let wrong = source.replacen(
                        &format!("[{first}u32"),
                        &format!("[{}u32", first.wrapping_add(1)),
                        1,
                    );
                    let refused = Ok(crate::eval::Value::I32(9));
                    assert_eq!(run(&wrong, family.table), refused, "{}", family.table);
                }
            }
        }
    }

    /// The reference engine runs each kernel source; the Rust reference must
    /// agree, so a wrong kernel or reference fails without native code.
    #[test]
    fn family_kernels_match_their_rust_references() {
        let tables = FAMILIES_BEYOND_MAP.map(|f| f.table);
        assert_eq!(
            ["map"].iter().chain(&tables).copied().collect::<Vec<_>>(),
            FAMILIES
        );
        for family in &FAMILIES_BEYOND_MAP {
            assert!(family_valid(family.id()), "{}", family.table);
            let mut library = compile_exports(
                family.source().as_bytes(),
                "calibration-family.inv",
                &["k"],
                ExportEngine::Reference,
            )
            .unwrap_or_else(|e| panic!("{}: {e:?}", family.table));
            for n in [0, 256, 4096] {
                let xs = input(n);
                let (a, b) = family.shape(n);
                let expected = family.reference(&xs);
                let mut out = vec![0u32; expected.len()];
                library
                    .invoke_arrays(
                        "k",
                        &mut [
                            A::Read(R::U32(&xs)),
                            A::U32(a),
                            A::U32(b),
                            A::Mut(M::U32(&mut out)),
                        ],
                        FUEL,
                    )
                    .unwrap_or_else(|e| panic!("{} n={n}: {e:?}", family.table));
                assert_eq!(out, expected, "{} n={n}", family.table);
            }
        }
        // The frontier fixture really expands past its source vertex.
        let frontier = &FAMILIES_BEYOND_MAP[8];
        let levels = frontier.reference(&input(4096));
        assert!(
            levels[..64].iter().filter(|l| **l > 1).count() > 32,
            "{levels:?}"
        );
        assert!(levels[..64].iter().max() > Some(&4), "{levels:?}");
        for family in &FAMILIES_BEYOND_MAP {
            assert_eq!(
                family.large.isqrt().pow(2),
                family.large,
                "{}",
                family.table
            );
        }
    }
}
