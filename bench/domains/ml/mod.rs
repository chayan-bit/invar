//! ML suite: normalisation, GEMM and attention written as ordinary loops.
//! Quantized matmul (i8 storage, i32 accumulate) waits for the reduced-
//! precision storage types of #280 (ENG-03) and is recorded pending in the
//! manifest, never approximated here.
use super::{Case, Input, Kernel, Wanted, oracle};

fn input(len: usize, seed: u32) -> Input {
    Input {
        len: len as u32,
        seed,
    }
}

fn softmax_rows() -> Kernel {
    let cases = [
        (1, 1, 11),
        (3, 7, 12),
        (4, 32, 13),
        (5, 33, 14),
        (2, 129, 15),
    ]
    .map(|(rows, cols, seed)| {
        let x = oracle::fill(rows * cols, seed);
        Case {
            shape: vec![("r", rows), ("c", cols)],
            inputs: vec![input((rows * cols) as usize, seed)],
            call: format!("softmax_rows(in0, {rows}u32, {cols}u32)"),
            wanted: Wanted::F32(oracle::softmax_rows(&x, rows as usize, cols as usize)),
        }
    })
    .into();
    Kernel {
        name: "softmax_rows",
        leaf: include_str!("softmax_rows.inv"),
        cases,
    }
}

fn rmsnorm_rows() -> Kernel {
    let cases = [(1, 1, 21), (3, 8, 22), (4, 31, 23), (6, 64, 24)]
        .map(|(rows, cols, seed)| {
            let x = oracle::fill(rows * cols, seed);
            let w = oracle::fill(cols, seed + 100);
            Case {
                shape: vec![("r", rows), ("c", cols)],
                inputs: vec![
                    input((rows * cols) as usize, seed),
                    input(cols as usize, seed + 100),
                ],
                call: format!("rmsnorm_rows(in0, in1, {rows}u32, {cols}u32)"),
                wanted: Wanted::F32(oracle::rmsnorm_rows(&x, &w, rows as usize, cols as usize)),
            }
        })
        .into();
    Kernel {
        name: "rmsnorm_rows",
        leaf: include_str!("rmsnorm_rows.inv"),
        cases,
    }
}

fn layernorm_rows() -> Kernel {
    let cases = [(1, 2, 31), (3, 8, 32), (4, 33, 33), (5, 64, 34)]
        .map(|(rows, cols, seed)| {
            let x = oracle::fill(rows * cols, seed);
            let gamma = oracle::fill(cols, seed + 100);
            let beta = oracle::fill(cols, seed + 200);
            Case {
                shape: vec![("r", rows), ("c", cols)],
                inputs: vec![
                    input((rows * cols) as usize, seed),
                    input(cols as usize, seed + 100),
                    input(cols as usize, seed + 200),
                ],
                call: format!("layernorm_rows(in0, in1, in2, {rows}u32, {cols}u32)"),
                wanted: Wanted::F32(oracle::layernorm_rows(
                    &x,
                    &gamma,
                    &beta,
                    rows as usize,
                    cols as usize,
                )),
            }
        })
        .into();
    Kernel {
        name: "layernorm_rows",
        leaf: include_str!("layernorm_rows.inv"),
        cases,
    }
}

fn gemm_f32() -> Kernel {
    let cases = [
        (1, 1, 1, 41),
        (8, 8, 8, 42),
        (7, 13, 5, 43),
        (16, 4, 9, 44),
        (3, 17, 11, 45),
    ]
    .map(|(m, k, n, seed)| {
        let a = oracle::fill(m * k, seed);
        let b = oracle::fill(k * n, seed + 100);
        Case {
            shape: vec![("m", m), ("k", k), ("n", n)],
            inputs: vec![
                input((m * k) as usize, seed),
                input((k * n) as usize, seed + 100),
            ],
            call: format!("gemm_f32(in0, in1, {m}u32, {k}u32, {n}u32)"),
            wanted: Wanted::F32(oracle::gemm_f32(&a, &b, m as usize, k as usize, n as usize)),
        }
    })
    .into();
    Kernel {
        name: "gemm_f32",
        leaf: include_str!("gemm_f32.inv"),
        cases,
    }
}

fn gemm_tc() -> Kernel {
    let cases = [(1, 1, 1, 81), (8, 8, 8, 82), (7, 13, 5, 83), (4, 17, 6, 84)]
        .map(|(m, k, n, seed)| {
            let a = oracle::fill(m * k, seed);
            let b = oracle::fill(k * n, seed + 100);
            Case {
                shape: vec![("m", m), ("k", k), ("n", n)],
                inputs: vec![
                    input((m * k) as usize, seed),
                    input((k * n) as usize, seed + 100),
                ],
                call: format!("gemm_tc(in0, in1, {m}u32, {k}u32, {n}u32)"),
                wanted: Wanted::F32(oracle::gemm_tc(&a, &b, m as usize, k as usize, n as usize)),
            }
        })
        .into();
    Kernel {
        name: "gemm_tc",
        leaf: include_str!("gemm_tc.inv"),
        cases,
    }
}

fn attention() -> Kernel {
    let cases = [(1, 1, 51), (4, 8, 52), (6, 16, 53), (9, 5, 54)]
        .map(|(seq, dim, seed)| {
            let q = oracle::fill(seq * dim, seed);
            let k = oracle::fill(seq * dim, seed + 100);
            let v = oracle::fill(seq * dim, seed + 200);
            Case {
                shape: vec![("s", seq), ("d", dim)],
                inputs: vec![
                    input((seq * dim) as usize, seed),
                    input((seq * dim) as usize, seed + 100),
                    input((seq * dim) as usize, seed + 200),
                ],
                call: format!("attention(in0, in1, in2, {seq}u32, {dim}u32)"),
                wanted: Wanted::F32(oracle::attention(&q, &k, &v, seq as usize, dim as usize)),
            }
        })
        .into();
    Kernel {
        name: "attention",
        leaf: include_str!("attention.inv"),
        cases,
    }
}

pub fn kernels() -> Vec<Kernel> {
    vec![
        softmax_rows(),
        rmsnorm_rows(),
        layernorm_rows(),
        gemm_f32(),
        gemm_tc(),
        attention(),
    ]
}
