//! v0.2 reductions on a live device: kernel partials finished on the host
//! equal the CPU engines bit for bit at every size class, every leaf chunk
//! exponent and on adversarial inputs. Shared by the CUDA and Metal binaries;
//! the including module supplies `Device`, `open`, `device_bits` and
//! `guard_fires` (a backend whose subnormal guard discards a dispatch returns
//! `None` from `device_bits`, and says where that is expected).
use super::{Device, device_bits, guard_fires, open};
use invar::fsum;
use invar_device::fsum::Op;

use super::cases::{Rng, adversarial_vectors};

pub const OPS: [Op; 4] = [Op::SumTree, Op::DotTree, Op::SumExact, Op::DotExact];

pub fn expected(op: Op, a: &[u32], b: &[u32]) -> u32 {
    match op {
        Op::SumTree => fsum::sum_tree(a),
        Op::DotTree => fsum::dot_tree(a, b),
        Op::SumExact => fsum::sum_exact(a),
        Op::DotExact => fsum::dot_exact(a, b),
    }
}

/// The host finish over a device result's partials.
pub fn finish(op: Op, terms: usize, rows: &[i64], nodes: &[u32]) -> u32 {
    if op.is_exact() {
        fsum::finish_rows(rows, terms as u64, op.is_dot())
    } else {
        fsum::canonical(fsum::fold_aligned(nodes.len(), |g| {
            f32::from_bits(nodes[g])
        }))
    }
}

#[test]
#[ignore = "requires a GPU (CUDA with NVRTC, or Apple Metal)"]
fn v02_fsum_adversarial_vectors_match_cpu_bits() {
    let mut device: Device = open();
    let mut published = 0;
    for (label, a, b) in adversarial_vectors()
        .into_iter()
        .filter(|(_, a, _)| !a.is_empty())
    {
        for op in OPS {
            let got = device_bits(&mut device, op, &a, &b, 2);
            if let Some(got) = got {
                assert_eq!(got, expected(op, &a, &b), "{label} {op:?}");
                published += 1;
            } else {
                assert!(!op.is_exact(), "the exact kernel is integer-only: {label}");
            }
        }
    }
    assert!(published > 100, "{published}");
}

#[test]
#[ignore = "requires a GPU (CUDA with NVRTC, or Apple Metal)"]
fn v02_fsum_group_boundaries_and_leaf_shifts_are_bit_identical() {
    let mut device: Device = open();
    let mut rng = Rng(0x3e7a1);
    for n in [
        1usize,
        255,
        2047,
        2048,
        2049,
        8191,
        8193,
        65_543,
        1 << 20,
        3_000_017,
    ] {
        let a = (0..n).map(|_| rng.word()).collect::<Vec<_>>();
        let b = (0..n).map(|_| rng.finite()).collect::<Vec<_>>();
        // Finite inputs keep every rounding visible; `a` mostly ends in NaN.
        let finite = (0..n).map(|_| rng.finite()).collect::<Vec<_>>();
        for op in OPS {
            for (input, normal) in [(&a, false), (&finite, true)] {
                let want = expected(op, input, &b);
                for shift in 0..=3 {
                    match device_bits(&mut device, op, input, &b, shift) {
                        Some(got) => assert_eq!(got, want, "n={n} {op:?} shift={shift}"),
                        None => assert!(!normal && !op.is_exact(), "n={n} {op:?} guard"),
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a GPU (CUDA with NVRTC, or Apple Metal)"]
fn v02_fsum_subnormal_guard_and_contraction_cases() {
    let mut device: Device = open();
    let subnormal = (1..=4096u32).map(|i| i % 0x7f_ffff).collect::<Vec<_>>();
    // A product that underflows from normal operands, and a sum of normals
    // whose exact result is subnormal.
    let tiny = vec![0x1f80_0000u32; 64];
    let cancel = vec![0x0080_0001u32, 0x8080_0000];
    // `1 + 2^-23` squared needs a rounded product: a fused multiply-add
    // would keep the `2^-46` term and change the sum.
    let near = vec![0x3f80_0001u32; 4096];
    let minus = vec![0xbf80_0002u32; 4096];
    let mixed = near
        .iter()
        .zip(&minus)
        .flat_map(|(&x, &y)| [x, y])
        .collect::<Vec<_>>();
    for (label, a) in [
        ("subnormal", &subnormal),
        ("tiny", &tiny),
        ("cancel", &cancel),
        ("near", &near),
        ("mixed", &mixed),
    ] {
        for op in OPS {
            let got = device_bits(&mut device, op, a, a, 2);
            if guard_fires(op, label) {
                assert_eq!(got, None, "guard must fire: {op:?} {label}");
            } else {
                assert_eq!(got, Some(expected(op, a, a)), "{op:?} {label}");
            }
        }
    }
}
