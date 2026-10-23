//! M4-01 Metal emission on the CPU: the Metal plan shares the verified IR with CUDA and
//! binds the cache key, the probe table agrees with the software binary32 oracle, and
//! every status word shares one guard encoding. The hardware tests are in
//! `tests/gpu_metal/m4_01.rs`.
#![forbid(unsafe_code)]

#[path = "issue_m4_01_device/corpus.rs"]
mod corpus;
#[path = "issue_m4_01_device/ops.rs"]
mod ops;

use corpus::MAT;
use invar::tensor;
use invar_device::Capabilities;

const METAL: Capabilities = Capabilities::Metal {
    apple_family: 9,
    msl_major: 3,
    msl_minor: 1,
};

#[test]
fn issue_m4_01_metal_emission_shares_the_verified_plan_and_binds_the_key() {
    let cuda = tensor::emit_cuda(
        MAT.as_bytes(),
        "mat",
        [3, 5, 3],
        [15, 15],
        Capabilities::cuda_12_9_sm89(),
    )
    .unwrap();
    let metal = tensor::emit_msl(MAT.as_bytes(), "mat", [3, 5, 3], [15, 15], METAL).unwrap();
    // Same verified IR; only the target capabilities differ.
    assert_eq!(cuda.plan().ir(), metal.plan().ir());
    assert_ne!(cuda.plan().capabilities(), metal.plan().capabilities());
    assert_eq!(metal.layouts().dimensions(), [3, 5, 3]);
    assert!(metal.source().contains("#pragma METAL fp contract(off)"));
    assert!(metal.source().contains("invar_f32_contract_v1"));
    for option in metal.plan().msl_options() {
        assert!(metal.source().contains(option));
    }
    assert!(!metal.matches_hint(cuda.key()));
    let again = tensor::emit(MAT.as_bytes(), "mat", [3, 5, 3], [15, 15], METAL).unwrap();
    assert!(metal.matches_hint(again.key()));
    assert_eq!(again.source(), metal.source());
    let other_family = Capabilities::Metal {
        apple_family: 7,
        msl_major: 3,
        msl_minor: 1,
    };
    let other = tensor::emit_msl(MAT.as_bytes(), "mat", [3, 5, 3], [15, 15], other_family);
    assert!(!metal.matches_hint(other.unwrap().key()));
    assert!(
        !metal.matches_hint(
            tensor::emit_msl(MAT.as_bytes(), "mat", [3, 5, 4], [15, 20], METAL)
                .unwrap()
                .key()
        )
    );
    // Every rejection keeps the typed error; nothing falls through to CUDA.
    assert!(matches!(
        tensor::emit_msl(
            MAT.as_bytes(),
            "mat",
            [3, 5, 3],
            [15, 15],
            Capabilities::cuda_12_9_sm89()
        ),
        Err(tensor::Error::Device(
            invar_device::floating::Error::Capabilities
        ))
    ));
    for caps in [
        Capabilities::Metal {
            apple_family: 6,
            msl_major: 3,
            msl_minor: 1,
        },
        Capabilities::Metal {
            apple_family: 9,
            msl_major: 3,
            msl_minor: 2,
        },
    ] {
        assert!(matches!(
            tensor::emit_msl(MAT.as_bytes(), "mat", [3, 5, 3], [15, 15], caps),
            Err(tensor::Error::Device(
                invar_device::floating::Error::Capabilities
            ))
        ));
    }
    assert!(matches!(
        tensor::emit_msl(MAT.as_bytes(), "mat", [3, 5, 3], [14, 15], METAL),
        Err(tensor::Error::Layout(_))
    ));
}

#[test]
fn issue_m4_01_metal_probe_kernel_covers_every_d1_operation() {
    use invar::numeric_contract::{Comparison, F32Bits};
    use invar_device::f32_probe::{CASES, Op};
    // Second, independent derivation of the probe table with the root's
    // software binary32 oracle (the device crate checks host arithmetic).
    for (index, case) in CASES.iter().enumerate() {
        let [a, b, c] = [case.a, case.b, case.c].map(F32Bits::from_bits);
        let actual = match case.op {
            Op::Mul => a.mul(b).to_bits(),
            Op::Add => a.add(b).to_bits(),
            Op::Sub => a.sub(b).to_bits(),
            Op::Fma => a.fma(b, c).to_bits(),
            Op::Accumulate => c.add(a.mul(b)).to_bits(),
            Op::Lt => u32::from(a.compare(Comparison::Lt, b)),
            Op::Eq => u32::from(a.compare(Comparison::Eq, b)),
            Op::FromU32 => F32Bits::from_u32(case.a).to_bits(),
            Op::FromI32 => F32Bits::from_i32(case.a as i32).to_bits(),
            Op::TruncI32 => a.trunc_i32().unwrap() as u32,
            Op::Div => a.div(b).to_bits(),
            Op::Sqrt => a.sqrt().to_bits(),
        };
        assert_eq!(actual, case.expected, "case {index} {}", case.what);
    }
    let msl = invar_device::f32_probe::emit_msl();
    assert!(msl.contains("#pragma METAL fp contract(off)"));
    assert!(msl.contains("s = s + a * b;"));
}

/// One encoding for every device-written status word: each word's legal bits
/// decode, and an untouched `UNWRITTEN` sentinel or any stray bit is refused.
/// Both rank-1 kernels claim their status word before any other statement.
#[test]
fn issue_m4_01_status_words_share_the_guard_encoding() {
    use invar_device::f32_ops::{CUDA_STATUS, Kind, METAL_STATUS, Node, Program, Scalar};
    use invar_device::f32_probe::{self, Report};
    use invar_device::floating::guard::{
        self, GUARD, InvalidWord, OVERFLOW, UNWRITTEN, decode_bits, msl_claim,
    };
    for (legal, ok, bad) in [
        (GUARD, &[0, 1][..], &[2, 3, UNWRITTEN, u32::MAX][..]),
        (CUDA_STATUS, &[0, OVERFLOW], &[GUARD, 3, 4, UNWRITTEN]),
        (
            METAL_STATUS,
            &[0, 1, 2, 3],
            &[4, 5, UNWRITTEN, UNWRITTEN | 1],
        ),
    ] {
        for &w in ok {
            assert_eq!(decode_bits(w, legal), Ok(w), "{legal} {w}");
        }
        for &w in bad {
            assert_eq!(decode_bits(w, legal), Err(InvalidWord(w)), "{legal} {w}");
        }
    }
    assert_eq!(guard::decode(0), Ok(false));
    assert_eq!(guard::decode(GUARD), Ok(true));
    assert_eq!(guard::decode(UNWRITTEN), Err(InvalidWord(UNWRITTEN)));
    assert_eq!(
        Report::evaluate(&f32_probe::unwritten()),
        Err(f32_probe::Error::Guard(InvalidWord(UNWRITTEN)))
    );
    let program = || Program {
        length: 2,
        inputs: vec![Scalar::F32, Scalar::F32],
        nodes: vec![Node::Input(0), Node::Input(1), Node::Mul(0, 1)],
        kind: Kind::Map,
    };
    let cuda =
        invar_device::f32_ops::emit(&ops::plan(program(), Capabilities::cuda_12_9_sm89())).unwrap();
    let head = cuda.find("unsigned flag = 0u;\n").unwrap() + "unsigned flag = 0u;\n".len();
    assert!(cuda[head..].starts_with(&format!("  atomicAnd(C + 2u, {CUDA_STATUS}u);\n")));
    let metal = invar_device::f32_ops::emit(&ops::plan(program(), METAL)).unwrap();
    let head = metal.find("uint flag = 0u;\n").unwrap() + "uint flag = 0u;\n".len();
    assert!(metal[head..].starts_with(&msl_claim(METAL_STATUS)));
}
