//! ENG-06 (#283) part C: the shipped `gemm` and `tc` schedules emit through
//! the generic contraction emitter, with the target chosen from live
//! `Capabilities`. Every capability the device gate admits has a target file,
//! and the emitted kernels for every admitted device and corpus shape are the
//! ones the fixed templates emitted (pinned digests taken from those
//! templates before the switch). Device rows are not run here.
#![forbid(unsafe_code)]

#[allow(dead_code)]
#[path = "../src/contraction_source.rs"]
mod contraction_source;
#[path = "issue_m4_03_device/corpus.rs"]
mod corpus;

use contraction_source::{Contract, DeviceEmission};
use corpus::{DIRECT, SHAPES, TC_DIRECT};
use invar::{check, front, general_net, hex, layout_contract, mir, numeric_contract, tensor_cache};
use invar_device::target::Registry;
use invar_device::{Capabilities, floating, gemm, source_hash, tc};

/// `gemm.rs` and `tc.rs` fixed-template digests (4bd68ca5, taken at b8298a5 before the switch): the
/// `source_hash` of every admitted device's text for every corpus shape,
/// folded in order. A change here is a change to the emitted kernels.
const GEMM_PIN: u64 = 0xffa6_fd51_4891_44b0;
const TC_PIN: u64 = 0x0d48_d42a_839c_b33b;

/// The device gate's own admission: NVRTC 12.4 to 12.9 options for CUDA, MSL
/// 3.1 on Apple families 7 to 10 for Metal.
fn admitted() -> Vec<Capabilities> {
    let mut caps = Vec::new();
    for sm in 0..200 {
        let newest = (4..=9)
            .map(|minor| Capabilities::Cuda {
                major: 12,
                minor,
                sm,
            })
            .rfind(|&c| floating::cuda_strict_options(c).is_some());
        caps.extend(newest);
    }
    caps.extend((7..=10).map(|apple_family| Capabilities::Metal {
        apple_family,
        msl_major: 3,
        msl_minor: 1,
    }));
    caps
}

fn emission(source: &str, dims: [u32; 3], caps: Capabilities) -> Option<DeviceEmission> {
    let [r, k, n] = dims.map(u64::from);
    Contract::from_source(source.as_bytes(), "mat")
        .unwrap()
        .emit_device(source.as_bytes(), dims, [r * k, k * n], caps)
        .ok()
}

#[test]
fn issue_eng_06_wiring_every_admitted_capability_selects_a_target() {
    let registry = Registry::builtin().unwrap();
    let caps = admitted();
    for &c in &caps {
        let target = registry.select(c).unwrap_or_else(|e| panic!("{c:?}: {e}"));
        assert!(target.gpu.is_some(), "{c:?}");
    }
    // And no GPU target file names a device the gate refuses.
    let gpus = registry
        .targets()
        .iter()
        .filter(|t| t.gpu.is_some())
        .count();
    assert_eq!(gpus, caps.len());
    assert_eq!(caps.len(), 19 + 4);
}

#[test]
fn issue_eng_06_wiring_emitted_kernels_equal_the_fixed_templates() {
    let (mut gemm_digest, mut tc_digest) = (0u64, 0u64);
    let (mut gemm_texts, mut tc_texts) = (0, 0);
    let fold =
        |digest: u64, text: &str| source_hash(&format!("{digest:016x}{:016x}", source_hash(text)));
    for caps in admitted() {
        for dims in SHAPES {
            let strict = emission(DIRECT, dims, caps).unwrap();
            let text = match caps {
                Capabilities::Cuda { .. } => gemm::emit_cuda(strict.plan()),
                Capabilities::Metal { .. } => gemm::emit_msl(strict.plan()),
            };
            gemm_digest = fold(gemm_digest, &text.unwrap());
            gemm_texts += 1;
            // mma.sync m16n8k16 needs sm_80; the gate refuses older devices.
            let Some(e) = emission(TC_DIRECT, dims, caps) else {
                assert!(matches!(caps, Capabilities::Cuda { sm, .. } if sm < 80));
                continue;
            };
            let text = match caps {
                Capabilities::Cuda { .. } => tc::emit_cuda(e.plan()),
                Capabilities::Metal { .. } => tc::emit_msl(e.plan()),
            };
            tc_digest = fold(tc_digest, &text.unwrap());
            tc_texts += 1;
        }
    }
    assert_eq!(gemm_texts, 23 * SHAPES.len());
    assert_eq!(tc_texts, (23 - 9) * SHAPES.len());
    assert_eq!(
        (gemm_digest, tc_digest),
        (GEMM_PIN, TC_PIN),
        "{gemm_digest:#018x} {tc_digest:#018x}"
    );
}
