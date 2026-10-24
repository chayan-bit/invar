//! CPU-only qualification of the production metadata, compiler-controller and
//! owning completion algorithm. No CUDA open, loader probe or NVRTC call.
use crate::child_deadline;

use super::*;
use invar_device::{Provenance, Span};
use std::cell::Cell;

const CAPS: Capabilities = Capabilities::Cuda {
    major: 12,
    minor: 4,
    sm: 86,
};

#[test]
fn issue_m4_floating_runtime_capacity_shortfall_precedes_storage() {
    let p = plan([2, 3, 4], 2);
    let s = shape(&p, CAPS, [10, 16]).unwrap();
    let allocations = Cell::new(0);
    let result = admit_before_storage(
        || crate::validate_device_capacity(s.transfer_bytes, s.transfer_bytes),
        || {
            allocations.set(allocations.get() + 1);
            Ok(())
        },
    );
    assert_eq!(allocations.get(), 0);
    assert!(matches!(result, Err(Error::ResourceLimit(_))));
}

fn plan(dimensions: [u32; 3], padding: usize) -> ir::Verified {
    let [r, k, c] = dimensions;
    let buffer = |id, shape: [u32; 2]| ir::Buffer {
        id,
        elements: u64::from(shape[0]) * u64::from(shape[1]) + padding as u64 * 2,
        shape,
        strides: [u64::from(shape[1]), 1],
        offset: padding as u64,
        alignment: 4,
    };
    ir::verify(
        ir::Ir {
            version: 1,
            numeric_version: 1,
            numeric_policy: ir::NUMERIC_POLICY.into(),
            layout_version: 1,
            span: Span { start: 1, end: 90 },
            source_identity: 71,
            graph_revision: 8,
            function_contract: [17; 32],
            dimensions,
            inputs: [buffer(1, [r, k]), buffer(2, [k, c])],
            output: buffer(3, [r, c]),
        },
        CAPS,
        Provenance {
            source_identity: 71,
            graph_revision: 8,
        },
    )
    .unwrap()
}

#[test]
fn issue_m4_03_matrix_size_2_28_fits_shared_cuda_budget() {
    let p = plan([1 << 14; 3], 0);
    let s = shape(&p, CAPS, [1 << 28; 2]).unwrap();
    assert_eq!(s.lengths, [1 << 28; 3]);
    assert_eq!(s.logical, 1 << 28);
    assert_eq!(s.transfer_bytes, 4 * 1024 * 1024 * 1024);
}

#[test]
fn issue_m4_floating_runtime_exact_capacities_identity_and_storage_budget() {
    let p = plan([2, 3, 4], 2);
    let s = shape(&p, CAPS, [10, 16]).unwrap();
    assert_eq!(s.lengths, [10, 16, 12]);
    assert_eq!((s.logical, s.offset), (8, 2));
    assert_eq!(s.transfer_bytes, (10 + 16 + 24) * 4);
    for lengths in [[9, 16], [11, 16], [10, 15], [10, 17]] {
        assert!(matches!(shape(&p, CAPS, lengths), Err(Error::Limit)));
    }
    assert!(matches!(
        shape(
            &p,
            Capabilities::Cuda {
                major: 12,
                minor: 5,
                sm: 86
            },
            [10, 16]
        ),
        Err(Error::Limit)
    ));
    assert!(matches!(
        crate::validate_transfer_budget(0, crate::MAX_TRANSFER_BYTES + 1),
        Err(Error::ResourceLimit(_))
    ));
    let origin = Provenance {
        source_identity: 71,
        graph_revision: 8,
    };
    let mut large = plan([8192, 1, 8192], 0).ir().clone();
    large.inputs[0].elements = 1 << 28;
    large.inputs[1].elements = 1 << 28;
    large.output.elements = 1 << 28;
    let large = ir::verify(large, CAPS, origin).unwrap();
    let s = shape(&large, CAPS, [1 << 28, 1 << 28]).unwrap();
    assert_eq!(s.transfer_bytes, 4 * 1024 * 1024 * 1024);
    let mut over = large.ir().clone();
    over.output.elements = 1 << 29;
    let over = ir::verify(over, CAPS, origin).unwrap();
    assert!(matches!(
        shape(&over, CAPS, [1 << 28, 1 << 28]),
        Err(Error::ResourceLimit(_))
    ));
    let mut exact = plan([1, 1, 1], 0).ir().clone();
    exact.inputs[0].elements = 1 << 27;
    exact.inputs[1].elements = 1 << 27;
    exact.output.elements = 1 << 29;
    let exact = ir::verify(exact, CAPS, origin).unwrap();
    assert!(shape(&exact, CAPS, [1 << 27, 1 << 27]).is_ok());
    for d in [[0, 3, 4], [2, 3, 0], [0, 0, 0]] {
        let p = plan(d, 1);
        let lengths = p.ir().inputs.each_ref().map(|b| b.elements as usize);
        let s = shape(&p, CAPS, lengths).unwrap();
        assert_eq!(s.logical, 0);
        let output = zeros(s.lengths[2]).unwrap();
        validate_output(&s, &output).unwrap();
    }
    let p = plan([2, 0, 3], 0);
    let s = shape(&p, CAPS, [0, 0]).unwrap();
    assert_eq!(s.lengths, [0, 0, 6]);
    assert_eq!(s.transfer_bytes, (1 + 1 + 6 * 2) * 4);
}

#[test]
fn issue_m4_floating_runtime_strict_child_options_and_early_identity_refusals() {
    let p = plan([1, 2, 1], 1);
    let source = ir::emit_cuda(&p).unwrap();
    assert_eq!(strict_options(&source, CAPS).unwrap(), p.nvrtc_options());
    for bad in [
        source.replace("strict-f32-v1", "relaxed"),
        source.replace("--ftz=false", "--ftz=true"),
        source.replace("compute_86", "compute_80"),
        source.replace("caps=Cuda", "caps=Unknown"),
    ] {
        assert!(matches!(strict_options(&bad, CAPS), Err(Error::Compile(_))));
        // Identity failures return before the optional library presence probe.
        assert!(matches!(
            compile_floating_cubin(&bad, CAPS),
            Err(Error::Compile(_))
        ));
    }
    assert_eq!(strict_options("\0", CAPS), Err(Error::Limit));
    assert_eq!(
        strict_options(&" ".repeat(invar_device::MAX_SOURCE_BYTES + 1), CAPS),
        Err(Error::Limit)
    );
    for caps in [
        Capabilities::Cuda {
            major: 11,
            minor: 8,
            sm: 86,
        },
        Capabilities::Cuda {
            major: 12,
            minor: 3,
            sm: 86,
        },
        Capabilities::Cuda {
            major: 12,
            minor: 4,
            sm: 81,
        },
        Capabilities::Cuda {
            major: 12,
            minor: 7,
            sm: 120,
        },
    ] {
        assert_eq!(strict_options(&source, caps), Err(Error::Unavailable));
    }
}

#[test]
fn issue_m4_01_floating_probe_child_accepts_only_the_generated_probe_source() {
    let probe = invar_device::f32_probe::emit_cuda(CAPS).unwrap();
    assert_eq!(
        probe_options(&probe, CAPS).unwrap(),
        invar_device::floating::cuda_strict_options(CAPS).unwrap()
    );
    for bad in [
        probe.replace("--ftz=false", "--ftz=true"),
        probe.replace("a * b", "a * b + 0.0f"),
        probe.replace("sm: 86", "sm: 80"),
        format!("{probe} "),
    ] {
        // Identity failures return before the optional library presence probe.
        assert!(matches!(
            compile_floating_probe_cubin(&bad, CAPS),
            Err(Error::Compile(_))
        ));
    }
    // A probe for another target is a mismatch for these capabilities.
    let other = Capabilities::Cuda {
        major: 12,
        minor: 9,
        sm: 120,
    };
    let blackwell = invar_device::f32_probe::emit_cuda(other).unwrap();
    assert!(matches!(
        probe_options(&blackwell, CAPS),
        Err(Error::Compile(_))
    ));
    assert_eq!(probe_options(&blackwell, other).unwrap().len(), 7);
    assert_eq!(probe_options("\0", CAPS), Err(Error::Limit));
    let unsupported = Capabilities::Cuda {
        major: 12,
        minor: 4,
        sm: 120,
    };
    assert!(invar_device::f32_probe::emit_cuda(unsupported).is_none());
    assert!(probe_options(&probe, unsupported).is_err());
}

#[test]
fn issue_m4_floating_runtime_finite_private_output_and_padding_validation() {
    let p = plan([1, 2, 1], 1);
    let s = shape(&p, CAPS, [4, 4]).unwrap();
    for word in [0, 0x80000000, 1, 0x00800000, 0x7f7fffff, 0xff7fffff] {
        assert!(finite(word));
        validate_output(&s, &[0, word, 0]).unwrap();
    }
    for word in [0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001, 0xff800001] {
        assert!(!finite(word));
        assert!(validate_output(&s, &[0, word, 0]).is_err());
    }
    for bad in [&[0, 0][..], &[0, 0, 0, 0], &[1, 0, 0], &[0, 0, 1]] {
        assert!(validate_output(&s, bad).is_err());
    }
}

#[test]
fn issue_m4_floating_runtime_private_input_finiteness_and_shared_identity() {
    let p = plan([2, 2, 2], 0);
    for word in [0, 0x80000000, 1, 0x7f7fffff] {
        validate_staging(&p, [&[word; 4], &[word; 4]]).unwrap();
    }
    for word in [0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001] {
        assert!(validate_staging(&p, [&[word; 4], &[0; 4]]).is_err());
        assert!(validate_staging(&p, [&[0; 4], &[word; 4]]).is_err());
    }
    let mut alias = p.ir().clone();
    alias.inputs[1].id = alias.inputs[0].id;
    let alias = ir::verify(
        alias,
        CAPS,
        Provenance {
            source_identity: 71,
            graph_revision: 8,
        },
    )
    .unwrap();
    validate_staging(&alias, [&[1, 2, 3, 4], &[1, 2, 3, 4]]).unwrap();
    assert_eq!(
        validate_staging(&alias, [&[1, 2, 3, 4], &[1, 2, 3, 5]]),
        Err(Error::StaleBorrow)
    );
}

#[cfg(unix)]
mod compiler {
    use super::*;
    use std::path::PathBuf;

    // Committed executables never have a write fd for sibling forks to inherit.
    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/floating-compiler")
            .join(name)
    }
    #[test]
    fn issue_m4_floating_runtime_child_policy_protocol_and_original_diagnostic() {
        let source = ir::emit_cuda(&plan([1, 2, 1], 0)).unwrap();
        let script = fixture("success.sh");
        let result = compile_bounded(
            &source,
            &script,
            CAPS,
            child_deadline::wall(Duration::from_secs(1)),
        )
        .unwrap();
        assert_eq!(result.image, b"mock-cubin");
        assert_eq!(result.log, "original warning");
        let script = fixture("diagnostic.sh");
        assert!(
            matches!(compile_bounded(&source,&script,CAPS,child_deadline::wall(Duration::from_secs(1))),Err(Error::Compile(message)) if message.contains("original strict-option diagnostic")&&message.contains("27"))
        );
    }
    #[test]
    fn issue_m4_floating_runtime_child_empty_image_and_output_caps_fail_closed() {
        let source = ir::emit_cuda(&plan([1, 2, 1], 0)).unwrap();
        for name in ["empty.sh", "oversized-image.sh", "oversized-log.sh"] {
            let script = fixture(name);
            assert!(matches!(
                compile_bounded(
                    &source,
                    &script,
                    CAPS,
                    child_deadline::wall(Duration::from_secs(2))
                ),
                Err(Error::Limit)
            ));
        }
    }
    /// A helper that exits but leaves a descendant on its pipes must neither
    /// stall the compile to its deadline nor leak the descendant.
    #[test]
    fn issue_m4_floating_runtime_child_lingering_descendant_neither_stalls_nor_leaks() {
        use crate::cuda::compiler_child::tests::{descendant, fixture, lingers};
        let source = ir::emit_cuda(&plan([1, 2, 1], 0)).unwrap();
        let deadline = Duration::from_secs(10);
        let begun = Instant::now();
        let compiled = compile_bounded(&source, &fixture("lingering.sh"), CAPS, deadline).unwrap();
        assert!(
            begun.elapsed() < Duration::from_secs(5),
            "{:?}",
            begun.elapsed()
        );
        assert_eq!(compiled.image, b"mock-cubin");
        assert!(!lingers(descendant(&compiled.log)), "{}", compiled.log);
        let begun = Instant::now();
        let result = compile_bounded(&source, &fixture("lingering-log.sh"), CAPS, deadline);
        assert!(
            matches!(result, Err(Error::Limit)),
            "{:?}",
            result.as_ref().err()
        );
        assert!(
            begun.elapsed() < Duration::from_secs(5),
            "{:?}",
            begun.elapsed()
        );
    }
    #[test]
    fn issue_m4_floating_runtime_child_deadline_covers_nonreading_stdin() {
        let mut source = ir::emit_cuda(&plan([1, 2, 1], 0)).unwrap();
        source.push_str("//");
        source.extend(std::iter::repeat_n(
            ' ',
            invar_device::MAX_SOURCE_BYTES - source.len(),
        ));
        let script = fixture("nonreading-stdin.sh");
        let begun = Instant::now();
        assert!(matches!(
            compile_bounded(&source, &script, CAPS, Duration::from_millis(50)),
            Err(Error::Timeout)
        ));
        assert!(begun.elapsed() < child_deadline::wall(Duration::from_secs(2)));
    }
}
