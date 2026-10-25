//! Real Metal cross-tile carry and head-flag regression.
use super::*;

#[test]
#[ignore = "requires Apple7+ Metal; cross-tile carry and head-flag regression"]
fn issue_m4_04_metal_tiled_boundaries() {
    // Generated MSL on the developer source path.
    let mut device = MetalDevice::open_runtime_source().unwrap();
    let n = 131_111;
    // Odd, nonzero values prevent wrapping products from masking missing carries.
    let input = (0..n)
        .map(|i: u32| i.wrapping_mul(1664525).wrapping_add(1013904223) | 1)
        .collect::<Vec<_>>();
    let offsets = [
        0, 0, 1, 256, 256, 257, 513, 514, 777, 4097, 65_537, 65_538, n, n,
    ];
    for family in [
        seg::Family::Reduce(seg::Law::WrappingSum),
        seg::Family::Reduce(seg::Law::WrappingProduct),
        seg::Family::Compact {
            predicate: seg::Predicate::Less,
            cutoff: 1 << 31,
        },
    ] {
        let compact = matches!(family, seg::Family::Compact { .. });
        let bounds = if compact { &[][..] } else { &offsets[..] };
        let plan = seg::verify(
            seg::Ir {
                version: seg::VERSION,
                span: invar_device::Span { start: 1, end: 2 },
                source_identity: 1,
                graph_revision: 1,
                function_contract: [1; 32],
                family,
                elements: n,
                segments: if compact { 0 } else { offsets.len() as u32 - 1 },
            },
            device.caps,
            invar_device::Provenance {
                source_identity: 1,
                graph_revision: 1,
            },
        )
        .unwrap();
        let expected = seg::reference(&plan, &input, bounds).unwrap();
        let run = device.run_segmented(&plan, &input, bounds).unwrap();
        assert_eq!(run.output, expected, "family={:?}", plan.ir().family);
        println!(
            "boundary family={:?} words={} launches={} bit_exact=true",
            plan.ir().family,
            run.output.len(),
            plan.launches(&input, bounds).unwrap().len()
        );
    }
}
