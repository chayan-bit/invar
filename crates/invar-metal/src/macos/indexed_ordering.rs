//! Real Metal frontier ordering regression.
use super::*;
use invar_device::{AddressSpace, Buffer, Layout, Span, indexed as ir};

#[test]
#[ignore = "real Metal adversarial ordered masks, cycles, self edges and round truncation"]
fn issue_m4_05_metal_ordering() {
    // Generated MSL on the developer source path.
    let mut device = MetalDevice::open_runtime_source().unwrap();
    let n = 7u32;
    let mut seed = 0x0405u32;
    for case in 0..32 {
        let mut edges = vec![0; n as usize * n as usize];
        for bit in &mut edges {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *bit = u32::from(seed >> 29 == 0);
        }
        // Force simultaneously active vertices with both directions and loops.
        for (u, v) in [
            (0, 1),
            (0, 2),
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (2, 6),
            (6, 0),
        ] {
            edges[u * n as usize + v] = 1;
        }
        for rounds in [0, 1, 2, 16] {
            let buf = |id, elements| Buffer {
                id,
                elements,
                address: AddressSpace::Device,
                layout: Layout::ContiguousU32,
            };
            let raw = ir::Ir {
                version: ir::VERSION,
                source_identity: 1,
                graph_revision: 0,
                source_contract: [1; 32],
                span: Span { start: 1, end: 2 },
                operation: ir::Operation::Frontier {
                    vertices: n,
                    rounds,
                    capacity: n,
                },
                values: buf(0, n * n),
                indices: buf(1, 0),
                output: buf(2, n * (rounds + 1)),
                law: None,
            };
            let bound = ir::verify(raw.clone(), &raw, device.caps)
                .unwrap()
                .bind(&edges, &[], 0)
                .unwrap();
            let mut expected = vec![0; n as usize * (rounds as usize + 1)];
            let mut front = vec![0; n as usize];
            front[0] = 1;
            for round in 0..rounds as usize {
                let mut next = vec![0; n as usize];
                for u in 0..n as usize {
                    if front[u] == 1 && expected[u] == 0 {
                        expected[u] = 1;
                        for v in 0..n as usize {
                            if edges[u * n as usize + v] == 1 && expected[v] == 0 {
                                next[v] = 1;
                            }
                        }
                    }
                }
                expected[(round + 1) * n as usize..(round + 2) * n as usize].copy_from_slice(&next);
                front = next;
            }
            let want = ir::decode(bound.plan(), expected, &[0, 0]).unwrap();
            let got = device.run_indexed(&bound).unwrap();
            assert_eq!(got.output, want, "case={case} rounds={rounds}");
            assert_eq!(got.completed_phases, rounds * 2);
        }
    }
    println!("128 adversarial frontier runs bit-exact, including every mask and stable map");
}
