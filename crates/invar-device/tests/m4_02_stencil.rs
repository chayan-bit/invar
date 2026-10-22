use invar_device::{
    Capabilities, Provenance, Span,
    floating::{self, Buffer},
    stencil::{self, Boundary, Error, Ir},
};
fn origin() -> Provenance {
    Provenance {
        source_identity: 71,
        graph_revision: 3,
    }
}
fn ir(dims: [u32; 2], boundary: Boundary) -> Ir {
    let make = |id, offset| Buffer {
        id,
        elements: u64::from(dims[0]) * u64::from(dims[1]) + offset + 8,
        shape: dims,
        strides: [u64::from(dims[1]), 1],
        offset,
        alignment: 4,
    };
    Ir {
        version: 1,
        numeric_version: 1,
        numeric_policy: floating::NUMERIC_POLICY.into(),
        layout_version: 1,
        boundary_version: 1,
        boundary,
        span: Span { start: 7, end: 93 },
        source_identity: 71,
        graph_revision: 3,
        function_contract: [42; 32],
        dimensions: dims,
        input: make(1, 3),
        output: make(2, 5),
    }
}
fn admit(ir: Ir) -> stencil::Verified {
    stencil::verify(ir, Capabilities::cuda_12_9_sm89(), origin()).unwrap()
}
#[test]
fn issue_m4_02_transpose_stencil_schedules_physical_mutations() {
    for change in 0..20 {
        let mut p = ir([3, 5], Boundary::Clamp);
        let error = match change {
            0 => {
                p.version += 1;
                Error::Version
            }
            1 => {
                p.numeric_version += 1;
                Error::Numeric
            }
            2 => {
                p.numeric_policy = "relaxed".into();
                Error::Numeric
            }
            3 => {
                p.layout_version += 1;
                Error::Layout
            }
            4 => {
                p.boundary_version += 1;
                Error::Boundary
            }
            5 => {
                p.span.end = p.span.start;
                Error::Source
            }
            6 => {
                p.source_identity = 0;
                Error::Source
            }
            7 => {
                p.source_identity += 1;
                Error::Graph
            }
            8 => {
                p.graph_revision += 1;
                Error::Graph
            }
            9 => {
                p.function_contract = [0; 32];
                Error::Contract
            }
            10 => {
                p.output.id = p.input.id;
                Error::Alias
            }
            11 => {
                p.input.shape[0] += 1;
                Error::Shape
            }
            12 => {
                p.output.shape[1] += 1;
                Error::Shape
            }
            13 => {
                p.input.strides[0] += 1;
                Error::Layout
            }
            14 => {
                p.output.strides[1] = 0;
                Error::Layout
            }
            15 => {
                p.input.alignment = 2;
                Error::Alignment
            }
            16 => {
                p.output.alignment = 3;
                Error::Alignment
            }
            17 => {
                p.input.elements = p.input.offset + 14;
                Error::Bounds
            }
            18 => {
                p.output.offset = u64::MAX;
                Error::Bounds
            }
            19 => {
                p.input.elements = floating::MAX_ELEMENTS + 1;
                Error::Bounds
            }
            _ => unreachable!(),
        };
        assert_eq!(
            stencil::verify(p, Capabilities::cuda_12_9_sm89(), origin()).unwrap_err(),
            error
        );
    }
    let metal = Capabilities::Metal {
        apple_family: 9,
        msl_major: 3,
        msl_minor: 1,
    };
    assert_eq!(
        stencil::verify(ir([1, 1], Boundary::Clamp), metal, origin()).unwrap_err(),
        Error::Capabilities
    );
    let p = admit(ir([3, 5], Boundary::Clamp));
    for (block, grid) in [
        ([16, 15, 1], [1, 1, 1]),
        ([16, 16, 2], [1, 1, 1]),
        ([16, 16, 1], [0, 1, 1]),
        ([16, 16, 1], [1, 2, 1]),
    ] {
        assert_eq!(p.validate_launch(block, grid), Err(Error::Geometry));
    }
    assert_eq!(p.validate_launch([16, 16, 1], [3, 1, 1]), Ok(()));
}
#[test]
fn issue_m4_02_transpose_stencil_schedules_independent_halo_maps() {
    for dims in [
        [0, 33],
        [33, 0],
        [1, 1],
        [1, 33],
        [33, 1],
        [3, 5],
        [31, 33],
        [127, 257],
    ] {
        for boundary in [Boundary::ConstantZero, Boundary::Clamp] {
            let p = admit(ir(dims, boundary));
            for r in 0..dims[0] {
                for c in 0..dims[1] {
                    let expected = [
                        (r as i64, c as i64),
                        (r as i64 - 1, c as i64),
                        (r as i64 + 1, c as i64),
                        (r as i64, c as i64 - 1),
                        (r as i64, c as i64 + 1),
                    ]
                    .map(|(y, x)| {
                        if boundary == Boundary::ConstantZero
                            && (y < 0 || x < 0 || y >= dims[0] as i64 || x >= dims[1] as i64)
                        {
                            None
                        } else {
                            let y = y.max(0).min(dims[0] as i64 - 1) as u64;
                            let x = x.max(0).min(dims[1] as i64 - 1) as u64;
                            Some(3 + y * dims[1] as u64 + x)
                        }
                    });
                    assert_eq!(p.reads([r, c]).unwrap(), expected);
                }
            }
            assert_eq!(p.reads(dims), Err(Error::Index));
            let cuda = stencil::emit_cuda(&p).unwrap();
            if dims.contains(&0) {
                assert!(!cuda.contains("input["));
                assert!(!cuda.contains("__syncthreads"));
            } else {
                assert!(cuda.contains("tile[18][19]"));
                assert_eq!(cuda.matches("__syncthreads();").count(), 2);
                assert!(cuda.find("return;").unwrap() < cuda.find("__syncthreads();").unwrap());
                assert!(cuda.contains("h<324; h+=256"));
                assert!(cuda.contains("__fadd_rn(a,b)"));
                assert!(!cuda.contains("__fmaf"));
                assert!(p.nvrtc_options().contains(&"--fmad=false".into()));
                assert!(p.nvrtc_options().contains(&"--ftz=false".into()));
            }
            println!(
                "boundary={} dims={dims:?} cuda_hash={} static_only",
                boundary.name(),
                invar_device::source_hash(&cuda)
            );
        }
    }
}

#[test]
fn issue_m4_02_stencil_cap_2_28_physical_verifier() {
    for dims in [[16384, 16384], [1, 1 << 28], [1 << 28, 1]] {
        let p = admit(ir(dims, Boundary::Clamp));
        assert_eq!(
            p.reads([dims[0] - 1, dims[1] - 1]).unwrap()[0],
            Some(3 + (1 << 28) - 1)
        );
    }
    assert_eq!(
        stencil::verify(
            ir([1, (1 << 28) + 1], Boundary::Clamp),
            Capabilities::cuda_12_9_sm89(),
            origin()
        )
        .unwrap_err(),
        Error::Bounds
    );
}
