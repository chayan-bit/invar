//! Issue #14 (L1-06).
#![forbid(unsafe_code)]
use invar::{
    eval::{
        EvalError,
        host_read::{HostArgument, HostMutArray, HostReadArray},
    },
    exports::{ExportEngine, ExportError, ScalarValue, compile_exports},
};

const SOURCE: &str = r#"
fn fcopy(xs: read Array<f32>, out: mut Array<f32>) -> u32 {
    for i in 0u32..len(xs) { out[i] = xs[i]; } return len(out);
}
fn arithmetic(xs: read Array<f32>, out: mut Array<f32>) -> u32 {
    for i in 0u32..len(xs) { out[i] = xs[i] * 0.5; } return len(out);
}
fn ordered(xs: read Array<f32>, out: mut Array<f32>) -> u32 {
    out[0u32] = xs[0u32] * xs[1u32] - 1.0;
    out[1u32] = fma(xs[0u32], xs[1u32], -1.0);
    out[2u32] = f32_from_i32(-7i32) + f32_from_u32(3u32);
    out[3u32] = -xs[2u32]; return len(out);
}
fn poke(xs: mut Array<f32>, i: u32) -> u32 {
    xs[0u32] = 1.0; xs[i] = -0.0; return len(xs);
}
fn depth(n: u32) -> u32 { return depth(n + 1u32); }
fn fail(xs: mut Array<f32>) -> u32 { xs[0u32] = 2.0; return depth(0u32); }
fn bump(xs: mut Array<f32>) -> u32 {
    for i in 0u32..len(xs) { xs[i] = xs[i] + 1.0; } return len(xs);
}
"#;
const NAMES: &[&str] = &["fcopy", "arithmetic", "ordered", "poke", "fail", "bump"];
const BITS: [u32; 12] = [
    0x7f800001, 0xffc12345, 0x7fc54321, 0x80000000, 0, 0x7f800000, 0xff800000, 1, 2, 0x007fffff,
    0x00800000, 0x80800000,
];
fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|x| x.to_bits()).collect()
}
fn engines() -> Vec<ExportEngine> {
    let engines = vec![ExportEngine::Reference];
    #[cfg(feature = "native")]
    let engines = [engines, vec![ExportEngine::NativeSerial]].concat();
    engines
}

#[test]
fn f32_same_address_exact_payloads_empty_and_maximum() {
    for engine in engines() {
        let mut module = compile_exports(SOURCE.as_bytes(), "f32.inv", NAMES, engine).unwrap();
        for size in [0, BITS.len(), 1_000_000] {
            let xs: Vec<f32> = (0..size)
                .map(|i| f32::from_bits(BITS[i % BITS.len()]))
                .collect();
            let mut out = vec![f32::from_bits(0xdeadbeef); size];
            let addresses = if size == 0 {
                vec![0, 0]
            } else {
                vec![xs.as_ptr().addr(), out.as_ptr().addr()]
            };
            let mut args = [
                HostArgument::Read(HostReadArray::Float32(&xs)),
                HostArgument::Mut(HostMutArray::Float32(&mut out)),
            ];
            let receipt = module
                .invoke_arrays("fcopy", &mut args, 10_000_000)
                .unwrap();
            assert_eq!(receipt.value, ScalarValue::U32(size as u32));
            assert_eq!(bits(&xs), bits(&out));
            assert_eq!(receipt.zero_copy, engine == ExportEngine::NativeSerial);
            assert_eq!(
                receipt.record_addresses,
                if receipt.zero_copy { addresses } else { vec![] }
            );
        }
        let max = match engine {
            ExportEngine::Reference => invar::array_limits::MAX_ARRAY_LEN,
            ExportEngine::NativeSerial => invar_foreign::host_array::MAX_ELEMENTS,
        };
        let xs = vec![0.0f32; max + 1];
        let mut out = [f32::from_bits(BITS[0])];
        let mut args = [
            HostArgument::Read(HostReadArray::Float32(&xs)),
            HostArgument::Mut(HostMutArray::Float32(&mut out)),
        ];
        let error = module.invoke_arrays("fcopy", &mut args, 100).unwrap_err();
        assert!(matches!(
            error,
            ExportError::Evaluation(EvalError::Tool {
                code: "ARRAY_SIZE_LIMIT",
                ..
            })
        ));
        assert_eq!(bits(&out), [BITS[0]]);
    }
}

#[test]
fn f32_strict_arithmetic_subnormals_and_no_contraction() {
    let input: Vec<f32> = BITS[3..].iter().map(|&x| f32::from_bits(x)).collect();
    let mut expected = vec![];
    for engine in engines() {
        let mut module = compile_exports(SOURCE.as_bytes(), "f32.inv", NAMES, engine).unwrap();
        let mut out = vec![0.0; input.len()];
        let mut args = [
            HostArgument::Read(HostReadArray::Float32(&input)),
            HostArgument::Mut(HostMutArray::Float32(&mut out)),
        ];
        module.invoke_arrays("arithmetic", &mut args, 1000).unwrap();
        if engine == ExportEngine::Reference {
            expected = bits(&out);
        }
        assert_eq!(bits(&out), expected);
        assert_eq!(out[5].to_bits(), 1, "2 * 0.5 must retain a subnormal");
        assert_eq!(
            out[7].to_bits(),
            0x00400000,
            "normal * 0.5 must underflow gradually"
        );
        assert_eq!(out[8].to_bits(), 0x80400000);
        let input = [f32::from_bits(0x3f800001), f32::from_bits(0x3f7ffffe), -0.0];
        let mut out = [0.0; 4];
        let mut args = [
            HostArgument::Read(HostReadArray::Float32(&input)),
            HostArgument::Mut(HostMutArray::Float32(&mut out)),
        ];
        module.invoke_arrays("ordered", &mut args, 1000).unwrap();
        assert_eq!(bits(&out), [0, 0xa8800000, (-4.0f32).to_bits(), 0]);
    }
    #[cfg(feature = "native")]
    {
        let checked = invar::check::check(
            invar::front::parse(format!("{SOURCE} fn main() -> i32 {{ return 0; }}").as_bytes())
                .unwrap(),
            false,
        )
        .unwrap();
        let program = invar::mir::lower(&checked).unwrap();
        let object = invar::native::emit_mir(&program).unwrap();
        assert!(object.ir.contains("\"denormal-fp-math-f32\"=\"ieee,ieee\""));
        assert!(!object.ir.contains("positive-zero"));
        for flag in [
            " fast ",
            " reassoc ",
            " contract ",
            " afn ",
            " nnan ",
            " ninf ",
            " nsz ",
            " arcp ",
        ] {
            assert!(!object.ir.contains(flag), "strict LLVM IR contains {flag}");
        }
    }
}

#[test]
fn f32_all_fuel_prefixes_trap_and_tool_failure_restore_bits() {
    for engine in engines() {
        let mut module = compile_exports(SOURCE.as_bytes(), "f32.inv", NAMES, engine).unwrap();
        let mut oracle =
            compile_exports(SOURCE.as_bytes(), "f32.inv", NAMES, ExportEngine::Reference).unwrap();
        let mut completed = false;
        let input = BITS.map(f32::from_bits);
        for fuel in 1..=150 {
            let mut outcomes = Vec::new();
            for artifact in [&mut module, &mut oracle] {
                let mut out = BITS.map(f32::from_bits);
                out.reverse();
                let before = bits(&out);
                let mut args = [
                    HostArgument::Read(HostReadArray::Float32(&input)),
                    HostArgument::Mut(HostMutArray::Float32(&mut out)),
                ];
                let result = artifact.invoke_arrays("fcopy", &mut args, fuel);
                let status = match result {
                    Ok(value) => {
                        completed = true;
                        assert_eq!(bits(&out), BITS);
                        serde_json::json!({"result": format!("{:?}", value.value)})
                    }
                    Err(error) => {
                        assert_eq!(bits(&out), before, "fuel={fuel}, {engine:?}");
                        artifact.diagnostics(&error)
                    }
                };
                outcomes.push((status, bits(&out)));
            }
            assert_eq!(outcomes[0], outcomes[1], "fuel={fuel}, {engine:?}");
        }
        assert!(completed);
        for (name, index, code) in [
            ("poke", 99, "INDEX_OUT_OF_BOUNDS"),
            ("fail", 0, "CALL_DEPTH_LIMIT"),
        ] {
            let mut xs = BITS.map(f32::from_bits);
            let mut args = vec![HostArgument::Mut(HostMutArray::Float32(&mut xs))];
            if name == "poke" {
                args.push(HostArgument::U32(index));
            }
            let error = module.invoke_arrays(name, &mut args, 10000).unwrap_err();
            let document = module.diagnostics(&error);
            assert_eq!(document["diagnostics"][0]["code"], code);
            drop(args);
            assert_eq!(bits(&xs), BITS, "{engine:?}, {name}");
        }
    }
}

#[test]
fn f32_take_and_public_scalar_tags_stay_refused() {
    for engine in engines() {
        for source in [
            "fn f(xs: take Array<f32>) -> u32 { return len(xs); }",
            "fn f(x: f32) -> u32 { return 1u32; }",
            "fn f(xs: read Array<f32>) -> f32 { return xs[0u32]; }",
            "fn f(xs: read Array<f32>) -> Array<f32> { return filled_f32(len(xs), 0.0); }",
        ] {
            let error = compile_exports(source.as_bytes(), "refused.inv", &["f"], engine)
                .err()
                .unwrap();
            assert!(matches!(
                error,
                ExportError::Boundary {
                    code: "NATIVE_ABI",
                    ..
                }
            ));
        }
    }
}
