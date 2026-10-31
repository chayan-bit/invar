//! Issue label M3-02.
use invar::{check::check, front::parse, mir};
use invar_device::Capabilities;

#[path = "../support/m3_02_cases.rs"]
mod m3_02_cases;
use m3_02_cases::{admitted, malformed_cases, observed_source, provenance};

#[test]
fn device_ir_verifier_admitted_graph_map_is_deterministic() {
    let ir = admitted();
    eprintln!("device_ir={ir:?}");
    let origin = provenance(&ir);
    let verified = invar_device::verify(ir, origin, Capabilities::cuda_12_9_sm89()).unwrap();
    let source = invar_device::emit_cuda(&verified).unwrap();
    assert_eq!(source, invar_device::emit_cuda(&verified).unwrap());
    assert!(source.contains("extern \"C\" __global__ void invar_map_v1"));
    assert!(source.contains("+ 7u"));
    assert_eq!(invar_device::source_hash(&source), 0xdbbc_b16d_ac2c_5e72);
}

#[test]
fn device_ir_verifier_reference_and_cpu_agree() {
    let source = observed_source();
    let checked = check(parse(source.as_bytes()).unwrap(), true).unwrap();
    assert_eq!(
        invar::eval::observe(&checked, 100_000).result,
        Ok(invar::eval::Value::I32(0))
    );
    let program = mir::lower(&checked).unwrap();
    assert_eq!(
        invar::schedule::run_source(source.as_bytes(), &program, 100_000, 4)
            .unwrap()
            .observation
            .result,
        Ok(invar::eval::Value::I32(0))
    );
}

#[test]
fn device_ir_verifier_malformed_matrix() {
    let base = admitted();
    for (label, expected, mutation) in malformed_cases() {
        let mut ir = base.clone();
        mutation(&mut ir);
        let actual =
            invar_device::verify(ir, provenance(&base), Capabilities::cuda_12_9_sm89()).err();
        assert_eq!(actual, Some(expected), "{label}");
        eprintln!("verifier_reject={label}: {expected:?}");
    }
}

#[test]
fn device_ir_verifier_boundaries_and_no_device_fallback() {
    let mut ir = admitted();
    ir.length = 0;
    ir.geometry.blocks = 1;
    assert!(
        invar_device::verify(ir.clone(), provenance(&ir), Capabilities::cuda_12_9_sm89()).is_ok()
    );
    let mut ir = admitted();
    ir.length = 1_000_001;
    assert!(
        invar_device::verify(ir.clone(), provenance(&ir), Capabilities::cuda_12_9_sm89()).is_err()
    );
    let ir = admitted();
    assert!(
        invar_device::verify(
            ir.clone(),
            provenance(&ir),
            Capabilities::Cuda {
                major: 12,
                minor: 4,
                sm: 86,
            }
        )
        .is_ok()
    );
    assert!(
        invar_device::verify(
            ir.clone(),
            provenance(&ir),
            Capabilities::Cuda {
                major: 12,
                minor: 3,
                sm: 89
            }
        )
        .is_err()
    );
    assert!(
        invar_device::verify(
            ir.clone(),
            provenance(&ir),
            Capabilities::Cuda {
                major: 12,
                minor: 4,
                sm: 49,
            }
        )
        .is_err()
    );
}
