//! #48 C4: f32 `sin` and `cos` run `invar-trig-f32-v1` on every device path:
//! f32 source regions and the general net (oracle, CUDA and Metal emitters,
//! run here on the CPU device model), with the bits of the host helpers.
#![forbid(unsafe_code)]
#[path = "support/gnet_entry.rs"]
mod entry;
use invar::{
    check, eval, front,
    general_net::{compiled::Target, device_host::cpu::CpuModel},
};
use invar_device::trig_f32;

#[test]
fn issue_210_trig_f32_source_regions_complete_on_the_device_plan() {
    let source = "fn kernel(a: read Array<f32>, n: u32) -> Array<f32> { var out = filled_f32(n, 0.0); \
                  for i in 0u32..n { out[i] = fma(sin(a[i]), cos(a[i]), sqrt(a[i])); } return out; } \
                  fn main() -> i32 { let a: Array<f32> = [0.5, 3.0, 100000.0, 1000000000000000000000000000000.0, 0.0]; \
                  let out = kernel(a, 5u32); for i in 0u32..5u32 { if out[i] != fma(sin(a[i]), cos(a[i]), sqrt(a[i])) { return 0; } } return 7; }";
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    let reference = eval::observe(&checked, 1_000_000);
    assert_eq!(reference.result, Ok(eval::Value::I32(7)));
    for workers in [1, 4] {
        let run = invar::f32_source::run_cpu_source(source.as_bytes(), 1_000_000, workers).unwrap();
        assert!(
            run.completed >= 1,
            "no region completed: {:?}",
            run.refusals
        );
        assert!(run.refusals.is_empty(), "{:?}", run.refusals);
        assert_eq!(run.observation.result, reference.result);
        assert_eq!(run.observation.steps_used, reference.steps_used);
    }
}

#[test]
fn issue_210_trig_general_net_runs_trig_f32_v1_on_both_device_targets() {
    let xs: Vec<u32> = (0u32..256)
        .map(|i| (i << 24) | (i.wrapping_mul(0x9e3779b1) >> 8))
        .chain([
            0,
            0x8000_0000,
            0x3fc9_0fdb,
            0x4049_0fdb,
            0x7f80_0000,
            0x7fc0_0000,
        ])
        .collect();
    for (name, host) in [
        ("sin", trig_f32::host_sin as fn(u32) -> u32),
        ("cos", trig_f32::host_cos),
    ] {
        let source = format!(
            "fn run(a: read Array<f32>, b: mut Array<f32>) -> u32 {{ for i in 0u32..len(a) {{ b[i] = {name}(a[i]); }} return len(a); }} fn main() -> i32 {{ return 0; }}"
        );
        for target in [Target::Cuda, Target::Metal] {
            let args = vec![entry::f(&xs), entry::f(&vec![0; xs.len()])];
            let eval::Value::ArrayF32(out) = &args[1] else {
                panic!()
            };
            let out = std::rc::Rc::clone(out);
            assert_eq!(
                entry::agree(
                    &mut CpuModel::default(),
                    target,
                    &source,
                    "run",
                    args,
                    100_000
                ),
                invar::general_net::compiled::Status::Success
            );
            for (&x, y) in xs.iter().zip(out.borrow().iter()) {
                assert_eq!(y.to_bits(), host(x), "{target:?} {name} {x:#010x}");
            }
        }
    }
}
