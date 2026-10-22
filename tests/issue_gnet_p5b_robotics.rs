//! Physical robotics checks on the pinned corpus, with explicit compiler budgets.
#[path = "support/gnet_device.rs"]
mod device;
use invar::{
    eval::Value,
    general_net::{
        compiled::{self, Target},
        device_host::{Backend, Limits, Plan},
        oracle,
    },
};
use std::{fs, path::PathBuf};

fn cases<B: Backend>(backend: &mut B, target: Target) {
    let root = PathBuf::from(std::env::var_os("INVAR_P5B_CORPUS").expect("pinned corpus"))
        .join("robotics");
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let mut count = 0;
    for id in ["gnet05r-fk-00", "gnet05r-pointcloud-00"] {
        let row = manifest["programs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["id"] == id)
            .unwrap();
        let source = fs::read(root.join(row["file"].as_str().unwrap())).unwrap();
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(&source)
            .iter()
            .map(|x| format!("{x:02x}"))
            .collect::<String>();
        assert_eq!(digest, row["source_sha256"].as_str().unwrap());
        let args = row["oracle_inputs"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| Value::U32(v.as_u64().unwrap().try_into().unwrap()))
            .collect::<Vec<_>>();
        let limits = Limits {
            compiler: compiled::Limits {
                values: 2048,
                places: 512,
                frame_bytes: 2 * 1024 * 1024,
                ..Default::default()
            },
            owned: compiled::OwnedLimits {
                elements: 2048,
                allocations: 128,
                cleanup_events: 512,
            },
            private: compiled::SpeculationLimits {
                slots: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        let plan = Plan::from_source(&source, "run", target, limits).unwrap();
        let config = compiled::DeviceConfig {
            fuel: 1_000_000,
            transitions: 1_000_000,
            trace_events: 500_000,
            quantum: 100_000,
            rounds: 64,
            capacity: 0,
        };
        let reference = oracle::observe_call(
            plan.net(),
            plan.proof(),
            "run",
            &args,
            oracle::Limits {
                fuel: config.fuel,
                transitions: config.transitions,
                trace_events: config.trace_events,
                array_elements: limits.owned.elements,
            },
        )
        .unwrap();
        assert!(
            reference.semantic.result.is_ok(),
            "{id}: {:?}",
            reference.semantic.result
        );
        let (got, _) = device::drive(&plan, backend, &args, config).unwrap();
        device::agree(id, &plan, &got, &reference);
        let Some(Value::ArrayF32(result)) = got.result else {
            panic!("{id}: f32 owner result");
        };
        println!(
            "P5B_ROBOTICS_RUNTIME {id} {target:?} {} words, {} source steps, bit-exact",
            result.borrow().len(),
            got.steps
        );
        count += 1;
    }
    assert_eq!(count, 2);
}

#[test]
#[ignore = "explicit pinned robotics corpus and CPU group model"]
fn robotics_cpu_model() {
    cases(
        &mut invar::general_net::device_host::cpu::CpuModel::default(),
        Target::Metal,
    );
}
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "physical Metal and explicit pinned robotics corpus"]
fn robotics_metal() {
    cases(
        &mut invar::general_net::device_host::metal::Metal::open_runtime_source().unwrap(),
        Target::Metal,
    );
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "physical CUDA and explicit pinned robotics corpus"]
fn robotics_cuda() {
    let compiler = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(format!("invar-nvrtc{}", std::env::consts::EXE_SUFFIX));
    cases(
        &mut invar::general_net::device_host::cuda_nvrtc::CudaNvrtc::open(&compiler).unwrap(),
        Target::Cuda,
    );
}

#[test]
#[ignore = "explicit robotics snapshot; structural admission diagnosis"]
fn robotics_structural_usage() {
    use invar::{
        check, front,
        general_net::{self, Net},
    };
    use std::collections::BTreeSet;
    let root = PathBuf::from(std::env::var_os("INVAR_P5B_CORPUS").expect("pinned corpus"))
        .join("robotics");
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
    for row in manifest["programs"].as_array().unwrap() {
        let source = fs::read(root.join(row["file"].as_str().unwrap())).unwrap();
        let checked = check::check(front::parse(&source).unwrap(), false).unwrap();
        let net = Net::lower(&checked, general_net::Limits::default()).unwrap();
        let proof = net.verify().unwrap();
        let mir = net.compiled_mir(&proof).unwrap();
        let mut values = 0;
        let mut places = 0;
        let mut states = 0;
        for function in &mir.functions {
            let ids = function
                .blocks
                .iter()
                .flat_map(|b| {
                    b.params
                        .iter()
                        .chain(b.instructions.iter().filter_map(|i| i.output.as_ref()))
                })
                .map(|(id, _)| *id)
                .collect::<BTreeSet<_>>();
            values = values.max(ids.len());
            places = places.max(function.locals.len());
            states += function
                .blocks
                .iter()
                .map(|b| b.instructions.len() + 1)
                .sum::<usize>();
        }
        println!(
            "P5B_ROBOTICS_USAGE {}",
            serde_json::json!({"id":row["id"],"values":values,"places":places,"states":states,"frame_bytes":(values+places+4)*compiled::MAX_FRAMES*8})
        );
    }
}
