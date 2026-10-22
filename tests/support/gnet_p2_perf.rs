//! Author-written flat-map baseline through the same physical capsule adapter.
//! Diagnostic only: both paths prepare on every call; no resident/trace-off API.
use super::*;
use std::time::Instant;
const SOURCE: &str = "fn run(a:mut Array<i32>,n:u32)->i32{for i in 0u32..n{a[i]=a[i]+1;}return 4;}fn main()->i32{return 0;}";
fn handwritten(target: Target) -> String {
    let args = (0..12)
        .map(|i| match target {
            Target::Cuda => format!("unsigned long long* b{i}"),
            Target::Cpu => panic!("timing probe requires a physical GPU"),
            Target::Metal => format!("device ulong* b{i} [[buffer({i})]]"),
        })
        .collect::<Vec<_>>()
        .join(",");
    match target {
        Target::Cpu => panic!("timing probe requires a physical GPU"),
        Target::Cuda => format!(
            "extern \"C\" __global__ void invar_gnet_body_map_v1({args}){{unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<b3[0])b1[i]=b0[i]+1;}}\nextern \"C\" __global__ void invar_gnet_body_resolve_v1({args}){{}}"
        ),
        Target::Metal => format!(
            "#include <metal_stdlib>\nusing namespace metal;\nkernel void invar_gnet_body_map_v1({args},uint i [[thread_position_in_grid]]){{if(i<b3[0])b1[i]=b0[i]+1;}}\nkernel void invar_gnet_body_resolve_v1({args}){{}}"
        ),
    }
}
fn measure<B: RegionBackend>(
    target: Target,
    b: &mut B,
    mut baseline: impl FnMut(&str, &device_host::host_frame::Image) -> device_host::host_frame::Image,
) {
    let session: u32 = std::env::var("INVAR_P2_SESSION").unwrap().parse().unwrap();
    assert!((1..=3).contains(&session));
    let n = 32;
    let mut p = plan(SOURCE, target);
    // Validate the actual generated operation against the independent oracle
    // before collecting timing. Every timed output is also checked below.
    agree(
        &mut p,
        b,
        vec![array(vec![1; n]), Value::U32(n as u32)],
        config(),
    );
    let source = handwritten(target);
    let mut state = u64::from(session);
    for pair in 0..40 {
        let mut times = [0u128; 2];
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let order = if state >> 63 == 0 { [0, 1] } else { [1, 0] };
        for path in order {
            let start = Instant::now();
            if path == 0 {
                let a = Rc::new(RefCell::new(vec![1; n]));
                let report = p
                    .run(
                        b,
                        &[Value::ArrayI32(Rc::clone(&a)), Value::U32(n as u32)],
                        config(),
                    )
                    .unwrap();
                times[path] = start.elapsed().as_nanos();
                assert_eq!(report.observation.status, compiled::Status::Success);
                assert_eq!(report.observation.result, Some(Value::I32(4)));
                assert!(report.launches > 0);
                assert_eq!(*a.borrow(), vec![2; n]);
            } else {
                let mut image: device_host::host_frame::Image = std::array::from_fn(|_| vec![0]);
                image[0] = vec![1; n];
                image[1] = vec![0; n];
                image[3] = vec![n as u64, 0, 0, 0, 0, 0];
                let result = baseline(&source, &image);
                times[path] = start.elapsed().as_nanos();
                assert_eq!(result[1], vec![2; n]);
            }
        }
        if pair >= 10 {
            eprintln!(
                "P2_TIMING {}",
                serde_json::json!({"session":session,"pair":pair-10,"n":n,"generated_ns":times[0],"handwritten_ns":times[1],"ratio":times[0] as f64/times[1] as f64})
            );
        }
    }
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "physical CUDA timing; one locked session, INVAR_P2_SESSION=1..3"]
fn p2_cuda_handwritten_diagnostic() {
    let compiler = std::path::PathBuf::from(std::env::var_os("INVAR_NVRTC").unwrap());
    let mut raw = invar_cuda::cuda::CudaDevice::open().unwrap();
    measure(
        Target::Cuda,
        &mut device_host::cuda_nvrtc::CudaNvrtc::open(&compiler).unwrap(),
        |source, image| raw.run_gnet_region(source, image, 1, &compiler).unwrap(),
    );
}
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "physical Metal timing; one locked session, INVAR_P2_SESSION=1..3"]
fn p2_metal_handwritten_diagnostic() {
    let mut raw = invar_metal::MetalDevice::open_runtime_source().unwrap();
    measure(
        Target::Metal,
        &mut device_host::metal::Metal::open_runtime_source().unwrap(),
        |source, image| raw.run_gnet_region(source, image, 1).unwrap(),
    );
}
