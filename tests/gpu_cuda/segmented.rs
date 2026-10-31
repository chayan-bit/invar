//! M4-04 CUDA qualification: the shared segmented tests plus the CUDA admission refusals.
use invar_cuda::{Error, cuda::CudaDevice};
use invar_device::segmented as seg;
use std::path::Path;

#[path = "../support/m4_04_cases.rs"]
mod cases;

type Device = CudaDevice;

fn compiler() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_invar-nvrtc"))
}

fn open() -> Device {
    CudaDevice::open().expect("explicit CUDA qualification: no usable device or NVRTC")
}

fn run_segmented(
    device: &mut Device,
    plan: &seg::Verified,
    input: &[u32],
    offsets: &[u32],
) -> Result<body::Out, String> {
    let r = device
        .run_segmented(plan, input, offsets, compiler())
        .map_err(|e| format!("{e:?}"))?;
    Ok(body::Out {
        output: r.output,
        kernel_ms: f64::from(r.kernel_ms),
        compile_ms: r.compile_ms,
        transfer_bytes: r.transfer_bytes,
        source_hash: r.source_hash,
    })
}

#[path = "../support/gpu_m4_04.rs"]
mod body;

#[test]
#[ignore = "requires an authorized CUDA device and NVRTC 12.4+"]
fn gpu_admission_refusals_launch_nothing() {
    let mut device = open();
    let caps = device.capabilities();
    let case = cases::cases(false)
        .into_iter()
        .find(|case| case.name == "Sum/empty-interior")
        .unwrap();
    let plan = cases::plan(&case, caps);
    let mut decreasing = case.offsets.clone();
    // [0, 3, 3, ...] becomes [0, 5, 3, ...]: decreasing offsets.
    decreasing[1] = decreasing[2] + 2;
    for (input, offsets) in [
        (&case.input[..], &decreasing[..]),
        (&case.input[1..], &case.offsets[..]),
    ] {
        assert_eq!(
            device
                .run_segmented(&plan, input, offsets, compiler())
                .unwrap_err(),
            Error::Limit
        );
    }
    assert_eq!(device.quarantine_len(), 0);
    let result = device
        .run_segmented(&plan, &case.input, &case.offsets, compiler())
        .unwrap();
    assert_eq!(result.output, cases::expected(&case));
}
