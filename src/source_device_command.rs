//! Explicit private ordinary-source placement qualification, no export ABI tags.
use invar::{
    check,
    eval::{Observation, Value},
    front, tensor, transpose_source,
};
use invar_cuda::Runtime;
use invar_device::Capabilities;
use sha2::{Digest, Sha256};
use std::{ffi::OsString, io::Read};

pub fn command(args: &[OsString]) -> i32 {
    match execute(args) {
        Ok((report, success)) => {
            println!("{report}");
            if success { 0 } else { 1 }
        }
        Err(error) => {
            println!(
                "{}",
                serde_json::json!({"schema":"invar-private-source-device-v1","error":error})
            );
            1
        }
    }
}
fn report(
    source: &[u8],
    family: &str,
    entry: &str,
    backend: &str,
    caps: Capabilities,
    run: (Observation, usize, Vec<&str>),
) -> (serde_json::Value, bool) {
    report_with_probe(source, family, entry, backend, caps, run, None)
}
fn report_with_probe(
    source: &[u8],
    family: &str,
    entry: &str,
    backend: &str,
    caps: Capabilities,
    run: (Observation, usize, Vec<&str>),
    probe: Option<String>,
) -> (serde_json::Value, bool) {
    let (observation, completed, refusals) = run;
    let success = observation.result.is_ok();
    let (result, error) = match observation.result {
        Ok(Value::I32(n)) => (serde_json::json!(n), serde_json::Value::Null),
        Ok(value) => (
            serde_json::Value::Null,
            serde_json::json!(format!("unexpected main result: {value:?}")),
        ),
        Err(error) => (
            serde_json::Value::Null,
            serde_json::json!(format!("{error:?}")),
        ),
    };
    let source_sha256 = Sha256::digest(source)
        .iter()
        .map(|n| format!("{n:02x}"))
        .collect::<String>();
    let mut doc = serde_json::json!({"schema":"invar-private-source-device-v1","scope":"source_reached_completed_publication","source_sha256":source_sha256,"family":family,"requested_entry":entry,"backend":backend,"capabilities":format!("{caps:?}"),"result":result,"error":error,"source_steps":observation.steps_used,"cleanup":format!("{:?}",observation.cleanup_trace),"completed_regions":completed,"refusals":refusals});
    if let Some(probe) = probe {
        doc["numeric_probe"] = serde_json::json!(probe);
    }
    (doc, success)
}
fn execute(args: &[OsString]) -> Result<(serde_json::Value, bool), String> {
    let artifact = match args {
        [_, _, _, _, _] => None,
        [_, _, _, _, _, flag, file] if flag == "--artifact" => Some(file),
        _ => {
            return Err(
                "expected PATH tensor|floating|transpose ENTRY cpu-model|cuda|metal FUEL [--artifact FILE]"
                    .into(),
            );
        }
    };
    let family = args[1].to_str().ok_or("invalid family")?;
    let entry = args[2].to_str().ok_or("invalid entry")?;
    let backend = args[3].to_str().ok_or("invalid backend")?;
    let fuel = args[4]
        .to_str()
        .and_then(|n| n.parse::<u64>().ok())
        .filter(|n| *n > 0)
        .ok_or("fuel must be a positive u64")?;
    if !matches!(family, "tensor" | "floating" | "transpose")
        || !matches!(backend, "cpu-model" | "cuda" | "metal")
        || (family, backend) == ("transpose", "metal")
    {
        return Err("unsupported family or backend".into());
    }
    if artifact.is_some() && (family, backend) != ("tensor", "metal") {
        return Err("--artifact applies only to the tensor family on metal".into());
    }
    let mut source = Vec::new();
    std::fs::File::open(&args[0])
        .map_err(|e| e.to_string())?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut source)
        .map_err(|e| e.to_string())?;
    if source.len() > 1024 * 1024 {
        return Err("source exceeds private admission budget".into());
    }
    // Source-invalid code and an unrecognized requested entry fail before GPU open.
    let checked = front::parse(&source)
        .and_then(|p| check::check(p, true))
        .map_err(|e| format!("source:{e:?}"))?;
    match family {
        "floating" => {
            invar::f32_source::Region::from_checked(&checked, entry)
                .map_err(|e| format!("source contract:{e:?}"))?;
        }
        "tensor" => {
            tensor::Contraction::from_checked(&checked, entry)
                .map_err(|e| format!("source contract:{e:?}"))?;
        }
        _ => {
            transpose_source::Contract::from_checked(&checked, entry)
                .map_err(|e| format!("source contract:{e:?}"))?;
        }
    }
    let model_caps = Capabilities::cuda_12_9_sm89();
    match (family, backend) {
        ("floating", "cpu-model") => {
            let run = invar::f32_source::run_cpu_source(&source, fuel, 4)
                .map_err(|e| format!("host:{e:?}"))?;
            Ok(report(
                &source,
                family,
                entry,
                backend,
                model_caps,
                (run.observation, run.completed, run.refusals),
            ))
        }
        ("floating", "cuda") => cuda_floating(&source, entry, fuel),
        ("floating", "metal") => metal_floating(&source, entry, fuel),
        ("tensor", "cpu-model") => {
            let mut runtime = Runtime::new(tensor::host::CpuDevice, 2);
            let run = tensor::host::run_source(&source, fuel, &mut runtime, model_caps)
                .map_err(|e| format!("host:{e:?}"))?;
            Ok(report(
                &source,
                family,
                entry,
                backend,
                model_caps,
                (run.observation, run.completed, run.refusals),
            ))
        }
        ("transpose", "cpu-model") => {
            let mut runtime = Runtime::new(transpose_source::host::CpuDevice, 2);
            let run = transpose_source::host::run_source(&source, fuel, &mut runtime, model_caps)
                .map_err(|e| format!("host:{e:?}"))?;
            Ok(report(
                &source,
                family,
                entry,
                backend,
                model_caps,
                (run.observation, run.completed, run.refusals),
            ))
        }
        ("tensor", "cuda") => cuda_tensor(&source, entry, fuel),
        ("tensor", "metal") => metal_tensor(&source, entry, fuel, artifact),
        ("transpose", "cuda") => cuda_transpose(&source, entry, fuel),
        _ => Err("unsupported source route".into()),
    }
}

#[cfg(feature = "cuda")]
fn cuda_floating(
    source: &[u8],
    entry: &str,
    fuel: u64,
) -> Result<(serde_json::Value, bool), String> {
    let compiler = super::nvrtc_helper();
    let mut device =
        invar_cuda::cuda::CudaDevice::open().map_err(|e| format!("forced CUDA:{e:?}"))?;
    let caps = device.capabilities();
    let probe = device
        .strict_f32_probe(&compiler)
        .map_err(|e| format!("probe:{e:?}"))?
        .summary();
    let run = invar::f32_source::run_cuda_source(source, fuel, &mut device, &compiler)
        .map_err(|e| format!("host:{e:?}"))?;
    Ok(report_with_probe(
        source,
        "floating",
        entry,
        "cuda",
        caps,
        (run.observation, run.completed, run.refusals),
        Some(probe),
    ))
}
#[cfg(not(feature = "cuda"))]
fn cuda_floating(_: &[u8], _: &str, _: u64) -> Result<(serde_json::Value, bool), String> {
    Err("forced CUDA requires the cuda feature".into())
}
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_floating(
    source: &[u8],
    entry: &str,
    fuel: u64,
) -> Result<(serde_json::Value, bool), String> {
    let mut device = invar_metal::MetalDevice::open().map_err(|e| format!("forced Metal:{e:?}"))?;
    let caps = device.capabilities();
    let probe = device
        .strict_f32_probe()
        .map_err(|e| format!("probe:{e:?}"))?
        .summary();
    let run = invar::f32_source::run_metal_source(source, fuel, &mut device)
        .map_err(|e| format!("host:{e:?}"))?;
    Ok(report_with_probe(
        source,
        "floating",
        entry,
        "metal",
        caps,
        (run.observation, run.completed, run.refusals),
        Some(probe),
    ))
}
#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_floating(_: &[u8], _: &str, _: u64) -> Result<(serde_json::Value, bool), String> {
    Err("forced Metal requires the metal feature on macOS".into())
}
#[cfg(feature = "cuda")]
fn cuda_tensor(source: &[u8], entry: &str, fuel: u64) -> Result<(serde_json::Value, bool), String> {
    let mut device = tensor::host::cuda::Cuda::open(source, entry, &super::nvrtc_helper())
        .map_err(|e| format!("forced CUDA:{e:?}"))?;
    let caps = device.capabilities();
    let probe = device
        .probe()
        .map_err(|e| format!("forced CUDA probe:{e:?}"))?
        .summary();
    let mut runtime = Runtime::new(device, 2);
    let run = tensor::host::run_source(source, fuel, &mut runtime, caps)
        .map_err(|e| format!("host:{e:?}"))?;
    Ok(report_with_probe(
        source,
        "tensor",
        entry,
        "cuda",
        caps,
        (run.observation, run.completed, run.refusals),
        Some(probe),
    ))
}
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_tensor(
    source: &[u8],
    entry: &str,
    fuel: u64,
    artifact: Option<&OsString>,
) -> Result<(serde_json::Value, bool), String> {
    use invar::schedule::metal::Kernels;
    let image = artifact
        .map(|file| invar::metal_aot::build::read(std::path::Path::new(file)))
        .transpose()
        .map_err(|e| format!("library admission:{e:?}"))?;
    let kernels = image
        .as_ref()
        .map_or_else(Kernels::developer, Kernels::Prebuilt);
    let device = tensor::host::metal::Metal::open(source, entry, kernels)
        .map_err(|e| format!("forced Metal:{e:?}"))?;
    let caps = device.capabilities();
    let probe = device.probe().summary();
    let mut runtime = Runtime::new(device, 2);
    let run = tensor::host::run_source(source, fuel, &mut runtime, caps)
        .map_err(|e| format!("host:{e:?}"))?;
    let (mut doc, success) = report_with_probe(
        source,
        "tensor",
        entry,
        "metal",
        caps,
        (run.observation, run.completed, run.refusals),
        Some(probe),
    );
    doc["library"] = image.map_or(serde_json::Value::Null, |image| image.receipt());
    Ok((doc, success))
}
#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_tensor(
    _source: &[u8],
    _entry: &str,
    _fuel: u64,
    _artifact: Option<&OsString>,
) -> Result<(serde_json::Value, bool), String> {
    Err("forced Metal requires the metal Cargo feature on macOS".into())
}
#[cfg(feature = "cuda")]
fn cuda_transpose(
    source: &[u8],
    entry: &str,
    fuel: u64,
) -> Result<(serde_json::Value, bool), String> {
    let device = transpose_source::host::cuda::Cuda::open(source, entry, &super::nvrtc_helper())
        .map_err(|e| format!("forced CUDA:{e:?}"))?;
    let caps = device.capabilities();
    let mut runtime = Runtime::new(device, 2);
    let run = transpose_source::host::run_source(source, fuel, &mut runtime, caps)
        .map_err(|e| format!("host:{e:?}"))?;
    Ok(report(
        source,
        "transpose",
        entry,
        "cuda",
        caps,
        (run.observation, run.completed, run.refusals),
    ))
}
#[cfg(not(feature = "cuda"))]
fn cuda_transpose(
    _source: &[u8],
    _entry: &str,
    _fuel: u64,
) -> Result<(serde_json::Value, bool), String> {
    Err("forced CUDA requires the cuda Cargo feature".into())
}
#[cfg(not(feature = "cuda"))]
fn cuda_tensor(
    _source: &[u8],
    _entry: &str,
    _fuel: u64,
) -> Result<(serde_json::Value, bool), String> {
    Err("forced CUDA requires the cuda Cargo feature".into())
}
