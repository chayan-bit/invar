//! `invar build PATH --target metal [--entry ENTRY] --output FILE`: Apple's
//! offline Metal compiler runs here, at build time, and never during a run.
//! Without `--entry` the container serves `invar run` (every Metal region, the
//! calibration probe, and each grid, segmented or tensor kernel a default-fuel
//! run reaches on this Mac); with it, the ordinary-source ABI5 general-net entry.
//! `--family tensor --fuel FUEL` serves `invar source-device PATH tensor ENTRY
//! metal FUEL`: the probe plus each contraction an exact CPU run reaches on
//! this Mac's Metal capabilities.
use invar::{
    general_net::{compiled::Target, device_host},
    metal_aot::{self, Route},
};
use std::{ffi::OsString, path::Path};

pub fn run(path: &OsString, entry: Option<&OsString>, output: &OsString) -> i32 {
    emit(build(path, entry, output))
}

pub fn run_tensor(path: &OsString, entry: &OsString, fuel: &OsString, output: &OsString) -> i32 {
    emit(build_tensor(path, entry, fuel, output))
}

fn emit(result: Result<serde_json::Value, String>) -> i32 {
    let (document, code) = match result {
        Ok(receipt) => (
            serde_json::json!({"schema":"invar-metal-build-v1","library":receipt}),
            0,
        ),
        Err(error) => (
            serde_json::json!({"schema":"invar-metal-build-v1","error":error}),
            1,
        ),
    };
    println!("{document}");
    code
}

fn build(
    path: &OsString,
    entry: Option<&OsString>,
    output: &OsString,
) -> Result<serde_json::Value, String> {
    let source = admit(path)?;
    let (route, entry, kernels) = match entry {
        None => {
            let checked = invar::front::parse(&source)
                .and_then(|program| invar::check::check(program, true))
                .map_err(|e| format!("source:{e:?}"))?;
            let program = invar::mir::lower(&checked).map_err(|e| format!("lowering:{e:?}"))?;
            let mut kernels = metal_aot::run_kernels(&source, &program)?;
            kernels.extend(family_kernels(&source, &program)?);
            (Route::Run, "main", kernels)
        }
        Some(entry) => {
            let entry = entry.to_str().ok_or("invalid entry")?;
            let plan = device_host::Plan::from_source(
                &source,
                entry,
                Target::Metal,
                device_host::Limits::certified(),
            )
            .map_err(|e| format!("source admission:{e:?}"))?;
            let msl = plan.artifact().source();
            let functions = metal_aot::msl_entry_points(msl);
            if !functions.iter().any(|f| f == metal_aot::GNET_KERNEL) {
                return Err("generated MSL lacks the ABI5 entry point".into());
            }
            (Route::Gnet, entry, metal_aot::gnet_kernels(plan.artifact()))
        }
    };
    write(route, entry, &source, &kernels, output)
}

/// The family kernels a default-fuel `invar run` reaches on this Mac; a run
/// with more fuel may reach more, which the library then refuses, typed.
fn family_kernels(
    source: &[u8],
    program: &invar::mir::Program,
) -> Result<Vec<(String, Vec<String>)>, String> {
    if !invar::schedule::has_family_region(source, program)
        .map_err(|e| format!("admission:{e:?}"))?
    {
        return Ok(Vec::new());
    }
    reached_family_kernels(source, program)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn reached_family_kernels(
    source: &[u8],
    program: &invar::mir::Program,
) -> Result<Vec<(String, Vec<String>)>, String> {
    let caps = invar_metal::MetalDevice::open()
        .map_err(|e| format!("Metal capabilities:{e:?}"))?
        .capabilities();
    invar::schedule::metal::family_kernels(source, program, invar::eval::DEFAULT_FUEL, caps)
        .map_err(|e| format!("family kernels:{e:?}"))
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn reached_family_kernels(
    _: &[u8],
    _: &invar::mir::Program,
) -> Result<Vec<(String, Vec<String>)>, String> {
    Err("grid, segmented and tensor kernels need this Mac's Metal capabilities: build with the metal feature on macOS".into())
}

fn admit(path: &OsString) -> Result<Vec<u8>, String> {
    let source = crate::read_source(path).map_err(|e| format!("source:{e:?}"))?;
    if source.len() > 1024 * 1024 {
        return Err("source exceeds the build admission budget".into());
    }
    Ok(source)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn build_tensor(
    path: &OsString,
    entry: &OsString,
    fuel: &OsString,
    output: &OsString,
) -> Result<serde_json::Value, String> {
    let source = admit(path)?;
    let entry = entry.to_str().ok_or("invalid entry")?;
    let fuel = fuel
        .to_str()
        .and_then(|n| n.parse::<u64>().ok())
        .filter(|n| *n > 0)
        .ok_or("fuel must be a positive u64")?;
    let checked = invar::front::parse(&source)
        .and_then(|program| invar::check::check(program, true))
        .map_err(|e| format!("source:{e:?}"))?;
    invar::tensor::Contraction::from_checked(&checked, entry)
        .map_err(|e| format!("source contract:{e:?}"))?;
    let caps = invar_metal::MetalDevice::open()
        .map_err(|e| format!("Metal capabilities:{e:?}"))?
        .capabilities();
    let kernels = invar::tensor::host::metal::kernels(&source, entry, fuel, caps)?;
    write(Route::Tensor, entry, &source, &kernels, output)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn build_tensor(
    _: &OsString,
    _: &OsString,
    _: &OsString,
    _: &OsString,
) -> Result<serde_json::Value, String> {
    Err("a tensor Metal library needs this Mac's Metal capabilities: build with the metal feature on macOS".into())
}

fn write(
    route: Route,
    entry: &str,
    source: &[u8],
    kernels: &[(String, Vec<String>)],
    output: &OsString,
) -> Result<serde_json::Value, String> {
    let bytes =
        metal_aot::build::build(route, entry, source, kernels).map_err(|e| format!("{e:?}"))?;
    let receipt = metal_aot::decode(&bytes)
        .map_err(|e| format!("built library:{e:?}"))?
        .receipt();
    metal_aot::build::write_atomic(Path::new(output), &bytes)
        .map_err(|e| format!("output:{e:?}"))?;
    Ok(receipt)
}
