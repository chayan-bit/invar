//! Fresh-process forced CUDA timing control for the M3-03 qualification suite.

use std::{path::Path, time::Instant};

use invar::{admission, check::check, device_ir, front::parse, mir};
use invar_cuda::cuda::CudaDevice;
use invar_device::Provenance;

fn run() -> Result<(), String> {
    let cold_started = Instant::now();
    let mut args = std::env::args();
    let _ = args.next();
    let source = args.next().ok_or("missing source".to_owned())?;
    let compiler = args.next().ok_or("missing NVRTC helper".to_owned())?;
    if source.len() > 16 * 1024 || args.next().is_some() {
        return Err("invalid cold-process arguments".into());
    }
    let checked = check(
        parse(source.as_bytes()).map_err(|e| format!("{e:?}"))?,
        true,
    )
    .map_err(|e| format!("{e:?}"))?;
    let program = mir::lower(&checked).map_err(|e| format!("{e:?}"))?;
    let report = admission::analyze(source.as_bytes(), &program).map_err(|e| format!("{e:?}"))?;
    let ir = device_ir::from_region(report.regions().first().ok_or("no map region")?, 4093)
        .map_err(|e| format!("{e:?}"))?;
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    let values = (0..4093)
        .map(|n| (n as u32).wrapping_mul(17).wrapping_add(1))
        .collect::<Vec<_>>();
    let mut device = CudaDevice::open().map_err(|e| format!("CUDA_UNAVAILABLE: {e:?}"))?;
    let verified =
        invar_device::verify(ir, origin, device.capabilities()).map_err(|e| format!("{e:?}"))?;
    let result = device
        .run_fused_map(&verified, &[&values], Path::new(&compiler))
        .map_err(|e| format!("CUDA_PENDING: {e:?}"))?;
    let first = values.iter().map(|n| n.wrapping_add(3)).collect::<Vec<_>>();
    let second = first.iter().map(|n| n.wrapping_mul(7)).collect::<Vec<_>>();
    let third = second
        .iter()
        .map(|n| n.wrapping_add(11))
        .collect::<Vec<_>>();
    if result.outputs != [first, second, third] {
        return Err("cold-process output differs from reference".into());
    }
    println!(
        "cold_device_ms={} compile_ms={} host_to_host_ms={} source_hash={:016x} completion=event-confirmed",
        cold_started.elapsed().as_secs_f64() * 1000.0,
        result.compile_ms,
        result.host_to_host_ms,
        result.source_hash
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
