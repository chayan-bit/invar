//! Issue #1 (M4-01).
#![forbid(unsafe_code)]
use std::{
    io::Write,
    process::{Command, Stdio},
};

// These refusal paths stop before any NVRTC symbol or GPU is accessed.
#[test]
fn private_nvrtc_modes_reject_nul_source_and_unknown_protocol_before_foreign_calls() {
    for mode in [
        None,
        Some("--gnet-v5"),
        Some("strict-f32-v1"),
        Some("strict-f32-probe-v1"),
        Some("transpose-bits-v1"),
        Some("unrecognized"),
    ] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_invar-nvrtc"));
        command.args(["12", "9", "89"]);
        if let Some(mode) = mode {
            command.arg(mode);
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        // The helper may refuse before reading stdin and close the pipe first;
        // a broken pipe is then expected, and every assertion below still applies.
        match child.stdin.take().unwrap().write_all(b"\0") {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::BrokenPipe => {}
            Err(error) => panic!("{mode:?}: {error}"),
        }
        let output = child.wait_with_output().unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.len() <= invar_cuda::cuda::MAX_ERROR_BYTES);
        assert!(error.ends_with('\n'));
        assert!(error.starts_with("NVRTC_COMPILE_FAILED:"), "{error}");
        if mode == Some("unrecognized") {
            assert!(error.contains("expected CUDA"), "{error}");
        } else {
            assert!(error.contains("Limit"), "{error}");
        }
    }
}

#[test]
#[ignore = "requires a real CUDA GPU and matching NVRTC"]
fn nvrtc_long_warning_log_is_bounded_and_marked() {
    use invar_cuda::cuda::{CudaDevice, MAX_ERROR_BYTES, compile_cubin, compile_cubin_bounded};
    let device = CudaDevice::open().expect("a real CUDA GPU and matching NVRTC are required");
    // NVRTC emits one diagnostic per directive. This is well below the source
    // cap, but the warning log is much larger than the parent's stderr budget.
    let mut source = "#warning invar_long_warning_log_regression\n".repeat(160);
    source.push_str("extern \"C\" __global__ void invar_warning_kernel() {}\n");
    let direct = compile_cubin(&source, device.capabilities()).unwrap();
    let unbounded = format!("NVRTC_LOG={:?}\n", direct.log);
    assert!(
        unbounded.len() > MAX_ERROR_BYTES,
        "warning fixture must exceed the pipe budget"
    );
    let compiled = compile_cubin_bounded(
        &source,
        std::path::Path::new(env!("CARGO_BIN_EXE_invar-nvrtc")),
        device.capabilities(),
    )
    .expect("warnings must not turn a successful NVRTC compile into a child failure");
    assert!(!compiled.image.is_empty());
    assert!(compiled.log.len() < MAX_ERROR_BYTES, "{}", compiled.log);
    assert!(compiled.log.starts_with("NVRTC_LOG="), "{}", compiled.log);
    assert!(compiled.log.contains("invar_long_warning_log_regression"));
    assert!(
        compiled.log.ends_with("... [truncated]"),
        "{}",
        compiled.log
    );
    eprintln!(
        "NVRTC warning compile succeeded: cubin={} bytes, unbounded frame={} bytes, framed stderr={} bytes including newline; {}",
        compiled.image.len(),
        unbounded.len(),
        compiled.log.len() + 1,
        compiled.log,
    );
}
