//! Bounded NVRTC compiler subprocess. It accepts source on stdin and emits
//! CUBIN bytes on stdout; the parent owns its deadline and output cap.

use std::io::{Read, Write};

// Leave one byte for eprintln's newline; Debug escaping, marker and prefixes count
// against the parent's unchanged byte budget, not only the raw NVRTC log.
fn diagnostic_prefix(message: &str) -> String {
    const MARKER: &str = "... [truncated]";
    let limit = invar_cuda::cuda::MAX_ERROR_BYTES - 1;
    if message.len() <= limit {
        return message.to_owned();
    }
    let mut end = limit - MARKER.len();
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}{MARKER}", &message[..end])
}

fn main() {
    let caps = (|| {
        let mut args = std::env::args().skip(1);
        let major = args.next()?.parse().ok()?;
        let minor = args.next()?.parse().ok()?;
        let sm = args.next()?.parse().ok()?;
        let mode = args.next();
        match mode.as_deref() {
            None
            | Some(
                "--gnet-v5"
                | "strict-f32-v1"
                | "tc-f16-f32-v1"
                | "strict-f32-probe-v1"
                | "transpose-bits-v1",
            ) => {}
            _ => return None,
        }
        args.next()
            .is_none()
            .then_some((invar_device::Capabilities::Cuda { major, minor, sm }, mode))
    })();
    let Some((caps, mode)) = caps else {
        eprintln!("NVRTC_COMPILE_FAILED: expected CUDA major minor and sm");
        std::process::exit(1);
    };
    let mut input = Vec::new();
    let read = std::io::stdin()
        .take(
            (if mode.as_deref() == Some("--gnet-v5") {
                invar_device::MAX_GNET_SOURCE_BYTES + 1
            } else {
                invar_device::MAX_SOURCE_BYTES + 1
            }) as u64,
        )
        .read_to_end(&mut input);
    let result = read
        .map_err(|e| e.to_string())
        .and_then(|_| String::from_utf8(input).map_err(|e| e.to_string()))
        .and_then(|source| {
            let result = match mode.as_deref() {
                Some("--gnet-v5") => invar_cuda::cuda::compile_gnet_cubin(&source, caps),
                Some("strict-f32-v1") => invar_cuda::cuda::compile_floating_cubin(&source, caps),
                Some("tc-f16-f32-v1") => invar_cuda::cuda::compile_tc_cubin(&source, caps),
                Some("strict-f32-probe-v1") => {
                    invar_cuda::cuda::compile_floating_probe_cubin(&source, caps)
                }
                Some("transpose-bits-v1") => {
                    invar_cuda::cuda::compile_transpose_cubin(&source, caps)
                }
                _ => invar_cuda::cuda::compile_cubin(&source, caps),
            };
            result.map_err(|e| format!("{e:?}"))
        })
        .and_then(|compiled| {
            eprintln!(
                "{}",
                diagnostic_prefix(&format!("NVRTC_LOG={:?}", compiled.log))
            );
            std::io::stdout()
                .write_all(&compiled.image)
                .map_err(|e| e.to_string())
        });
    if let Err(error) = result {
        eprintln!(
            "{}",
            diagnostic_prefix(&format!("NVRTC_COMPILE_FAILED: {error}"))
        );
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn issue_m4_01_compiler_diagnostic_respects_the_parent_byte_budget() {
        for message in [
            String::new(),
            "short".into(),
            "x".repeat(invar_cuda::cuda::MAX_ERROR_BYTES - 1),
            "x".repeat(invar_cuda::cuda::MAX_ERROR_BYTES),
            format!("NVRTC_LOG={:?}", "warning\n".repeat(100)),
            format!("NVRTC_COMPILE_FAILED: {}", "λ".repeat(512)),
        ] {
            let prefix = super::diagnostic_prefix(&message);
            assert!(prefix.len() < invar_cuda::cuda::MAX_ERROR_BYTES);
            if message.len() < invar_cuda::cuda::MAX_ERROR_BYTES {
                assert_eq!(prefix, message);
            } else {
                let retained = prefix.strip_suffix("... [truncated]").unwrap();
                assert!(message.starts_with(retained));
                assert!(prefix.len() >= "... [truncated]".len());
            }
        }
    }
}
