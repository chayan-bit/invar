#![forbid(unsafe_code)]
// The currently unexported M4-03 source contract uses crate-root modules.
pub use invar::{
    check, front, general_net, hex, layout_contract, mir, numeric_contract, tensor_cache,
};
#[path = "../../qualification/mod.rs"]
mod qualification;
use serde_json::json;

fn main() {
    let result = (|| {
        let mut args = std::env::args().skip(1);
        let mut path = None;
        let mut gpu = false;
        let mut device = None;
        let mut nvrtc = None;
        let mut libraries = None;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--device" if device.is_none() => {
                    device = Some(match args.next().as_deref() {
                        Some("cuda") => qualification::device::Target::Cuda,
                        Some("metal") => qualification::device::Target::Metal,
                        _ => {
                            return Err(qualification::Failure {
                                code: "M5_USAGE",
                                detail: "--device requires cuda or metal".into(),
                            });
                        }
                    });
                }
                "--nvrtc" if nvrtc.is_none() => {
                    nvrtc = Some(std::path::PathBuf::from(args.next().ok_or_else(|| {
                        qualification::Failure {
                            code: "M5_USAGE",
                            detail: "missing NVRTC path".into(),
                        }
                    })?))
                }
                "--libraries" if libraries.is_none() => {
                    libraries = Some(std::path::PathBuf::from(args.next().ok_or_else(|| {
                        qualification::Failure {
                            code: "M5_USAGE",
                            detail: "missing library directory".into(),
                        }
                    })?))
                }
                "--require-gpu" if !gpu => gpu = true,
                "--report" if path.is_none() => {
                    path = Some(std::path::PathBuf::from(args.next().ok_or_else(|| {
                        qualification::Failure {
                            code: "M5_USAGE",
                            detail: "missing report path".into(),
                        }
                    })?))
                }
                _ => {
                    return Err(qualification::Failure {
                        code: "M5_USAGE",
                        detail: "usage: invar-qualify [--device cuda|metal] [--nvrtc ABSOLUTE_PATH] [--libraries ABSOLUTE_DIRECTORY] [--require-gpu] [--report NEW_PATH]".into(),
                    });
                }
            }
        }
        if (device.is_none() && (nvrtc.is_some() || libraries.is_some()))
            || (device == Some(qualification::device::Target::Metal) && nvrtc.is_some())
        {
            return Err(qualification::Failure {
                code: "M5_USAGE",
                detail: "--nvrtc requires --device cuda; --libraries requires --device".into(),
            });
        }
        if libraries
            .as_ref()
            .is_some_and(|p| !p.is_absolute() || !p.is_dir())
        {
            return Err(qualification::Failure {
                code: "M5_LIBRARY_PATH",
                detail: "absolute existing library directory required".into(),
            });
        }
        let mut report = qualification::run(qualification::corpus::SEED)?;
        report["boundary_fuzz"] = qualification::fuzz_boundary(qualification::corpus::SEED)?;
        report["mutations"] = qualification::mutations()?;
        if let Some(target) = device {
            qualification::device::attach(
                &mut report,
                target,
                nvrtc.as_deref(),
                libraries.as_deref(),
            )?;
        }
        report["checkout_at_invocation"] = json!(
            std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        );
        if gpu {
            qualification::require_gpu(&report)?;
        }
        if let Some(path) = path {
            qualification::publish(&path, &report)?;
        }
        println!("{report}");
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!(
            "{}",
            json!({"status":"refused","diagnostic":error.code,"detail":error.detail})
        );
        std::process::exit(2);
    }
}
