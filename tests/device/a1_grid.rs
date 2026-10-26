//! Ordinary run-path evidence for grid adapters. CPU boundary tests live beside
//! the adapter; these tests exercise real Metal and CUDA runtimes when selected.

use a1_grid_common::{FUEL, fixtures, parity, program, sources};
use invar::{mir_eval, schedule};

#[path = "../support/a1_grid_common.rs"]
mod a1_grid_common;

#[test]
fn a1_grid_cpu_retains_serial_and_cli_forced_refusals() {
    for source in sources() {
        let program = program(&source);
        let cpu = schedule::run_source(source.as_bytes(), &program, FUEL, 2).unwrap();
        parity(&cpu.observation, &mir_eval::observe(&program, FUEL));
        assert!(cpu.executed.is_empty());
    }
    // Unsupported f32 stencils retain typed refusals on either build.
    let directory = std::env::temp_dir().join(format!("a1-grid-cli-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("grid.inv");
    std::fs::write(&path, fixtures::stencil(false, true, 3, 5, 1)).unwrap();
    for engine in ["metal", "cuda"] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
            .env("INVAR_METAL_RUNTIME_SOURCE", "1")
            .arg("run")
            .arg(&path)
            .args(["--engine", engine, "--format", "json"])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            json["diagnostics"][0]["code"],
            format!("{}_UNAVAILABLE", engine.to_uppercase()),
            "{json}"
        );
        println!("forced-{engine}-f32-stencil {json}");
    }
    std::fs::remove_dir_all(directory).unwrap();
}
