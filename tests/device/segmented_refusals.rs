//! CLI refusals for forced engines on a program with no device candidate; device runs are in `tests/gpu_{cuda,metal}/segmented_publication.rs`.

#[path = "../support/a1_segmented_common.rs"]
mod a1_segmented_common;

#[cfg(not(feature = "cuda"))]
#[test]
fn forced_engines_refuse_a_program_without_a_candidate() {
    use a1_segmented_common::cli;
    let dir = std::env::temp_dir().join(format!("invar-a1-segmented-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("no-region.inv");
    std::fs::write(&path, "fn main() -> i32 { return 0; }").unwrap();
    for (engine, code) in [("metal", "METAL_UNAVAILABLE"), ("cuda", "CUDA_UNAVAILABLE")] {
        let result = cli(engine, &path);
        assert!(result.to_string().contains(code), "{result}");
    }
    let auto = cli("auto", &path);
    assert_eq!(auto["status"], "ok", "{auto}");
    std::fs::remove_dir_all(dir).unwrap();
}
