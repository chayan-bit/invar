//! Fresh-process backend archive qualification. Source compilation is armed
//! to abort for AOT tests, and FailOnBinaryArchiveMiss proves physical hits.
use invar::compiled_ahead::{self, Route};
use invar_metal::MetalDevice;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};
#[path = "../support/scratch.rs"]
mod scratch;
const MSL: &str = "#include <metal_stdlib>\nusing namespace metal;\nkernel void p5b_archive(device uint* out [[buffer(0)]],uint i [[thread_position_in_grid]]) {out[i]=i*3u+7u;}\n";
const OTHER_MSL: &str = "#include <metal_stdlib>\nusing namespace metal;\nkernel void p5b_archive(device uint* out [[buffer(0)]],uint i [[thread_position_in_grid]]) {out[i]=i*5u+11u;}\n";

fn scratch(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("p5b-{label}-{}", scratch::unique()));
    let mut builder = fs::DirBuilder::new();
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder.create(&path).unwrap();
    path
}
fn child(cache: &Path, image: Option<&Path>, msl: Option<&Path>, label: &str) -> Command {
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args([
        "--ignored",
        "--exact",
        "backend_archives::archive_child",
        "--nocapture",
    ])
    .env("INVAR_METAL_ARCHIVE_DIR", cache)
    .env("INVAR_P5B_CHILD", label)
    .env_remove("INVAR_P5B_OTHER")
    .env("INVAR_P5B_LOG", cache.parent().unwrap());
    if let Some(image) = image {
        cmd.env("INVAR_P5B_IMAGE", image)
            .env(invar_metal::SOURCE_TRIPWIRE, "1");
    }
    if let Some(msl) = msl {
        cmd.env("INVAR_P5B_MSL", msl)
            .env_remove(invar_metal::SOURCE_TRIPWIRE);
    }
    cmd
}
fn collect(cmd: Command) -> Value {
    collect_bounded(cmd, 180).unwrap_or_else(|error| panic!("Metal child failed: {error}"))
}
fn collect_bounded(mut cmd: Command, seconds: u64) -> Result<Value, Value> {
    let root = PathBuf::from(
        cmd.get_envs()
            .find(|(k, _)| *k == "INVAR_P5B_LOG")
            .unwrap()
            .1
            .unwrap(),
    );
    let log = root.join(format!("child-{}.log", scratch::unique()));
    let file = fs::File::create(&log).unwrap();
    cmd.stdout(file.try_clone().unwrap())
        .stderr(Stdio::from(file));
    let mut child = cmd.spawn().unwrap();
    let begun = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if begun.elapsed() > Duration::from_secs(seconds) {
            child.kill().unwrap();
            let _ = child.wait();
            return Err(json!({"refusal":"timeout","limit_seconds":seconds}));
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let text = fs::read_to_string(&log).unwrap();
    if !status.success() {
        return Err(json!({"refusal":"child_exit","exit":status.code(),"diagnostic":text}));
    }
    let line = text
        .lines()
        .find_map(|line| line.strip_prefix("ARCHIVE "))
        .ok_or_else(|| json!({"refusal":"missing_receipt"}))?;
    let row: Value =
        serde_json::from_str(line).map_err(|_| json!({"refusal":"invalid_receipt"}))?;
    if cmd.get_envs().any(|(key, _)| key == "INVAR_P5B_IMAGE") {
        let label = cmd
            .get_envs()
            .find(|(key, _)| *key == "INVAR_P5B_CHILD")
            .unwrap()
            .1
            .unwrap()
            .to_string_lossy();
        println!("P5B_ARCHIVE {}", json!({"label":label,"receipt":row}));
    }
    Ok(row)
}
fn build_image(root: &Path, source: &str) -> PathBuf {
    let bytes = compiled_ahead::build::build(
        Route::Run,
        "main",
        b"p5b archive probe",
        &[(source.to_owned(), vec!["p5b_archive".to_owned()])],
    )
    .unwrap();
    let file = root.join("probe.invmetal");
    compiled_ahead::build::write_atomic(&file, &bytes).unwrap();
    file
}
fn cache_file(root: &Path) -> PathBuf {
    let files: Vec<_> = fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|e| e == "invgmba"))
        .collect();
    assert_eq!(files.len(), 1);
    files[0].clone()
}
#[test]
#[ignore = "requires an Apple GPU and offline Metal toolchain"]
fn cold_process_hit_corruption_rebuild_and_concurrent_publication() {
    let root = scratch("archive");
    let image = build_image(&root, MSL);
    let cache = root.join("cache");
    let first = collect(child(&cache, Some(&image), None, "cold-miss"));
    assert_eq!(first["state"], "Miss");
    assert_eq!(first["writes"], 1);
    assert_eq!(first["write_refusal"], Value::Null);
    let hit = collect(child(&cache, Some(&image), None, "cold-hit"));
    assert_eq!(hit["state"], "Hit");
    assert_eq!(hit["source_compiles"], 0);
    assert_eq!(hit["writes"], 0);
    assert_eq!(hit["key"], first["key"]);
    let mut changed = child(&cache, Some(&image), None, "compiler-identity");
    changed.env("INVAR_P5B_COMPILER_SUFFIX", "changed-compiler");
    let changed = collect(changed);
    assert_eq!(changed["state"], "Miss");
    assert_ne!(changed["key"], hit["key"]);
    let changed_key = changed["key"].as_str().unwrap();
    fs::remove_file(cache.join(format!("{changed_key}.invgmba"))).unwrap();
    let file = cache_file(&cache);
    let mut bytes = fs::read(&file).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    compiled_ahead::build::write_atomic(&file, &bytes).unwrap();
    let rebuilt = collect(child(&cache, Some(&image), None, "corrupt"));
    assert_eq!(rebuilt["state"], "Rebuilt(Digest)");
    assert_eq!(rebuilt["writes"], 1);
    let repaired = collect(child(&cache, Some(&image), None, "repaired"));
    assert_eq!(repaired["state"], "Hit");
    // A valid outer checksum never makes an invalid opaque Metal payload valid.
    use sha2::{Digest, Sha256};
    let mut forged = fs::read(&file).unwrap();
    forged.truncate(72);
    let garbage = [0u8; 64];
    forged[40..72].copy_from_slice(&Sha256::digest(garbage));
    forged.extend_from_slice(&garbage);
    compiled_ahead::build::write_atomic(&file, &forged).unwrap();
    let refused = collect(child(&cache, Some(&image), None, "invalid-metal"));
    assert_eq!(refused["state"], "Rebuilt(Metal)");
    assert_eq!(refused["writes"], 1);

    // A valid archive for different code under the same kernel symbol must
    // not be called a hit.
    // Without FailOnBinaryArchiveMiss, Metal can silently compile the missing
    // pipeline from the admitted library and make a false hit report.
    let unrelated_root = root.join("other-image");
    fs::create_dir(&unrelated_root).unwrap();
    let unrelated_image = build_image(&unrelated_root, OTHER_MSL);
    let unrelated_cache = root.join("unrelated-cache");
    let mut unrelated = child(&unrelated_cache, Some(&unrelated_image), None, "unrelated");
    unrelated.env("INVAR_P5B_OTHER", "1");
    assert_eq!(collect(unrelated)["state"], "Miss");
    let other = fs::read(cache_file(&unrelated_cache)).unwrap();
    let payload = &other[72..];
    let mut mismatched = fs::read(&file).unwrap();
    mismatched.truncate(72);
    mismatched[40..72].copy_from_slice(&Sha256::digest(payload));
    mismatched.extend_from_slice(payload);
    compiled_ahead::build::write_atomic(&file, &mismatched).unwrap();
    let refused = collect(child(&cache, Some(&image), None, "pipeline-miss"));
    assert_eq!(refused["state"], "Rebuilt(PipelineMiss)");
    assert_eq!(refused["writes"], 1);

    let other = root.join("external");
    fs::write(&other, b"do not overwrite").unwrap();
    fs::remove_file(&file).unwrap();
    std::os::unix::fs::symlink(&other, &file).unwrap();
    let symlink = collect(child(&cache, Some(&image), None, "symlink"));
    assert_eq!(symlink["state"], "Rebuilt(Permissions)");
    assert_eq!(fs::read(&other).unwrap(), b"do not overwrite");
    fs::remove_file(&file).unwrap();
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let cmd = child(&cache, Some(&image), None, &format!("race-{i}"));
            std::thread::spawn(move || collect(cmd))
        })
        .collect();
    for handle in handles {
        let row = handle.join().unwrap();
        assert_eq!(row["write_refusal"], Value::Null);
        assert!(matches!(row["state"].as_str(), Some("Hit" | "Miss")));
    }
    assert_eq!(
        collect(child(&cache, Some(&image), None, "race-hit"))["state"],
        "Hit"
    );
    println!(
        "P5B archive: cold-process hit, digest refusal/rebuild, symlink refusal, 4-writer race passed"
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "child entry, invoked only with a private cache"]
fn archive_child() {
    assert!(std::env::var_os("INVAR_P5B_CHILD").is_some());
    let (mut device, source, kernel) = if let Some(path) = std::env::var_os("INVAR_P5B_MSL") {
        (
            MetalDevice::open_runtime_source().unwrap(),
            fs::read_to_string(path).unwrap(),
            "invar_gnet_children_v5",
        )
    } else {
        let image =
            compiled_ahead::build::read(Path::new(&std::env::var_os("INVAR_P5B_IMAGE").unwrap()))
                .unwrap();
        let mut device = MetalDevice::open().unwrap();
        if let Ok(suffix) = std::env::var("INVAR_P5B_COMPILER_SUFFIX") {
            let mut header = image.header().clone();
            header.toolchain.metal.push_str(&suffix);
            let libraries: Vec<_> = image.libraries().map(|(_, bytes)| bytes.to_vec()).collect();
            let bytes = compiled_ahead::encode(&header, &libraries).unwrap();
            compiled_ahead::decode(&bytes)
                .unwrap()
                .install(&mut device)
                .unwrap();
        } else {
            image.install(&mut device).unwrap();
        }
        let source = if std::env::var_os("INVAR_P5B_OTHER").is_some() {
            OTHER_MSL
        } else {
            MSL
        };
        (device, source.to_owned(), "p5b_archive")
    };
    let m = device.measure_pipeline(&source, kernel).unwrap();
    let report = device.cache_report();
    println!(
        "\nARCHIVE {}",
        json!({"state":format!("{:?}",m.archive.last),"key":m.archive.key.iter().map(|b|format!("{b:02x}")).collect::<String>(),"hits":m.archive.hits,"misses":m.archive.misses,"rebuilds":m.archive.rebuilds,"writes":m.archive.writes,"write_refusal":m.archive.write_refusal.map(|r|format!("{r:?}")),"archive_bytes":m.archive.bytes,"library_ms":m.library_ms,"pipeline_ms":m.pipeline_ms,"pipeline_creation_ms":m.pipeline_creation_ms,"total_ms":m.total_ms,"source_bytes":source.len(),"max_threads":m.max_threads,"execution_width":m.execution_width,"static_threadgroup_bytes":m.static_threadgroup_bytes,"source_compiles":report.source_compiles})
    );
}

#[test]
#[ignore = "explicit before/after MSL snapshot directories required"]
fn measure_before_and_after_pipelines() {
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    let root = PathBuf::from(std::env::var_os("INVAR_P5B_MEASURE").unwrap());
    let mut prior = BTreeMap::new();
    if let Some(path) = std::env::var_os("INVAR_P5B_RESUME") {
        for line in fs::read_to_string(path).unwrap().lines() {
            let row: Value = serde_json::from_str(line).unwrap();
            prior.insert(
                (
                    row["id"].as_str().unwrap().to_owned(),
                    row["phase"].as_str().unwrap().to_owned(),
                ),
                row,
            );
        }
    }
    let mut files: Vec<_> = fs::read_dir(root.join("before"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "msl"))
        .collect();
    files.sort();
    let mut measured = 0;
    for before in &files {
        let id = before.file_stem().unwrap().to_str().unwrap();
        for phase in ["before", "after"] {
            let file = root.join(phase).join(format!("{id}.msl"));
            let hash: String = Sha256::digest(fs::read(&file).unwrap())
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            if let Some(row) = prior.get(&(id.to_owned(), phase.to_owned())) {
                assert_eq!(row["source_sha256"], hash, "resume lineage");
                println!("P5B_PIPELINE {row}");
                measured += 1;
                continue;
            }
            let cache = root.join(format!("cache-{phase}-{id}"));
            // A failed 180s baseline probe demonstrated the need for a larger
            // bounded compile-only window. Correctness children stay at 180s.
            let result = collect_bounded(child(&cache, None, Some(&file), id), 600);
            let row = match &result {
                Ok(value) | Err(value) => value,
            };
            println!(
                "P5B_PIPELINE {}",
                json!({"phase":phase,"id":id,"source_sha256":hash,"measurement":row})
            );
            let _ = fs::remove_dir_all(cache);
            assert!(
                result.is_ok(),
                "compile refusal retained above; stop before dispatching more work"
            );
            measured += 1;
        }
    }
    assert!(!files.is_empty());
    assert_eq!(measured, files.len() * 2);
}

#[test]
#[ignore = "physical Metal pipeline cache and offline compiler"]
fn memory_hit_reports_the_reused_identity_and_keeps_cumulative_counts() {
    let mut device = MetalDevice::open_runtime_source().unwrap();
    let a = device.measure_pipeline(MSL, "p5b_archive").unwrap();
    let changed = MSL.replace("3u+7u", "5u+11u");
    let b = device.measure_pipeline(&changed, "p5b_archive").unwrap();
    assert_ne!(a.archive.key, b.archive.key);
    let reused = device.measure_pipeline(MSL, "p5b_archive").unwrap();
    assert_eq!(reused.archive.last, invar_metal::ArchiveState::MemoryHit);
    assert_eq!(reused.archive.key, a.archive.key);
    assert_eq!(reused.archive.bytes, a.archive.bytes);
    assert_eq!(reused.archive.hits, b.archive.hits);
    assert_eq!(reused.archive.misses, b.archive.misses);

    // Registering AOT after a developer-source compile changes the compiler
    // identity even when the MSL and symbol are identical.
    let root = scratch("memory-identity");
    let image = build_image(&root, MSL);
    let before = device.cache_report();
    compiled_ahead::build::read(&image)
        .unwrap()
        .install(&mut device)
        .unwrap();
    let installed = device.measure_pipeline(MSL, "p5b_archive").unwrap();
    assert_eq!(
        device.cache_report().source_compiles,
        before.source_compiles
    );
    assert_eq!(
        device.cache_report().pipeline_creations,
        before.pipeline_creations + 1
    );
    assert_ne!(installed.archive.key, a.archive.key);
    assert_ne!(installed.archive.last, invar_metal::ArchiveState::MemoryHit);
    let other = device.measure_pipeline(&changed, "p5b_archive").unwrap();
    assert_eq!(other.archive.last, invar_metal::ArchiveState::MemoryHit);
    assert_eq!(other.archive.key, b.archive.key);
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "physical Metal production CLI and offline compiler"]
fn gnet_cli_reports_cold_archive_miss_then_hit_without_runtime_source() {
    for (source_text, expected, libraries) in [
        (
            "fn run(x:i32)->i32{return x+7;} fn main()->i32{return 0;}",
            json!(12),
            1,
        ),
        (
            "fn run(x:u32)->f32{return f32_from_u32(x)+0.5;} fn main()->i32{return 0;}",
            json!({"bits":0x40b00000u32}),
            2,
        ),
        (
            "fn run(x:u32)->Array<f32>{return [-0.0,f32_from_u32(x)+0.5];} fn main()->i32{return 0;}",
            json!([{"bits":0x80000000u32},{"bits":0x40b00000u32}]),
            2,
        ),
    ] {
        let root = scratch("gnet-cli");
        let source = root.join("program.inv");
        let artifact = root.join("program.invmetal");
        let cache = root.join("cache");
        fs::write(&source, source_text).unwrap();
        let built = Command::new(env!("CARGO_BIN_EXE_invar"))
            .arg("build")
            .arg(&source)
            .args(["--target", "metal", "--entry", "run", "--output"])
            .arg(&artifact)
            .output()
            .unwrap();
        assert!(
            built.status.success(),
            "{} {}",
            String::from_utf8_lossy(&built.stdout),
            String::from_utf8_lossy(&built.stderr)
        );
        for state in ["Miss", "Hit"] {
            let result=Command::new(env!("CARGO_BIN_EXE_invar")).arg("gnet-device").arg(&source)
                .args(["run","metal","[5]",r#"{"fuel":10000,"transitions":10000,"trace_events":10000,"quantum":64,"rounds":64,"capacity":0}"#,"--artifact"])
                .arg(&artifact).env("INVAR_METAL_ARCHIVE_DIR",&cache).env(invar_metal::SOURCE_TRIPWIRE,"1").output().unwrap();
            assert!(
                result.status.success(),
                "{} {}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            let row: Value = serde_json::from_slice(&result.stdout).unwrap();
            assert_eq!(row["status"], 0);
            assert_eq!(row["result"], expected);
            assert_eq!(row["metal_archive"]["state"], state);
            assert_eq!(row["image"]["runtime_compiler"], "none");
            assert_eq!(
                row["image"]["libraries"].as_array().unwrap().len(),
                libraries
            );
            if libraries == 2 {
                assert!(!row["numerics"].is_null(), "{row}");
            }
            println!(
                "P5B_GNET_CLI {}",
                json!({"result":row["result"],"archive":row["metal_archive"],"runtime_compiler":row["image"]["runtime_compiler"],"libraries":libraries,"numerics":row["numerics"]})
            );
        }
        fs::remove_dir_all(root).unwrap();
    }
}
