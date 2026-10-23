//! Runner for `baselines/metal-families.mm` and `baselines/cuda-families.cu`
//! (protocol schema 3): Apple MPSGraph or NVIDIA CUB/Thrust library calls
//! where a library primitive exists, otherwise reviewed handwritten kernels.
//! The fixture is an untrusted child process: its run is bounded, its report
//! is validated field by field and its full output is compared with the
//! independent Rust references.

use serde_json::{Value, json};
use std::{
    fs,
    path::Path,
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use super::{family_references as references, hash::sha256};

/// `*_lcg` families use this protocol's input recipe; the names stay distinct
/// from the v1 `map`/`sum`/`scan`/`affine` fixtures, whose hashes v1 pins.
pub const FAMILIES: [&str; 11] = [
    "transpose",
    "gather",
    "gather_hw",
    "stencil",
    "segsum",
    "compact",
    "frontier",
    "map_lcg",
    "sum_lcg",
    "scan_lcg",
    "affine_lcg",
];
const TIMEOUT: Duration = Duration::from_secs(60);
// macOS SystemTime has only microsecond resolution, so concurrent runs in one
// process also need a counter for distinct scratch roots.
static NEXT_ROOT: AtomicU64 = AtomicU64::new(0);
/// Every fixture family sweeps to 2^28 elements (frontier: adjacency cells).
const MAX_ELEMENTS: usize = 1 << 28;
/// At most one u32 word per element plus a small header.
const MAX_OUTPUT: u64 = 4 * MAX_ELEMENTS as u64 + 4096;
const MAX_REPORT: u64 = 64 * 1024;

fn read_limit(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let meta = fs::metadata(path).map_err(|e| e.to_string())?;
    if !meta.is_file() || meta.len() > limit {
        return Err("baseline artifact size exceeded".into());
    }
    fs::read(path).map_err(|e| e.to_string())
}

/// Which fixture protocol a run expects.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Metal,
    Cuda,
}

impl Backend {
    /// The GPU fixture this host runs: Metal on macOS, CUDA elsewhere.
    pub fn host() -> Self {
        if cfg!(target_os = "macos") {
            Backend::Metal
        } else {
            Backend::Cuda
        }
    }
}

/// The baseline a fixture family names on a backend; CUDA has no separate
/// handwritten gather because Thrust gather is the contract's library.
fn baseline_name(family: &str, backend: Backend) -> Option<&'static str> {
    Some(match (backend, family) {
        (Backend::Metal, "transpose") => "Apple MPSGraph transpose",
        (Backend::Metal, "gather") => "Apple MPSGraph gather",
        (Backend::Metal, "gather_hw") => "Handwritten Metal gather",
        (Backend::Metal, "stencil") => "Handwritten Metal stencil",
        (Backend::Metal, "segsum") => "Handwritten Metal segmented sum",
        (Backend::Metal, "compact") => "Handwritten Metal stable compaction",
        (Backend::Metal, "frontier") => "Handwritten Metal level-synchronous frontier",
        (Backend::Cuda, "transpose") => "NVIDIA reference tiled transpose (port)",
        (Backend::Cuda, "gather") => "NVIDIA Thrust gather",
        (Backend::Cuda, "stencil") => "Handwritten CUDA stencil",
        (Backend::Cuda, "segsum") => "NVIDIA CUB DeviceSegmentedReduce",
        (Backend::Cuda, "compact") => "NVIDIA CUB DeviceSelect",
        (Backend::Cuda, "frontier") => "Handwritten CUDA level-synchronous frontier",
        (Backend::Cuda, "map_lcg") => "NVIDIA CUB DeviceTransform",
        (Backend::Cuda, "sum_lcg") => "NVIDIA CUB DeviceReduce",
        (Backend::Cuda, "scan_lcg") => "NVIDIA CUB DeviceScan",
        (Backend::Cuda, "affine_lcg") => "NVIDIA CUB DeviceScan affine compose",
        _ => return None,
    })
}

/// A vendor library call is an independent library baseline; a ported vendor
/// sample or our own kernel is a reviewed handwritten baseline pinned by hash.
fn baseline_kind(name: &str) -> &'static str {
    if (name.starts_with("Apple") || name.starts_with("NVIDIA")) && !name.ends_with("(port)") {
        "independent_library_baseline"
    } else {
        "reviewed_handwritten_baseline"
    }
}

/// The manifest family a fixture family measures.
pub fn manifest_family(family: &str) -> &'static str {
    match family {
        "transpose" => "transpose",
        "gather" | "gather_hw" => "gather_scatter",
        "stencil" => "stencil",
        "segsum" => "segmented",
        "compact" => "compaction",
        "map_lcg" => "map",
        "sum_lcg" => "reduction",
        "scan_lcg" | "affine_lcg" => "scan",
        _ => "frontier",
    }
}

fn expected(family: &str, n: usize, backend: Backend) -> Result<(Vec<u32>, &'static str), String> {
    let name = baseline_name(family, backend).ok_or("no such fixture family on this backend")?;
    let input = || references::inputs(n, 1).remove(0);
    let (rows, cols) = references::grid(n);
    Ok((
        match family {
            "transpose" => references::ref_transpose(&input(), rows, cols),
            "gather" | "gather_hw" => {
                let values = input();
                references::gather_indices(n)
                    .iter()
                    .map(|&i| values[i as usize])
                    .collect()
            }
            "stencil" => references::ref_stencil_u32(&input(), rows, cols),
            "segsum" => {
                if !n.is_multiple_of(16) {
                    return Err("segsum needs a multiple of 16 elements".into());
                }
                let offsets: Vec<u32> = (0..=n / 16).map(|s| (s * 16) as u32).collect();
                references::ref_segment_sums(&input(), &offsets)
            }
            "compact" => references::ref_compact(&input()),
            "map_lcg" | "sum_lcg" | "scan_lcg" | "affine_lcg" => lcg_family(family, n),
            _ => {
                let v = n.isqrt();
                if v * v != n || v > 1 << 14 {
                    return Err(
                        "frontier needs a square adjacency cell count of at most 2^28".into(),
                    );
                }
                references::ref_frontier(
                    &references::frontier_edges(v),
                    v,
                    references::FRONTIER_ROUNDS,
                )
                .0
            }
        },
        name,
    ))
}

/// References for the map, reduction and scan families over input slots 0 and 1.
fn lcg_family(family: &str, n: usize) -> Vec<u32> {
    let slots = references::inputs(n, 2);
    let (x0, x1) = (&slots[0], &slots[1]);
    match family {
        "map_lcg" => x0
            .iter()
            .zip(x1)
            .map(|(a, b)| a.wrapping_add(*b).wrapping_mul(7).wrapping_add(11))
            .collect(),
        "sum_lcg" => vec![x0.iter().fold(0u32, |s, x| s.wrapping_add(*x))],
        "scan_lcg" => x0
            .iter()
            .scan(0u32, |s, x| {
                *s = s.wrapping_add(*x);
                Some(*s)
            })
            .collect(),
        _ => x0
            .iter()
            .zip(x1)
            .scan(17u32, |y, (a, b)| {
                *y = a.wrapping_mul(*y).wrapping_add(*b);
                Some(*y)
            })
            .collect(),
    }
}

fn samples(document: &Value, key: &str) -> Result<Vec<f64>, String> {
    let raw = document[key].as_array().ok_or("missing raw samples")?;
    // Every entry must be a number: nothing is filtered out before counting.
    let values: Vec<f64> = raw
        .iter()
        .map(Value::as_f64)
        .collect::<Option<_>>()
        .ok_or("nonnumeric raw sample")?;
    if values.len() != 30
        || values
            .iter()
            .any(|x| !x.is_finite() || !(0.0..=30_000.0).contains(x))
    {
        return Err("invalid raw samples".into());
    }
    Ok(values)
}

pub fn run(executable: &Path, family: &str, n: usize, backend: Backend) -> Result<Value, String> {
    if !FAMILIES.contains(&family) || !(1..=MAX_ELEMENTS).contains(&n) {
        return Err("invalid family baseline or element count".into());
    }
    let (want, name) = expected(family, n, backend)?;
    let executable_bytes = read_limit(executable, 64 * 1024 * 1024)?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "invar-bench-fam-{}-{nonce}-{}",
        std::process::id(),
        NEXT_ROOT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&root).map_err(|e| e.to_string())?;
    let result = (|| {
        let (output, stdout) = (root.join("output.bin"), root.join("stdout.json"));
        let mut child = Command::new(executable)
            .arg(family)
            .arg(n.to_string())
            .arg(&output)
            .stdin(Stdio::null())
            .stdout(Stdio::from(
                fs::File::create(&stdout).map_err(|e| e.to_string())?,
            ))
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| e.to_string())?;
        let started = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
                break status;
            }
            if started.elapsed() >= TIMEOUT {
                let _ = child.kill();
                let _ = child.wait();
                return Err("baseline process timed out".into());
            }
            thread::sleep(Duration::from_millis(10));
        };
        if !status.success() {
            return Err(format!("baseline exited {status}"));
        }
        let document: Value =
            serde_json::from_slice(&read_limit(&stdout, MAX_REPORT)?).map_err(|e| e.to_string())?;
        let versions = &document["vendor_versions"];
        let positive = |key: &str| versions[key].as_u64().is_some_and(|v| v > 0);
        let vendor_ok = match backend {
            Backend::Metal => {
                let framework = if name.starts_with("Apple MPSGraph") {
                    "MetalPerformanceShadersGraph system framework"
                } else {
                    "Metal system framework"
                };
                versions["framework"] == framework
                    && versions["macos"].as_array().is_some_and(|v| v.len() == 3)
            }
            Backend::Cuda => {
                versions["framework"] == "CUDA runtime with CCCL"
                    && [
                        "cuda_runtime",
                        "cuda_headers",
                        "cuda_driver",
                        "cub",
                        "thrust",
                    ]
                    .iter()
                    .all(|k| positive(k))
                    && versions["device"].as_str().is_some_and(|d| !d.is_empty())
                    && versions["compute_capability"]
                        .as_str()
                        .is_some_and(|c| c.starts_with("sm_"))
            }
        };
        // Accounting and boundaries are published, so they are protocol fields too.
        let input_words = match family {
            "gather" | "gather_hw" => 2 * n,
            "map_lcg" | "affine_lcg" => 2 * n,
            _ => n,
        };
        let boundaries = match backend {
            Backend::Metal => (
                "sum of metal command buffer gpu start to end",
                "host input copy through host output copy, after compile",
            ),
            Backend::Cuda => (
                "sum of cuda event intervals around the library calls or kernels",
                "pageable host input copy through host output copy, after workspace sizing",
            ),
        };
        if document["input_bytes"].as_u64() != Some(4 * input_words as u64)
            || document["cold_process_ms"]
                .as_f64()
                .is_none_or(|x| !x.is_finite() || x <= 0.0)
            || document["device_resident_boundary"] != boundaries.0
            || document["host_to_host_boundary"] != boundaries.1
        {
            return Err("family baseline accounting or boundary mismatch".into());
        }
        if document["schema_version"] != 3
            || document["baseline"] != name
            || document["family"] != family
            || document["n"] != n
            || document["warmups"] != 10
            || document["samples"] != 30
            || document["input_recipe"] != "lcg1664525_1013904223_v1"
            || !vendor_ok
            || document["compilation_ms"]
                .as_f64()
                .is_none_or(|x| !x.is_finite() || x <= 0.0)
        {
            return Err("family baseline protocol mismatch".into());
        }
        let device = samples(&document, "device_samples_ms")?;
        let host = samples(&document, "host_samples_ms")?;
        let words = read_limit(&output, MAX_OUTPUT)?;
        let got: Vec<u32> = words
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        if words.len() % 4 != 0 || document["output_bytes"] != words.len() {
            return Err("family baseline output accounting mismatch".into());
        }
        let correct = got == want;
        let summary = |s: &[f64]| {
            let mut s = s.to_vec();
            s.sort_by(f64::total_cmp);
            json!({"median_ms":(s[14]+s[15])/2.0,"median_ci95_ms":[s[9],s[20]],"q1_ms":s[7],"q3_ms":s[22]})
        };
        Ok(json!({
            "schema_version":3,"kind":baseline_kind(name),
            "family":manifest_family(family),"fixture_family":family,"elements":n,
            "status":if correct {"measured_correct"} else {"failed"},
            "full_output_correctness":if correct {"exact_full_output_pass"} else {"exact_full_output_fail"},
            "baseline_name":name,"vendor_versions":document["vendor_versions"],
            "fixture_source_sha256":match backend {
                Backend::Metal => sha256(include_bytes!("baselines/metal-families.mm")),
                Backend::Cuda => sha256(include_bytes!("baselines/cuda-families.cu")),
            },
            "executable_sha256":sha256(&executable_bytes),"output_sha256":sha256(&words),
            "input_recipe":document["input_recipe"],"input_bytes":document["input_bytes"],
            "output_bytes":words.len(),"compilation_ms":document["compilation_ms"],
            "cold_process_ms":document["cold_process_ms"],
            "device_resident_boundary":document["device_resident_boundary"],
            "host_to_host_boundary":document["host_to_host_boundary"],
            "device_resident_samples_ms":device,"host_to_host_samples_ms":host,
            "device_resident_summary":summary(&device),"host_to_host_summary":summary(&host),
            "warmups":10,"samples":30,"timing_source":"vendor fixture",
            "gpu_performance":"baseline_only; no generated ratio"
        }))
    })();
    let _ = fs::remove_dir_all(&root);
    result
}

// Fake fixtures are executed while other test threads fork; the shared writer never
// leaves a writable descriptor for a sibling fork to inherit (ETXTBSY).
#[cfg(all(test, unix))]
#[path = "../../tests/support/executable.rs"]
pub(crate) mod executable;

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// A stand-in fixture: copies prepared output words and prints a prepared
    /// protocol document, so the runner's validation is tested without a GPU.
    fn fake(dir: &Path, words: &[u32], document: &Value, exit: i32) -> std::path::PathBuf {
        fake_padded(dir, words, document, exit, 0)
    }

    /// `padding` bytes of shell comment widen the write window, as in the writer's ETXTBSY stress test.
    fn fake_padded(
        dir: &Path,
        words: &[u32],
        document: &Value,
        exit: i32,
        padding: usize,
    ) -> std::path::PathBuf {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        fs::write(dir.join("words.bin"), bytes).unwrap();
        fs::write(dir.join("doc.json"), document.to_string()).unwrap();
        let script = dir.join("fixture.sh");
        super::executable::write(
            &script,
            format!(
                "#!/bin/sh\ncp '{}' \"$3\"\ncat '{}'\nexit {exit}\n#{}\n",
                dir.join("words.bin").display(),
                dir.join("doc.json").display(),
                "x".repeat(padding)
            )
            .as_bytes(),
        )
        .unwrap();
        script
    }

    fn document(family: &str, n: usize, backend: Backend, output_bytes: usize) -> Value {
        let versions = match backend {
            Backend::Metal => {
                let mpsgraph = baseline_name(family, backend)
                    .unwrap()
                    .starts_with("Apple MPSGraph");
                json!({"framework":if mpsgraph {"MetalPerformanceShadersGraph system framework"} else {"Metal system framework"},"macos":[15,0,0]})
            }
            Backend::Cuda => {
                json!({"framework":"CUDA runtime with CCCL","cuda_runtime":12090,"cuda_headers":12090,
                "cuda_driver":12090,"cub":200802,"thrust":200802,"device":"synthetic","compute_capability":"sm_120"})
            }
        };
        let words = match family {
            "gather" | "gather_hw" | "map_lcg" | "affine_lcg" => 2 * n,
            _ => n,
        };
        let (device, host) = match backend {
            Backend::Metal => (
                "sum of metal command buffer gpu start to end",
                "host input copy through host output copy, after compile",
            ),
            Backend::Cuda => (
                "sum of cuda event intervals around the library calls or kernels",
                "pageable host input copy through host output copy, after workspace sizing",
            ),
        };
        json!({"schema_version":3,"baseline":baseline_name(family, backend).unwrap(),"family":family,"n":n,
            "warmups":10,"samples":30,"input_recipe":"lcg1664525_1013904223_v1","vendor_versions":versions,
            "compilation_ms":1.0,"device_samples_ms":vec![1.0;30],"host_samples_ms":vec![2.0;30],
            "output_bytes":output_bytes,"input_bytes":4 * words,"cold_process_ms":3.0,
            "device_resident_boundary":device,"host_to_host_boundary":host})
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "invar-family-baseline-{}-{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir(&dir).unwrap();
        dir
    }

    #[test]
    fn family_baseline_runner_accepts_only_the_exact_reference_output() {
        for backend in [Backend::Cuda, Backend::Metal] {
            for family in [
                "transpose",
                "stencil",
                "segsum",
                "compact",
                "frontier",
                "map_lcg",
                "scan_lcg",
            ] {
                if baseline_name(family, backend).is_none() {
                    // Not every fixture family exists on every backend.
                    assert!(expected(family, 256, backend).is_err());
                    continue;
                }
                let n = 256;
                let (want, _) = expected(family, n, backend).unwrap();
                let dir = scratch(family);
                let good = fake(
                    &dir,
                    &want,
                    &document(family, n, backend, want.len() * 4),
                    0,
                );
                let report =
                    run(&good, family, n, backend).unwrap_or_else(|e| panic!("{family}: {e}"));
                assert_eq!(
                    report["full_output_correctness"], "exact_full_output_pass",
                    "{family}"
                );
                assert_eq!(report["status"], "measured_correct");

                // One flipped word anywhere fails the whole output.
                let mut wrong = want.clone();
                *wrong.last_mut().unwrap() ^= 1;
                let bad = fake(
                    &dir,
                    &wrong,
                    &document(family, n, backend, wrong.len() * 4),
                    0,
                );
                let report = run(&bad, family, n, backend).unwrap();
                assert_eq!(
                    report["full_output_correctness"], "exact_full_output_fail",
                    "{family}"
                );
                assert_eq!(report["status"], "failed");
                fs::remove_dir_all(dir).unwrap();
            }
        }
    }

    #[test]
    fn family_baseline_fake_fixtures_survive_parallel_forks() {
        // Each worker writes and executes its own fixture while the others fork
        // children; a writable descriptor inherited by a sibling fork would make
        // exec fail with ETXTBSY.
        let (family, n, backend) = ("transpose", 64, Backend::Cuda);
        let (want, _) = expected(family, n, backend).unwrap();
        let doc = document(family, n, backend, want.len() * 4);
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..16)
                .map(|worker| {
                    let (want, doc) = (&want, &doc);
                    scope.spawn(move || {
                        let dir = scratch(&format!("forks-{worker}"));
                        for _ in 0..8 {
                            let script = fake_padded(&dir, want, doc, 0, 1024 * 1024);
                            let report = run(&script, family, n, backend)
                                .unwrap_or_else(|e| panic!("worker {worker}: {e}"));
                            assert_eq!(report["full_output_correctness"], "exact_full_output_pass");
                        }
                        fs::remove_dir_all(dir).unwrap();
                    })
                })
                .collect();
            for worker in workers {
                worker.join().unwrap();
            }
        });
    }

    #[test]
    fn family_baseline_concurrent_runs_get_distinct_scratch_roots() {
        // Released together, these runs name their scratch roots within one
        // clock tick; a time-only name collides on macOS.
        let (family, n, backend) = ("transpose", 64, Backend::Cuda);
        let (want, _) = expected(family, n, backend).unwrap();
        let dir = scratch("roots");
        let script = fake(
            &dir,
            &want,
            &document(family, n, backend, want.len() * 4),
            0,
        );
        let start = std::sync::Barrier::new(32);
        std::thread::scope(|scope| {
            let runs: Vec<_> = (0..32)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        run(&script, family, n, backend)
                    })
                })
                .collect();
            for run in runs {
                assert_eq!(
                    run.join().unwrap().unwrap()["full_output_correctness"],
                    "exact_full_output_pass"
                );
            }
        });
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn family_baseline_runner_refuses_protocol_and_accounting_faults() {
        let (family, n, backend) = ("transpose", 64, Backend::Cuda);
        let (want, _) = expected(family, n, backend).unwrap();
        let dir = scratch("faults");
        let doc = document(family, n, backend, want.len() * 4);
        for (field, value) in [
            ("schema_version", json!(2)),
            ("family", json!("gather")),
            ("n", json!(n + 1)),
            ("warmups", json!(9)),
            ("input_recipe", json!("other")),
            ("compilation_ms", json!(0.0)),
            ("output_bytes", json!(want.len() * 4 + 4)),
            ("input_bytes", json!(n * 4 + 4)),
            ("input_bytes", Value::Null),
            ("cold_process_ms", json!(-1.0)),
            ("cold_process_ms", Value::Null),
            ("device_resident_boundary", json!("other")),
            ("host_to_host_boundary", Value::Null),
        ] {
            let mut d = doc.clone();
            d[field] = value;
            let script = fake(&dir, &want, &d, 0);
            assert!(
                run(&script, family, n, backend).is_err(),
                "accepted a bad {field}"
            );
        }
        // 30 numbers plus a string is not 30 samples.
        let mut padded = doc.clone();
        padded["device_samples_ms"]
            .as_array_mut()
            .unwrap()
            .push(json!("x"));
        assert!(run(&fake(&dir, &want, &padded, 0), family, n, backend).is_err());
        let mut no_cub = doc.clone();
        no_cub["vendor_versions"]["cub"] = json!(0);
        assert!(run(&fake(&dir, &want, &no_cub, 0), family, n, backend).is_err());
        let failed = fake(&dir, &want, &doc, 3);
        assert!(
            run(&failed, family, n, backend)
                .unwrap_err()
                .contains("exited")
        );
        // The runner never accepts a family or size outside its contract.
        assert!(run(&failed, "unknown", n, backend).is_err());
        assert!(run(&failed, family, 0, backend).is_err());
        assert!(expected("segsum", 17, backend).is_err());
        assert!(expected("frontier", 17, backend).is_err());
        assert!(expected("gather_hw", n, Backend::Cuda).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
