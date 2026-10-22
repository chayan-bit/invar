use serde_json::{Value, json};
use std::{
    fs,
    io::Read,
    path::Path,
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use super::hash::sha256;

const MAX_OUTPUT: u64 = 4 * 1024 * 1024;
const MAX_LOG: u64 = 64 * 1024;
const TIMEOUT: Duration = Duration::from_secs(30);
// macOS SystemTime has only microsecond resolution, so concurrent runs in one
// process also need a counter for distinct scratch roots (#255).
static NEXT_ROOT: AtomicU64 = AtomicU64::new(0);

fn summary(samples: &Value) -> Result<Value, String> {
    let mut values = samples
        .as_array()
        .ok_or("missing samples")?
        .iter()
        .map(|v| v.as_f64().ok_or_else(|| "invalid sample".to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() != 30 {
        return Err("expected 30 timing samples".into());
    }
    values.sort_by(f64::total_cmp);
    Ok(
        json!({"median_ms":(values[14]+values[15])/2.0,"q1_ms":values[7],
        "q3_ms":values[22],"iqr_ms":values[22]-values[7],
        "minimum_ms":values[0],"maximum_ms":values[29],
        "median_ci95_ms":[values[9],values[20]],
        "uncertainty":"two-sided exact binomial order-statistic interval for median; 95.72% coverage for independent continuous samples; ties conservative"}),
    )
}

fn read_limit(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let meta = fs::metadata(path).map_err(|e| e.to_string())?;
    if !meta.is_file() || meta.len() > limit {
        return Err("baseline artifact size exceeded".into());
    }
    let file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > limit {
        return Err("baseline artifact size exceeded during read".into());
    }
    Ok(bytes)
}

pub fn run(
    executable: &Path,
    family: &str,
    n: usize,
    v2: bool,
    one_shot: bool,
) -> Result<Value, String> {
    if ![
        "map", "sum", "scan", "affine", "f32sum", "f32scan", "f32gemm",
    ]
    .contains(&family)
        || !(1..=1_048_576).contains(&n)
    {
        return Err("invalid baseline family or element count".into());
    }
    if v2 && family != "sum" {
        return Err("v2 matched baseline currently supports reduction only".into());
    }
    if one_shot && !v2 {
        return Err("baseline cold child requires v2".into());
    }
    let executable_bytes = read_limit(executable, 64 * 1024 * 1024)?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "invar-bench-{}-{nonce}-{}",
        std::process::id(),
        NEXT_ROOT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&root).map_err(|e| e.to_string())?;
    let output = root.join("output.bin");
    let input = root.join("input.bin");
    let stdout = root.join("stdout.json");
    let stderr = root.join("stderr.txt");
    let run_result = (|| {
        let out_file = fs::File::create(&stdout).map_err(|e| e.to_string())?;
        let err_file = fs::File::create(&stderr).map_err(|e| e.to_string())?;
        let mut command = Command::new(executable);
        command.arg(family).arg(n.to_string()).arg(&output);
        if v2 {
            command.arg(&input);
        }
        if one_shot {
            command.arg("--cold-child");
        }
        let mut child = command
            .stdin(Stdio::null())
            .stdout(Stdio::from(out_file))
            .stderr(Stdio::from(err_file))
            .spawn()
            .map_err(|e| e.to_string())?;
        let started = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
                break status;
            }
            if started.elapsed()
                >= if one_shot {
                    Duration::from_secs(25)
                } else {
                    TIMEOUT
                }
            {
                let _ = child.kill();
                let _ = child.wait();
                return Err("baseline process timed out".into());
            }
            thread::sleep(Duration::from_millis(10));
        };
        if !status.success() {
            let err = read_limit(&stderr, MAX_LOG).unwrap_or_default();
            let detail = String::from_utf8_lossy(&err);
            return Err(format!(
                "baseline exited {status}: {}",
                detail.chars().take(200).collect::<String>()
            ));
        }
        let document: Value =
            serde_json::from_slice(&read_limit(&stdout, MAX_LOG)?).map_err(|e| e.to_string())?;
        if document["schema_version"] != if v2 { 2 } else { 1 }
            || document["family"] != family
            || document["n"] != n
            || document["warmups"] != if one_shot { 0 } else { 10 }
            || document["samples"] != if one_shot { 1 } else { 30 }
        {
            return Err("baseline protocol mismatch".into());
        }
        if v2
            && (document["input_recipe"] != "lcg1664525_1013904223_plus_slot17_v1"
                || document["device_registry_id"]
                    .as_u64()
                    .is_none_or(|id| id == 0)
                || document["device_resident_boundary"] != "metal_command_buffer_gpu_start_end_v1"
                || document["host_to_host_boundary"]
                    != "metal_runner_after_compile_through_host_output_copy_v1")
        {
            return Err("v2 baseline input or timing boundary mismatch".into());
        }
        let baseline_name = document["baseline"]
            .as_str()
            .ok_or("missing baseline name")?;
        let apple = baseline_name.starts_with("Apple MPS");
        let graph = baseline_name.starts_with("Apple MPSGraph");
        let handwritten = baseline_name == "Handwritten Metal affine quality control";
        let metal = apple || handwritten;
        let (handwritten_workspace, handwritten_launches) = if handwritten {
            let (mut count, mut workspace, mut levels) = (n, 0usize, 0usize);
            loop {
                let groups = count.div_ceil(256);
                workspace += (count + groups) * 8;
                levels += 1;
                if groups == 1 {
                    break;
                }
                count = groups;
            }
            (workspace, 2 * levels)
        } else {
            (0, 0)
        };
        let metal_framework = if graph {
            "MetalPerformanceShadersGraph system framework"
        } else if handwritten {
            "Metal system framework"
        } else {
            "MetalPerformanceShaders system framework"
        };
        let valid_name = match family {
            "map" => [
                "CUB DeviceTransform",
                "Thrust transform (CUB DeviceTransform unavailable)",
                "Apple MPSGraph elementwise",
            ]
            .contains(&baseline_name),
            "sum" if v2 => baseline_name == "Apple MPSGraph reductionSum v2",
            "sum" => ["CUB", "Apple MPSGraph reductionSum"].contains(&baseline_name),
            "scan" => ["CUB", "Apple MPSGraph cumulativeSum"].contains(&baseline_name),
            "affine" => [
                "CUB",
                "Apple MPSGraph affinePrefix",
                "Handwritten Metal affine quality control",
            ]
            .contains(&baseline_name),
            "f32sum" => ["CUB", "Apple MPSImageReduceRowSum"].contains(&baseline_name),
            "f32scan" => ["CUB", "Apple MPSImageIntegral"].contains(&baseline_name),
            "f32gemm" => ["Apple MPSMatrixMultiplication", "cuBLAS SGEMM"].contains(&baseline_name),
            _ => false,
        };
        if !valid_name
            || (metal
                && (document["vendor_versions"]["framework"] != metal_framework
                    || document["vendor_versions"]["macos"]
                        .as_array()
                        .is_none_or(|x| x.len() != 3)))
            || ((graph || handwritten)
                && (document["compilation_ms"]
                    .as_f64()
                    .is_none_or(|x| !x.is_finite() || x <= 0.0)))
            || (graph
                && document["workspace_scope"]
                    != "explicit buffers only; framework internals unknown")
            || (handwritten
                && (document["workspace_scope"] != "explicit pair prefixes and block totals"
                    || document["workspace_bytes"] != handwritten_workspace
                    || document["launches_per_call"] != handwritten_launches))
            || (baseline_name == "cuBLAS SGEMM"
                && (document["vendor_versions"]["math_mode"] != "pedantic"
                    || ["cublas", "cuda_runtime", "driver"].iter().any(|key| {
                        document["vendor_versions"][key]
                            .as_u64()
                            .is_none_or(|v| v == 0)
                    })))
            || (!metal
                && baseline_name != "cuBLAS SGEMM"
                && ["cub", "cuda_runtime", "driver"].iter().any(|key| {
                    document["vendor_versions"][key]
                        .as_u64()
                        .is_none_or(|v| v == 0)
                }))
        {
            return Err("invalid baseline provenance".into());
        }
        for key in ["device_samples_ms", "host_samples_ms"] {
            let samples = document[key].as_array().ok_or("missing raw samples")?;
            if samples.len() != if one_shot { 1 } else { 30 }
                || samples.iter().any(|v| {
                    v.as_f64()
                        .is_none_or(|x| !x.is_finite() || !(0.0..=30_000.0).contains(&x))
                })
            {
                return Err("invalid raw samples".into());
            }
        }
        let values = read_limit(&output, MAX_OUTPUT)?;
        let (input_recipe, input_sha256) = if v2 {
            let actual = read_limit(&input, MAX_OUTPUT)?;
            let mut expected = Vec::with_capacity(n * 4);
            for i in 0..n {
                expected.extend_from_slice(
                    &((i as u32)
                        .wrapping_mul(1_664_525)
                        .wrapping_add(1_013_904_223))
                    .to_le_bytes(),
                );
            }
            if actual != expected {
                return Err("v2 baseline input differs from generated recipe".into());
            }
            (
                json!("lcg1664525_1013904223_plus_slot17_v1"),
                json!(sha256(&actual)),
            )
        } else {
            (Value::Null, Value::Null)
        };
        let count = if family == "sum" || family == "f32sum" {
            1
        } else if family == "f32gemm" {
            n.checked_mul(n).ok_or("GEMM size overflow")?
        } else {
            n
        };
        if values.len() != count.checked_mul(4).ok_or("output size overflow")? {
            return Err("incomplete baseline output".into());
        }
        let input_bytes = if family == "affine" || family == "map" {
            n.checked_mul(8)
        } else if family == "f32gemm" {
            count.checked_mul(8)
        } else {
            n.checked_mul(4)
        }
        .ok_or("input size overflow")?;
        if !metal && baseline_name != "cuBLAS SGEMM" {
            let vendor = document["vendor_temporary_bytes"]
                .as_u64()
                .ok_or("missing vendor temporary bytes")?;
            let intermediate = document["explicit_intermediate_bytes"]
                .as_u64()
                .ok_or("missing explicit intermediate bytes")?;
            let total = vendor
                .checked_add(intermediate)
                .ok_or("baseline scratch byte overflow")?;
            let expected_intermediate = if family == "affine" {
                input_bytes as u64
            } else {
                0
            };
            if intermediate != expected_intermediate
                || document["total_scratch_bytes"] != total
                || document["workspace_bytes"] != total
                || document["workspace_scope"]
                    != "vendor temporary allocation plus explicit fixture intermediate"
            {
                return Err("baseline scratch accounting mismatch".into());
            }
        }
        if document["input_bytes"] != input_bytes
            || document["output_bytes"] != values.len()
            || document["workspace_bytes"]
                .as_u64()
                .is_none_or(|x| x > 64 * 1024 * 1024)
            || document["cold_process_ms"]
                .as_f64()
                .is_none_or(|x| !x.is_finite() || x <= 0.0)
        {
            return Err("baseline byte accounting or cold timing mismatch".into());
        }
        let correctness = match family {
            "map" | "sum" | "scan" | "affine" => {
                let mut expected = Vec::with_capacity(count);
                let mut sum = if v2 { 17u32 } else { 0u32 };
                let mut affine = 17u32;
                for i in 0..n {
                    let x = if v2 {
                        (i as u32)
                            .wrapping_mul(1_664_525)
                            .wrapping_add(1_013_904_223)
                    } else {
                        (i as u32).wrapping_mul(17).wrapping_add(3)
                    };
                    sum = sum.wrapping_add(x);
                    affine = (i as u32)
                        .wrapping_mul(3)
                        .wrapping_add(2)
                        .wrapping_mul(affine)
                        .wrapping_add((i as u32).wrapping_mul(7).wrapping_add(5));
                    if family == "scan" {
                        expected.push(sum);
                    }
                    if family == "map" {
                        expected.push(
                            x.wrapping_add((i as u32).wrapping_mul(7).wrapping_add(5))
                                .wrapping_mul(7)
                                .wrapping_add(11),
                        );
                    }
                    if family == "affine" {
                        expected.push(affine);
                    }
                }
                if family == "sum" {
                    expected.push(sum);
                }
                if values
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .eq(expected)
                {
                    "exact_full_output_pass"
                } else {
                    "exact_full_output_fail"
                }
            }
            _ => "pending_m4_numeric_policy",
        };
        let numeric_comparison = if family.starts_with("f32") {
            let mut expected = Vec::with_capacity(count);
            if family == "f32gemm" {
                for row in 0..n {
                    for col in 0..n {
                        let mut sum = 0.0f32;
                        for k in 0..n {
                            let a = ((row * n + k) % 13) as f32 / 13.0;
                            let b = ((k * n + col) % 11) as f32 / 11.0;
                            sum += a * b;
                        }
                        expected.push(sum);
                    }
                }
            } else {
                let mut sum = 0.0f32;
                for i in 0..n {
                    sum += (i % 17) as f32 / 17.0;
                    if family == "f32scan" {
                        expected.push(sum);
                    }
                }
                if family == "f32sum" {
                    expected.push(sum);
                }
            }
            let mut max_abs = 0.0f32;
            let mut differing_bits = 0usize;
            for (observed, expected) in values.chunks_exact(4).zip(expected) {
                let observed = f32::from_le_bytes(observed.try_into().unwrap());
                if !observed.is_finite() {
                    return Err("nonfinite float baseline output".into());
                }
                if (observed - expected).abs() > 1e-3 * expected.abs().max(1.0) {
                    return Err(
                        "float baseline differs substantially from independent oracle".into(),
                    );
                }
                max_abs = max_abs.max((observed - expected).abs());
                if observed.to_bits() != expected.to_bits() {
                    differing_bits += 1;
                }
            }
            json!({"reference":"independent ordered f32 host loop","compared_elements":count,
                "max_absolute_difference":max_abs,"bitwise_differences":differing_bits,
                "acceptance":"pending M4-01 numeric policy"})
        } else {
            Value::Null
        };
        let device_summary = if one_shot {
            Value::Null
        } else {
            summary(&document["device_samples_ms"])?
        };
        let host_summary = if one_shot {
            Value::Null
        } else {
            summary(&document["host_samples_ms"])?
        };
        let opaque_vendor = apple || baseline_name == "cuBLAS SGEMM";
        let vendor_temporary_bytes = if handwritten {
            json!(0)
        } else {
            document["vendor_temporary_bytes"].clone()
        };
        let explicit_intermediate_bytes = if opaque_vendor || handwritten {
            document["workspace_bytes"].clone()
        } else {
            document["explicit_intermediate_bytes"].clone()
        };
        let total_scratch_bytes = if opaque_vendor {
            Value::Null
        } else {
            document["workspace_bytes"].clone()
        };
        let fixture_source_sha256 = if v2 {
            sha256(include_bytes!("baselines/metal-mpsgraph-reduction-v2.mm"))
        } else if baseline_name == "Apple MPSMatrixMultiplication" {
            sha256(include_bytes!("baselines/metal-mps-gemm.mm"))
        } else if graph {
            sha256(include_bytes!("baselines/metal-mpsgraph-u32.mm"))
        } else if handwritten {
            sha256(include_bytes!("baselines/metal-affine-handwritten.mm"))
        } else if apple {
            sha256(include_bytes!("baselines/metal-mps-image.mm"))
        } else if baseline_name == "cuBLAS SGEMM" {
            sha256(include_bytes!("baselines/cuda-cublas-gemm.cu"))
        } else {
            sha256(include_bytes!("baselines/cuda-cub.cu"))
        };
        let mut result = json!({
            "schema_version":if v2 {2} else {1},"kind":if handwritten {"handwritten_quality_control"} else {"independent_library_baseline"},"family":family,
            "elements":n,
            "status":if correctness=="exact_full_output_pass" {"measured_correct"} else {"pending_or_failed"},
            "full_output_correctness":correctness,"arithmetic_compatibility":if family.starts_with("f32") {"pending_m4_01"} else {"u32_wrapping_v1"},
            "executable_sha256":sha256(&executable_bytes),"output_sha256":sha256(&values),
            "input_recipe":input_recipe,"input_sha256":input_sha256,
            "fixture_source_sha256":fixture_source_sha256,
            "baseline_name":baseline_name,"vendor_versions":document["vendor_versions"],
            "target_compatibility":if metal {"pending M4-01 policy"} else if document["vendor_versions"]["cuda_runtime"].as_u64().is_some_and(|v|v>=12040) {"toolkit_minimum_met"} else {"toolkit_below_invar_minimum"},
            "numeric_comparison":numeric_comparison,
            "build_provenance":"executable_hash_only; source/build pin pending",
            "device_resident_boundary":document["device_resident_boundary"],
            "host_to_host_boundary":document["host_to_host_boundary"],
            "device_resident_samples_ms":document["device_samples_ms"],
            "host_to_host_samples_ms":document["host_samples_ms"],
            "device_resident_summary":device_summary,"host_to_host_summary":host_summary,
            "cold_process_ms":document["cold_process_ms"],
            "cold_boundary":document["cold_boundary"],"compilation_ms":document["compilation_ms"],
            "input_bytes":document["input_bytes"],"output_bytes":document["output_bytes"],"workspace_bytes":document["workspace_bytes"],
            "vendor_temporary_bytes":vendor_temporary_bytes,"explicit_intermediate_bytes":explicit_intermediate_bytes,"total_scratch_bytes":total_scratch_bytes,
            "required_transfer_bytes":input_bytes+values.len(),"launches_per_call":if handwritten {document["launches_per_call"].clone()} else if apple {Value::Null} else if family=="affine" {json!(2)} else {json!(1)},
            "transfers_per_call":if family=="map" || family=="f32gemm" || (graph && family=="affine") {3} else {2},
            "allocation_policy":document["allocation_policy"].as_str().unwrap_or("buffers and workspace preallocated and reused for warm samples"),
            "workspace_scope":document["workspace_scope"].as_str().unwrap_or(if metal {"explicit buffers only; vendor internals unknown"} else if opaque_vendor {"explicit fixture scratch only; vendor internals unknown"} else {"vendor temporary allocation plus explicit fixture intermediate"}),
            "warmups":if one_shot {0} else {10},"samples":if one_shot {1} else {30},
            "timing_source":"vendor fixture",
            "gpu_performance":"baseline_only; no generated ratio",
        });
        result["device_registry_id"] = document["device_registry_id"].clone();
        Ok(result)
    })();
    let _ = fs::remove_dir_all(&root);
    run_result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn issue_m5_01_benchmark_summary_ci() {
        let constant = summary(&json!(vec![1.25; 30])).unwrap();
        assert_eq!(constant["median_ci95_ms"], json!([1.25, 1.25]));
        let samples = json!((0..30).map(f64::from).collect::<Vec<_>>());
        let observed = summary(&samples).unwrap();
        assert_eq!(observed, summary(&samples).unwrap());
        assert_eq!(observed["median_ci95_ms"], json!([9.0, 20.0]));
    }
}
