use serde_json::{Value, json};

use super::{corpus, hash::sha256};

const FAMILIES: [&str; 3] = ["map", "reduction", "scan"];
const FUTURE: [&str; 7] = [
    "compaction",
    "transpose",
    "stencil",
    "matrix",
    "segmented",
    "gather_scatter",
    "frontier",
];

fn prime(n: u32) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2;
    while d <= n / d {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 1;
    }
    true
}

fn adjacent_primes(n: u32) -> (u32, u32) {
    let mut lo = n - 1;
    let mut hi = n + 1;
    while !prime(lo) {
        lo -= 1;
    }
    while !prime(hi) {
        hi += 1;
    }
    (lo, hi)
}

pub fn sizes() -> Vec<u32> {
    let mut sizes = vec![0, 1, 31, 32, 33, 255, 256, 257, 4093];
    for exponent in (10..=28).step_by(2) {
        let n = 1u32 << exponent;
        let (lo, hi) = adjacent_primes(n);
        sizes.extend([lo, n, hi]);
    }
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

/// The current v1 manifest: the measured base plus the provisional families.
pub fn build(corpus: &Value, corpus_bytes: &[u8]) -> Result<Value, String> {
    let mut manifest = base(corpus, corpus_bytes)?;
    append_provisional(&mut manifest)?;
    Ok(manifest)
}

/// The map/reduction/scan base that the pending v2 proposal is derived from.
pub fn base(corpus: &Value, corpus_bytes: &[u8]) -> Result<Value, String> {
    let held = corpus::held_out_hash(corpus)?;
    let cuda_source_sha256 = sha256(include_bytes!("baselines/cuda-cub.cu"));
    let cuda_cublas_source_sha256 = sha256(include_bytes!("baselines/cuda-cublas-gemm.cu"));
    let metal_mps_source_sha256 = sha256(include_bytes!("baselines/metal-mps-gemm.mm"));
    let metal_mps_image_source_sha256 = sha256(include_bytes!("baselines/metal-mps-image.mm"));
    let metal_mpsgraph_u32_source_sha256 =
        sha256(include_bytes!("baselines/metal-mpsgraph-u32.mm"));
    let mut cases = Vec::new();
    for family in FAMILIES {
        for n in sizes() {
            let (inputs, outputs) = match family {
                "map" => (2u64, 1u64),
                "reduction" => (1, 1),
                _ => (2, 1),
            };
            let bytes = (n as u64)
                .checked_mul(4)
                .and_then(|x| x.checked_mul(inputs + outputs))
                .ok_or("byte count overflow")?;
            let law = match family {
                "map" => "wrapping_map",
                "reduction" => "wrapping_sum",
                "scan" => "affine_composition",
                _ => unreachable!(),
            };
            cases.push(json!({"id":format!("{family}-n{n}"),"family":family,"law":law,"elements":n,"input_buffers":inputs,"output_buffers":outputs,"stages":if family=="map" {3} else {1},"required_bytes":bytes,"numeric_policy":"u32-wrapping-v1","timing":["device_resident","host_to_host","cold_process"],"memory_preflight":"required"}));
        }
    }
    for n in [33u32, 4093, 1 << 20] {
        cases.push(json!({"id":format!("map-long-chain-n{n}"),"family":"map","law":"wrapping_map","elements":n,"input_buffers":4,"output_buffers":4,"stages":16,"required_bytes":(n as u64)*32,"numeric_policy":"u32-wrapping-v1","timing":["device_resident","host_to_host","cold_process"],"memory_preflight":"required"}));
    }
    let baselines = json!({
        "cuda": {
            "map":{"library":"CUB DeviceTransform on CUDA 12.9+; Thrust transform on older supported toolkit","status":"pending-pinned-toolkit-and-semantic-review","independent":true},
            "reduction":{"library":"CUB DeviceReduce","status":"pending-pinned-toolkit-and-measurement","independent":true},
            "scan":{"library":"CUB DeviceScan for modular sum; affine custom-op review pending","status":"pending-pinned-toolkit-and-measurement","independent":true}
        },
        "metal": {
            "map":{"library":"MPSGraph UInt32 elementwise","status":"pending-baseline-quality-and-build-review","independent":true},
            "reduction":{"library":"MPSGraph UInt32 reductionSum","status":"pending-baseline-quality-and-build-review","independent":true},
            "scan":{"library":"MPSGraph UInt32 affine-prefix graph composition; handwritten quality comparison pending","status":"pending-baseline-quality-and-build-review","independent":true}
        }
    });
    Ok(json!({
        "schema_version":1,"freeze_status":"provisional-pre-m4-01","seed":corpus::SEED,
        "corpus_sha256":sha256(corpus_bytes),"held_out_sha256":held,
        "families":FAMILIES,"pending_families":FUTURE,"tracks":["generated","library-assisted","same-structure"],
        "targets":[{"name":"RTX A6000","compute_capability":"sm_86","status":"pending-preflight"},{"name":"local Apple GPU","status":"pending-preflight"}],
        "baselines":baselines,"baseline_fixture_sources":{"cuda_cub_sha256":cuda_source_sha256,"cuda_cublas_gemm_sha256":cuda_cublas_source_sha256,"metal_mps_gemm_sha256":metal_mps_source_sha256,"metal_mps_image_sha256":metal_mps_image_source_sha256,"metal_mpsgraph_u32_sha256":metal_mpsgraph_u32_source_sha256},"cases":cases,
        "acceptance":{"large_case_ratio_geomean_max":1.20,"large_case_ratio_each_max":1.50,"library_ratio_each_max":1.25,"warmups":10,"samples":30},
        "report_schema_version":1
    }))
}

pub fn family_counts(manifest: &Value) -> Value {
    let mut counts = serde_json::Map::new();
    for family in manifest["families"].as_array().into_iter().flatten() {
        let Some(name) = family.as_str() else {
            continue;
        };
        let count = manifest["cases"].as_array().map_or(0, |cases| {
            cases.iter().filter(|c| c["family"] == name).count()
        });
        counts.insert(name.into(), json!(count));
    }
    Value::Object(counts)
}

/// Refuses a case that this revision records as `unavailable` (or whose family
/// has an `unavailable` baseline on `backend`) from being used as a measured result.
pub fn require_measurable(
    manifest: &Value,
    case: &Value,
    backend: Option<&str>,
) -> Result<(), String> {
    let family = case["family"].as_str().unwrap_or("");
    let baseline = backend.map(|b| &manifest["baselines"][b][family]);
    for entry in [Some(case), baseline].into_iter().flatten() {
        if entry["status"] == "unavailable" {
            return Err(format!(
                "manifest entry for {} is unavailable: {}",
                case["id"].as_str().unwrap_or(family),
                entry["reason"].as_str().unwrap_or("no reason recorded")
            ));
        }
    }
    Ok(())
}

const CASE_REASON: &str = "provisional M5-01 revision: no generated device path, independent baseline or GPU measurement exists for this family; the shape is recorded from the frozen contract table only";
const BASELINE_REASON: &str = "provisional M5-01 revision: no pinned library or toolkit version, reviewed fixture source or GPU measurement for this family";
const TARGET_REASON: &str = "provisional M5-01 revision: no device path for this family and no authorized preflight on this target";

/// The seven families of `docs/performance-acceptance.md` ("Frozen workload
/// families") without measurable cases yet. Source behavior, shapes and the
/// CUDA comparison are quoted from that table; nothing else is bound.
fn provisional_families() -> [(&'static str, &'static str, &'static str, Vec<Value>); 7] {
    let classes = |names: &[&str]| -> Vec<Value> {
        names.iter().map(|name| json!({"class": name})).collect()
    };
    [
        (
            "compaction",
            "Predicate preserves source order and handles empty results",
            "Stable CUDA/CUB selection with matching output order",
            [0, 1, 50, 99, 100]
                .map(|p| json!({"selectivity_percent": p}))
                .to_vec(),
        ),
        (
            "transpose",
            "Rectangular row-major transpose and strided read view",
            "Shared-memory tiled CUDA transpose",
            [[31, 33], [127, 257], [1024, 1024]]
                .map(|[rows, cols]| json!({"rows": rows, "cols": cols}))
                .into_iter()
                .chain(classes(&["larger bounded cases"]))
                .collect(),
        ),
        (
            "stencil",
            "Out-of-place radius-1 2D stencil, then bounded multi-step ping-pong",
            "Tiled CUDA with identical boundary and numeric policy",
            classes(&[
                "tiny domains",
                "halo boundaries",
                "odd widths",
                "wide/tall",
                "larger square",
            ]),
        ),
        (
            "matrix",
            "Recognized loop contraction, FP32 ordered track and explicit Tensor Core contract track",
            "Semantically matched CUDA; cuBLAS/CUTLASS reported as additional reference where contracts match",
            classes(&[
                "rectangular",
                "1-sized dimensions",
                "zero reduction extent",
                "odd dimensions",
            ])
            .into_iter()
            .chain(
                [256, 1024, 4096]
                    .map(|d| json!({"dimension": d, "condition": "where budget permits"})),
            )
            .collect(),
        ),
        (
            "segmented",
            "Per-segment maps/reductions and a nested loop with data-dependent segment lengths",
            "Reviewed segmented CUDA/CUB implementation",
            classes(&[
                "empty segments",
                "uniform",
                "skewed",
                "one dominant segment",
                "ragged totals",
            ]),
        ),
        (
            "gather_scatter",
            "Read gather, injective scatter, lawful modular histogram conflict mode",
            "CUDA gather/scatter/atomic histogram with identical conflict semantics",
            classes(&[
                "sorted/random indices",
                "duplicates",
                "bounds negatives for rejection",
            ]),
        ),
        (
            "frontier",
            "Bounded level-synchronous traversal or worklist transformation with declared set/order observations",
            "Equivalent level-synchronous CUDA implementation",
            classes(&[
                "chains",
                "stars",
                "balanced and skewed small graphs",
                "bounded sparse larger cases",
            ]),
        ),
    ]
}

/// Planned independent baselines named by the M5-01 and M4 contracts.
fn planned_baseline(family: &str) -> (&'static str, &'static str) {
    const HANDWRITTEN_METAL: &str =
        "reviewed handwritten Metal kernel pinned by content hash (no Apple library named)";
    match family {
        "compaction" => ("CUB DeviceSelect", HANDWRITTEN_METAL),
        "transpose" => (
            "NVIDIA reference tiled shared-memory transpose",
            "reviewed handwritten Metal transpose pinned by content hash unless an MPS primitive with the same contract exists",
        ),
        "stencil" => (
            "reviewed handwritten tiled CUDA stencil pinned by content hash (no library primitive)",
            HANDWRITTEN_METAL,
        ),
        "matrix" => ("cuBLAS and CUTLASS", "MPSMatrixMultiplication"),
        "segmented" => ("CUB DeviceSegmentedReduce", HANDWRITTEN_METAL),
        "gather_scatter" => (
            "CUB DeviceHistogram for the histogram mode; gather/scatter per the contract table",
            HANDWRITTEN_METAL,
        ),
        _ => (
            "equivalent level-synchronous CUDA implementation (no library named)",
            HANDWRITTEN_METAL,
        ),
    }
}

fn append_provisional(manifest: &mut Value) -> Result<(), String> {
    let object = manifest
        .as_object_mut()
        .ok_or("manifest is not an object")?;
    object.remove("pending_families");
    let mut families: Vec<Value> = FAMILIES.iter().map(|f| json!(f)).collect();
    let mut cases = Vec::new();
    let mut detail = serde_json::Map::new();
    for (family, source, comparison, shapes) in provisional_families() {
        families.push(json!(family));
        for shape in shapes {
            let suffix = match (
                &shape["rows"],
                &shape["selectivity_percent"],
                &shape["dimension"],
            ) {
                (Value::Number(rows), _, _) => format!("{rows}x{}", shape["cols"]),
                (_, Value::Number(percent), _) => format!("selectivity-{percent}pct"),
                (_, _, Value::Number(dimension)) => format!("dim-{dimension}"),
                _ => shape["class"]
                    .as_str()
                    .ok_or("provisional shape without class")?
                    .replace(['/', ' '], "-"),
            };
            cases.push(
                json!({"id":format!("{family}-{suffix}"),"family":family,"shape":shape,
                "status":"unavailable","reason":CASE_REASON}),
            );
        }
        let (cuda, metal) = planned_baseline(family);
        for (backend, library) in [("cuda", cuda), ("metal", metal)] {
            manifest_baselines(object, backend)?.insert(
                family.into(),
                json!({"planned":library,"independent":true,"status":"unavailable","reason":BASELINE_REASON}),
            );
        }
        detail.insert(family.into(), json!({"source":source,"cuda_comparison":comparison,
            "shape_source":"docs/performance-acceptance.md, Frozen workload families",
            "targets":[{"name":"RTX A6000","compute_capability":"sm_86","status":"unavailable","reason":TARGET_REASON},
                {"name":"local Apple GPU","status":"unavailable","reason":TARGET_REASON}]}));
    }
    object.insert("families".into(), Value::Array(families));
    object.insert("provisional_families".into(), Value::Object(detail));
    object
        .get_mut("cases")
        .and_then(Value::as_array_mut)
        .ok_or("missing cases")?
        .extend(cases);
    object.insert("held_out_freeze".into(), json!({
        "status":"deferred",
        "scope":"held_out_sha256 covers the provisional pre-M4-01 corpus (map, reduction, scan) and is unchanged by this revision",
        "condition":"the permanent held-out freeze is made after the M4-02..M4-05 device paths exist, by a reviewed manifest version bump, before any M4-06 tuning"
    }));
    object.insert(
        "v2_proposal".into(),
        json!("manifest-v2-proposal.json is derived from the map/reduction/scan base and stays pending M5 review; this revision neither adopts nor rejects it"),
    );
    Ok(())
}

fn manifest_baselines<'a>(
    object: &'a mut serde_json::Map<String, Value>,
    backend: &str,
) -> Result<&'a mut serde_json::Map<String, Value>, String> {
    object
        .get_mut("baselines")
        .and_then(|b| b.get_mut(backend))
        .and_then(Value::as_object_mut)
        .ok_or_else(|| format!("missing {backend} baselines"))
}

/// A runnable provisional contract; neither schema is frozen for M5 acceptance.
pub fn propose_v2(v1: &Value) -> Result<Value, String> {
    if v1["schema_version"] != 1 || v1["report_schema_version"] != 1 {
        return Err("expected v1 manifest".into());
    }
    let mut v2 = v1.clone();
    let cases = v2["cases"].as_array_mut().ok_or("missing v1 cases")?;
    for case in cases {
        if case["family"] == "reduction" {
            let n = case["elements"]
                .as_u64()
                .ok_or("invalid reduction length")?;
            case["required_bytes"] = json!(
                n.checked_mul(4)
                    .and_then(|bytes| bytes.checked_add(4))
                    .ok_or("reduction required byte overflow")?
            );
        }
    }
    v2["schema_version"] = json!(2);
    v2["report_schema_version"] = json!(2);
    v2["freeze_status"] = json!("proposal_pending_m5_review");
    v2["baseline_fixture_sources"]["metal_mpsgraph_reduction_v2_sha256"] = json!(sha256(
        include_bytes!("baselines/metal-mpsgraph-reduction-v2.mm")
    ));
    v2["byte_accounting"] = json!({
        "version":"source_required_u32_v2_proposal",
        "required_bytes":"bytes of source-observable required inputs plus outputs; reduction output is one u32 scalar; inclusive scan output is n u32 prefixes",
        "transfer_bytes":"actual host/device input and output transfer including a non-source-observable terminal-state word when present",
        "workspace_bytes":"explicit backend temporary allocations; never credited to required bytes",
        "ratio_status":"pending_review_and_regenerated_baselines"
    });
    Ok(v2)
}
