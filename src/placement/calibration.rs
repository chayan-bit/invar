//! Explicit offline probes. Unmeasured families stay unknown; no runtime tuning.
use super::*;
use std::{io::Write, time::Instant};
// Native builds probe the families on the CPU; CUDA and Metal builds run the
// same kernels as forced-device source, so each build uses only part of `cpu`.
#[cfg(any(
    feature = "native",
    feature = "cuda",
    all(feature = "metal", target_os = "macos"),
    test
))]
#[cfg_attr(not(all(feature = "native", feature = "cuda")), allow(dead_code))]
mod cpu;
#[cfg(any(
    feature = "native",
    feature = "cuda",
    all(feature = "metal", target_os = "macos"),
    test
))]
const SIZES: [usize; 2] = [4096, 262144];
#[cfg(any(
    feature = "native",
    feature = "cuda",
    all(feature = "metal", target_os = "macos")
))]
#[cfg(feature = "native")]
pub(crate) const MAP:&[u8]=b"fn map(xs:read Array<u32>,ys:mut Array<u32>)->u32 {for i in 0u32..len(xs) {ys[i]=wrapping_add(xs[i],1u32);} return len(xs);}";

fn command(program: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8(out.stdout).ok())
        .flatten()
        .map(|s| s.trim().to_owned())
}
pub fn observations() -> Value {
    #[cfg(target_os = "linux")]
    {
        let load = std::fs::read_to_string("/proc/loadavg")
            .ok()
            .and_then(|s| s.split_whitespace().next()?.parse::<f64>().ok());
        let swap = std::fs::read_to_string("/proc/meminfo").ok().and_then(|s| {
            let kb = |key| {
                s.lines()
                    .find(|l| l.starts_with(key))?
                    .split_whitespace()
                    .nth(1)?
                    .parse::<u64>()
                    .ok()
            };
            kb("SwapTotal:")?
                .checked_sub(kb("SwapFree:")?)?
                .checked_mul(1024)
        });
        json!({"load":load,"swap_bytes":swap})
    }
    #[cfg(target_os = "macos")]
    {
        let load = command("sysctl", &["-n", "vm.loadavg"])
            .and_then(|s| s.split_whitespace().nth(1)?.parse::<f64>().ok());
        let swap = command("sysctl", &["-n", "vm.swapusage"]).and_then(|s| {
            let words = s.split_whitespace().collect::<Vec<_>>();
            let n = words.iter().position(|w| *w == "used")?;
            let value = *words.get(n + 2)?;
            let multiplier = match value.as_bytes().last()? {
                b'M' => 1048576.0,
                b'G' => 1073741824.0,
                _ => return None,
            };
            let bytes = value[..value.len() - 1].parse::<f64>().ok()? * multiplier;
            (bytes.is_finite() && bytes >= 0.0).then_some(bytes as u64)
        });
        json!({"load":load,"swap_bytes":swap})
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    json!({"load":null,"swap_bytes":null})
}
pub fn host_identity(host_class: &str, device: &str, driver: &str) -> Value {
    let cpu = if cfg!(target_os = "macos") {
        command("sysctl", &["-n", "machdep.cpu.brand_string"])
    } else {
        std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|s| {
            s.lines().find_map(|l| {
                let (k, v) = l.split_once(':')?;
                (k.trim() == "model name" || k.trim() == "CPU part").then(|| v.trim().to_owned())
            })
        })
    };
    #[cfg(target_os = "linux")]
    let host_key = std::fs::read_to_string("/etc/machine-id").ok();
    #[cfg(target_os = "macos")]
    let host_key = command("ioreg", &["-rd1", "-c", "IOPlatformExpertDevice"]).and_then(|s| {
        s.lines()
            .find(|l| l.contains("\"IOPlatformUUID\""))
            .and_then(|l| l.split_once('=').map(|(_, v)| v.trim().to_owned()))
    });
    // `reg query` prints `MachineGuid <type> <guid>`; no admin rights needed.
    #[cfg(windows)]
    let host_key = command(
        "reg",
        &[
            "query",
            r"HKLM\SOFTWARE\Microsoft\Cryptography",
            "/v",
            "MachineGuid",
        ],
    )
    .and_then(|s| {
        s.lines()
            .find_map(|l| match l.split_whitespace().collect::<Vec<_>>()[..] {
                ["MachineGuid", _, guid] => Some(guid.to_owned()),
                _ => None,
            })
    });
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    let host_key: Option<String> = None;
    #[cfg(feature = "native")]
    let llvm = format!("{:?}", inkwell::support::get_llvm_version());
    #[cfg(not(feature = "native"))]
    let llvm = "unavailable";
    json!({"host_class":host_class,"cpu":cpu.unwrap_or_else(||std::env::consts::ARCH.into()),
        "host_id":host_key.filter(|v|!v.trim().is_empty()).map(|key| fingerprint(key.trim().as_bytes())),
        "device":device,"driver":driver,"toolchain":format!("rust-{}:llvm-{}:{}",env!("CARGO_PKG_RUST_VERSION"),llvm,env!("INVAR_BUILD_DIGEST"))})
}
/// Read identity only; never measure or calibrate during ordinary execution.
pub fn live_identity(host_class: &str) -> Value {
    #[allow(unused_mut)]
    let mut device = "unavailable".to_owned();
    #[allow(unused_mut)]
    let mut driver = command("uname", &["-r"]).unwrap_or_else(|| "unknown".into());
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if let Ok(gpu) = invar_metal::MetalDevice::open() {
        device = gpu.name();
        driver = format!(
            "{}:{}:{:?}",
            driver,
            command("sw_vers", &["-buildVersion"]).unwrap_or_else(|| "unknown".into()),
            gpu.capabilities()
        );
    }
    #[cfg(feature = "cuda")]
    if let Ok(gpu) = invar_cuda::cuda::CudaDevice::open() {
        device = gpu.name().to_owned();
        driver = format!(
            "{}:{}:{:?}",
            driver,
            command(
                "nvidia-smi",
                &[
                    "--id=0",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader"
                ]
            )
            .unwrap_or_else(|| "unknown".into()),
            gpu.capabilities()
        );
    }
    host_identity(host_class, &device, &driver)
}
#[cfg(any(
    feature = "native",
    feature = "cuda",
    all(feature = "metal", target_os = "macos"),
    test
))]
fn coefficients(fixed: f64, work: f64, link: Option<f64>, latency: f64, launch: f64) -> Value {
    json!({"fixed_ms":fixed,"per_work_ms":work,"link_bytes_per_ms":link,
        "link_latency_ms":latency,"launch_floor_ms":launch,"page_in_ms_per_byte":null})
}
#[cfg(any(
    feature = "native",
    feature = "cuda",
    all(feature = "metal", target_os = "macos"),
    test
))]
fn fit(times: &[f64], sizes: [usize; 2]) -> Result<(f64, f64), String> {
    if times.len() != 3 || times.iter().any(|n| !n.is_finite() || *n <= 0.0) {
        return Err("probe did not resolve positive times".into());
    }
    // Empty-call overhead is measured, not an extrapolated intercept. Use the
    // slower measured work rate so the line covers both bounded probe sizes.
    let fixed = times[0];
    let work = ((times[1] - fixed) / sizes[0] as f64).max((times[2] - fixed) / sizes[1] as f64);
    if work <= 0.0 {
        return Err(format!("unresolved work cost; raw_ms={times:?}"));
    }
    Ok((fixed, work))
}

#[cfg(any(
    all(test, any(target_os = "linux", target_os = "macos")),
    feature = "native",
    feature = "cuda",
    all(feature = "metal", target_os = "macos")
))]
fn row(
    engine: Engine,
    boundary: &str,
    passes: Vec<Value>,
    samples: Vec<Value>,
    compile_ms: Option<f64>,
    sizes: &[usize],
    points: Vec<Value>,
) -> Result<Value, String> {
    let parsed = passes
        .iter()
        .map(|p| Coefficients::parse(p, engine, boundary))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| "invalid probe coefficients")?;
    let best = Coefficients::best(&parsed);
    Ok(
        json!({"family":MAP_FAMILY,"table_family":"map","numeric":"u32-wrapping","numeric_contract_version":MAP_NUMERIC_CONTRACT,"engine":engine.name(),
        "boundary":boundary,"region":null,"input_pages":"resident-before-each-timing","sizes":sizes,"passes":passes,"samples":samples,"cold_compile_ms":compile_ms,
        "points":points,"best":coefficients(best.fixed,best.work,best.bandwidth,best.latency,best.launch)}),
    )
}
#[cfg(any(
    all(test, any(target_os = "linux", target_os = "macos")),
    feature = "native",
    feature = "cuda",
    all(feature = "metal", target_os = "macos")
))]
fn nanos(now: Instant) -> u64 {
    u64::try_from(now.elapsed().as_nanos()).unwrap_or(u64::MAX)
}
/// The M5-01 protocol for one size: `WARMUPS` untimed calls, then `SAMPLES`
/// timed ones. `call` times only its own region and returns nanoseconds.
/// Returns the first warmup (the cold call) and the timed samples.
#[cfg(any(
    test,
    feature = "native",
    feature = "cuda",
    all(feature = "metal", target_os = "macos")
))]
fn sample<T>(
    start: Instant,
    mut call: impl FnMut() -> Result<(u64, T), String>,
) -> Result<(u64, Vec<(u64, T)>), String> {
    let mut first = None;
    for _ in 0..WARMUPS {
        budget(start)?;
        first.get_or_insert(call()?.0);
    }
    let samples = (0..SAMPLES)
        .map(|_| {
            budget(start)?;
            call()
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((first.ok_or("no warmup")?, samples))
}
#[cfg(any(
    test,
    feature = "native",
    feature = "cuda",
    all(feature = "metal", target_os = "macos")
))]
fn point(elements: Option<usize>, boundary: &str, samples_ns: &[u64]) -> Result<Value, String> {
    let summary =
        summarize(samples_ns).map_err(|_| format!("unresolved samples {samples_ns:?}"))?;
    Ok(json!({"elements":elements,"boundary":boundary,"samples_ns":samples_ns,"summary":summary}))
}
/// The five coefficient passes each use the fastest sample of one sixth.
#[cfg(any(
    test,
    feature = "native",
    feature = "cuda",
    all(feature = "metal", target_os = "macos")
))]
fn group<T: Copy>(samples: &[(u64, T)], pass: usize) -> (u64, T) {
    let width = SAMPLES / 5;
    *samples[pass * width..(pass + 1) * width]
        .iter()
        .min_by_key(|s| s.0)
        .expect("SAMPLES is a nonzero multiple of 5")
}
#[cfg(any(
    test,
    feature = "native",
    feature = "cuda",
    all(feature = "metal", target_os = "macos")
))]
fn budget(start: Instant) -> Result<(), String> {
    if start.elapsed() >= std::time::Duration::from_secs(540) {
        Err("calibration deadline exceeded".into())
    } else {
        Ok(())
    }
}
#[cfg(any(
    feature = "native",
    feature = "cuda",
    all(feature = "metal", target_os = "macos"),
    all(test, any(target_os = "linux", target_os = "macos"))
))]
fn resident(input: &[u32]) -> Result<(), String> {
    match invar_cuda::pages::observe(input) {
        PageState::Known { resident, total } if resident == total => Ok(()),
        _ => Err("resident-input probe could not establish live page residency".into()),
    }
}
#[cfg(feature = "native")]
use cpu::cpu;
#[cfg(not(feature = "native"))]
fn cpu(_start: Instant) -> Result<Value, String> {
    Err("native-feature-unavailable; MIR/Rust folds are not compiled CPU calibration".into())
}

#[cfg(any(test, feature = "cuda", all(feature = "metal", target_os = "macos")))]
fn plan(n: usize, caps: invar_device::Capabilities) -> Result<invar_device::Verified, String> {
    let source = format!(
        "fn main()->i32 {{let xs:Array<u32> = filled_u32({n}u32,7u32);var ys:Array<u32> = filled_u32(len(xs),0u32);for i in 0u32..len(xs) {{ys[i]=wrapping_add(xs[i],1u32);}} return 0;}}"
    );
    let source = source.as_bytes();
    let p = crate::front::parse(source)
        .and_then(|p| crate::check::check(p, false))
        .map_err(|e| e.message)?;
    let p = crate::mir::lower(&p).map_err(|e| e.message)?;
    let report = crate::schedule::regions(source, &p).map_err(|e| format!("{e:?}"))?;
    let region = report
        .regions()
        .iter()
        .find(|r| r.plan_kind() != crate::plan::PlanKind::Serial)
        .ok_or("map probe not admitted")?;
    let mut ir = crate::device_ir::from_region(region, n as u32).map_err(|e| format!("{e:?}"))?;
    if matches!(caps, invar_device::Capabilities::Metal { .. }) {
        crate::device_ir::plan_metal(&mut ir).map_err(|e| format!("{e:?}"))?;
    }
    let origin = invar_device::Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    invar_device::verify(ir, origin, caps).map_err(|e| format!("{e:?}"))
}
/// A conservative envelope of the measured n=1, small and large API calls.
// ponytail: a three-shape envelope can overprice intermediate work; replace it
// with frozen multi-shape fits when M5-01 supplies the dispatch corpus.
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos"), test))]
fn device_coefficients(samples: &[[f64; 3]; 3]) -> Result<Value, String> {
    // Columns: full API wall, GPU kernel, actual transfers. Keep unattributed
    // host management in compute cost instead of silently discarding it.
    if samples.iter().flatten().any(|n| !n.is_finite() || *n < 0.0)
        || samples.iter().any(|s| s[0] < s[1] + s[2])
        || samples[0][1] <= 0.0
        || samples[0][2] <= 0.0
    {
        return Err("unresolved device phases".into());
    }
    let slope = |column: usize| {
        ((samples[1][column] - samples[0][column]) / (SIZES[0] - 1) as f64)
            .max((samples[2][column] - samples[0][column]) / (SIZES[1] - 1) as f64)
            .max(0.0)
    };
    let overhead = samples.map(|s| s[0] - s[1] - s[2]);
    let work = ((overhead[1] - overhead[0]) / (SIZES[0] - 1) as f64)
        .max((overhead[2] - overhead[0]) / (SIZES[1] - 1) as f64)
        .max(0.0);
    let transfer_per_byte = slope(2) / 8.0;
    if transfer_per_byte <= 0.0 {
        return Err("unresolved positive transfer bandwidth".into());
    }
    // Minimal actual launch and two actual one-word copies are upper bounds
    // for their fixed floors. Their tiny work is deliberately counted twice.
    Ok(coefficients(
        overhead[0],
        slope(1) + work,
        Some(1.0 / transfer_per_byte),
        samples[0][2] / 2.0,
        samples[0][1],
    ))
}
#[cfg(any(
    all(test, any(target_os = "linux", target_os = "macos")),
    feature = "cuda",
    all(feature = "metal", target_os = "macos")
))]
fn device_samples(
    start: Instant,
    engine: Engine,
    mut probe: impl FnMut(usize, &[u32]) -> Result<(Vec<u32>, f64, f64), String>,
) -> Result<Value, String> {
    let sizes = [1, SIZES[0], SIZES[1]];
    let cold = Instant::now();
    let mut timed = Vec::new();
    let mut points = Vec::new();
    for n in sizes {
        let input = (0..n)
            .map(|i| (i as u32).wrapping_mul(65537))
            .collect::<Vec<_>>();
        let (first, samples) = sample(start, || {
            resident(&input)?;
            let now = Instant::now();
            let (output, kernel, transfer) = probe(n, &input)?;
            let ns = nanos(now);
            if output.len() != n
                || input
                    .iter()
                    .zip(&output)
                    .any(|(x, y)| *y != x.wrapping_add(1))
            {
                return Err("device output mismatch".into());
            }
            Ok((ns, (kernel, transfer)))
        })?;
        if n == 1 {
            points.push(point(None, FIRST_CALL, &[first])?);
        }
        let ns = samples.iter().map(|s| s.0).collect::<Vec<_>>();
        points.push(point(Some(n), "host-map-v1", &ns)?);
        timed.push(samples);
    }
    let warm_ms = cold.elapsed().as_secs_f64() * 1000.0;
    let mut samples = Vec::new();
    let mut passes = Vec::new();
    let mut host_to_host = Vec::new();
    let mut refusal = None;
    for pass in 0..5 {
        let phases: [[f64; 3]; 3] = std::array::from_fn(|i| {
            let (ns, (kernel, transfer)) = group(&timed[i], pass);
            [ns as f64 / 1e6, kernel, transfer]
        });
        match device_coefficients(&phases) {
            Ok(c) => passes.push(c),
            Err(e) => refusal = Some(e),
        }
        host_to_host.push(json!([phases[1][0], phases[2][0]]));
        samples.push(json!(phases));
    }
    let measurement = if refusal.is_none() {
        let mut row = row(
            engine,
            "host-map-v1",
            passes,
            samples.clone(),
            None,
            &sizes,
            points,
        )?;
        row["coefficient_model"] = json!("measured-n1-envelope");
        Some(row)
    } else {
        None
    };
    Ok(
        json!({"boundary":"host-map-v1","sizes":SIZES,"passes_ms":host_to_host,
        "phase_columns":["api_wall_ms","kernel_ms","transfer_ms"],"phase_samples":samples,
        "warmups":WARMUPS,"samples":SAMPLES,"compile_and_sampling_ms":warm_ms,"measurement":measurement,"refusal":refusal}),
    )
}
/// Every unmeasured (engine, table family) pair, giving the family's own
/// refusal, else the engine's; GNET-04 must not price these.
fn pending(measurements: &[Value], refusals: &[Value]) -> Vec<Value> {
    let mut pending = Vec::new();
    for engine in ENGINES.map(Engine::name) {
        for family in FAMILIES {
            let reason = refusals
                .iter()
                .find(|r| r["engine"] == engine && r["family"] == family)
                .or_else(|| {
                    refusals
                        .iter()
                        .find(|r| r["engine"] == engine && r.get("family").is_none())
                })
                .map_or(json!("unmeasured"), |r| r["reason"].clone());
            if !measurements
                .iter()
                .any(|m| m["engine"] == engine && m["table_family"] == family)
            {
                pending.push(json!({"engine":engine,"family":family,"reason":reason}));
            }
        }
    }
    pending
}
/// One family beyond `map` as forced-device source on `engine`: its native
/// kernel over its own input, with literal shapes. A row counts only when the
/// device ran a region inside the kernel and every output word matched the
/// Rust reference on every call. Only a kernel admission plans serially is
/// `device-plan-ineligible`; every other failure is a failed probe. Times are
/// whole-program host to host (host input build, transfers and launches
/// included), so the row prices nothing beyond them. `run` executes one
/// program on the open device and returns the executed regions' start offsets
/// and whether `main` returned 42.
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
fn device_family(
    start: Instant,
    engine: Engine,
    family: &cpu::Family,
    mut run: impl FnMut(&[u8], &crate::mir::Program) -> Result<(Vec<usize>, bool), String>,
) -> Result<Value, (&'static str, String)> {
    let failed = |detail: String| ("calibration-probe-failed", detail);
    // The host evaluator builds every input inside the timed program, so a
    // smaller large size keeps the 40 calls per size bounded.
    let sizes = [1, SIZES[0], family.large.min(16384)];
    let mut timed = Vec::new();
    let mut points = Vec::new();
    for n in sizes {
        let (source, kernel) = family.device_source(n);
        let program = crate::front::parse(source.as_bytes())
            .and_then(|p| crate::check::check(p, false))
            .map_err(|e| failed(e.message))?;
        let program = crate::mir::lower(&program).map_err(|e| failed(e.message))?;
        let report = crate::schedule::regions(source.as_bytes(), &program)
            .map_err(|e| failed(format!("{e:?}")))?;
        if !report.regions().iter().any(|r| {
            r.plan_kind() != crate::plan::PlanKind::Serial && kernel.contains(&r.span().start)
        }) {
            return Err((
                "device-plan-ineligible",
                format!("admission plans the kernel serially at n={n}"),
            ));
        }
        let (first, samples) = sample(start, || {
            let now = Instant::now();
            let (executed, matched) = run(source.as_bytes(), &program)?;
            let ns = nanos(now);
            if !executed.iter().any(|start| kernel.contains(start)) {
                return Err(format!("no device region ran in the kernel at n={n}"));
            }
            if !matched {
                return Err(format!("output mismatch at n={n}"));
            }
            Ok((ns, ()))
        })
        .map_err(failed)?;
        if n == 1 {
            points.push(point(None, FIRST_CALL, &[first]).map_err(failed)?);
        }
        let ns = samples.iter().map(|s| s.0).collect::<Vec<_>>();
        points.push(point(Some(n), SOURCE_DEVICE_BOUNDARY, &ns).map_err(failed)?);
        timed.push(samples);
    }
    let mut passes = Vec::new();
    let mut per_pass = Vec::new();
    for pass in 0..5 {
        let times = timed
            .iter()
            .map(|s| group(s, pass).0 as f64 / 1e6)
            .collect::<Vec<_>>();
        let (fixed, work) = fit(&times, [sizes[1] - 1, sizes[2] - 1]).map_err(failed)?;
        passes.push(coefficients(fixed, work, None, 0.0, 0.0));
        per_pass.push(json!(times));
    }
    let mut row = row(
        engine,
        SOURCE_DEVICE_BOUNDARY,
        passes,
        per_pass,
        None,
        &sizes,
        points,
    )
    .map_err(failed)?;
    row["family"] = json!(family.id());
    row["table_family"] = json!(family.table);
    row["coefficient_model"] = json!("whole-program-host-to-host");
    Ok(row)
}
/// The executed regions' start offsets, and whether `main` returned 42.
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
fn executed_42(
    executed: &[crate::schedule::Executed],
    observation: &crate::eval::Observation,
) -> (Vec<usize>, bool) {
    (
        executed.iter().map(|s| s.span.start).collect(),
        matches!(observation.result, Ok(crate::eval::Value::I32(42))),
    )
}
/// Generate provisional data only. A profile with missing coefficients is
/// valid and safely retains CPU; its refusal is part of the exported artifact.
pub fn calibrate(host_class: &str, source_revision: &str) -> Result<Value, Error> {
    if host_class.is_empty()
        || host_class.len() > 128
        || !host_class
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err(Error::Invalid);
    }
    let start = Instant::now();
    let before = observations();
    #[allow(unused_mut)]
    let mut measurements = Vec::new();
    let mut refusals = Vec::new();
    #[allow(unused_mut)]
    let mut probes: Vec<Value> = Vec::new();
    match cpu(start) {
        Ok(row) => measurements.push(row),
        Err(detail) => refusals
            .push(json!({"engine":"cpu","reason":"calibration-probe-failed","detail":detail})),
    }
    #[cfg(feature = "native")]
    for family in &cpu::FAMILIES_BEYOND_MAP {
        match cpu::probe(start, family) {
            Ok(row) => measurements.push(row),
            Err(detail) => refusals
                .push(json!({"engine":"cpu","reason":"calibration-probe-failed",
                "family":family.table,"detail":detail})),
        }
    }
    #[allow(unused_mut)]
    let mut device = "unavailable".to_owned();
    #[allow(unused_mut)]
    let mut driver = command("uname", &["-r"]).unwrap_or_else(|| "unknown".into());
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let result = (|| {
            let mut gpu =
                invar_metal::MetalDevice::open_runtime_source().map_err(|e| format!("{e:?}"))?;
            device = gpu.name();
            let os_build = command("sw_vers", &["-buildVersion"])
                .ok_or("Metal driver/OS build identity unavailable")?;
            driver = format!("{}:{}:{:?}", driver, os_build, gpu.capabilities());
            let plans = [1, SIZES[0], SIZES[1]]
                .map(|n| plan(n, gpu.capabilities()))
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;
            let map = device_samples(start, Engine::Metal, |n, input| {
                let index = [1, SIZES[0], SIZES[1]]
                    .iter()
                    .position(|size| *size == n)
                    .ok_or("unmeasured shape")?;
                gpu.run_map(&plans[index], input)
                    .map(|r| (r.values, r.kernel_ms, r.transfer_ms))
                    .map_err(|e| format!("{e:?}"))
            })?;
            let families = cpu::FAMILIES_BEYOND_MAP
                .iter()
                .map(|f| {
                    let probe = device_family(start, Engine::Metal, f, |source, program| {
                        let run = crate::schedule::metal::run_source_with_device(
                            source,
                            program,
                            cpu::FUEL,
                            &mut gpu,
                        )
                        .map_err(|e| format!("{}: {}", e.code, e.message))?;
                        Ok(executed_42(&run.executed, &run.observation))
                    });
                    (f.table, probe)
                })
                .collect::<Vec<_>>();
            Ok::<_, String>((map, families))
        })();
        match result {
            Ok((v, families)) => {
                if let Some(row) = v["measurement"].as_object() {
                    measurements.push(Value::Object(row.clone()));
                } else {
                    refusals.push(json!({"engine":"metal","reason":"calibration-probe-failed","detail":v["refusal"]}));
                }
                probes.push(json!({"engine":"metal","probe":v}));
                for (family, probe) in families {
                    match probe {
                        Ok(row) => measurements.push(row),
                        Err((reason, detail)) => refusals.push(
                            json!({"engine":"metal","reason":reason,"family":family,"detail":detail}),
                        ),
                    }
                }
            }
            Err(detail) => refusals.push(
                json!({"engine":"metal","reason":"calibration-probe-failed","detail":detail}),
            ),
        }
    }
    #[cfg(feature = "cuda")]
    {
        let result = (|| {
            let mut gpu = invar_cuda::cuda::CudaDevice::open().map_err(|e| format!("{e:?}"))?;
            device = gpu.name().to_owned();
            let driver_version = command(
                "nvidia-smi",
                &[
                    "--id=0",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
            )
            .ok_or("CUDA driver identity unavailable")?;
            driver = format!("{}:{}:{:?}", driver, driver_version, gpu.capabilities());
            let plans = [1, SIZES[0], SIZES[1]]
                .map(|n| plan(n, gpu.capabilities()))
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;
            let compiler = std::env::current_exe()
                .map_err(|e| e.to_string())?
                .with_file_name(format!("invar-nvrtc{}", std::env::consts::EXE_SUFFIX));
            let map = device_samples(start, Engine::Cuda, |n, input| {
                let index = [1, SIZES[0], SIZES[1]]
                    .iter()
                    .position(|size| *size == n)
                    .ok_or("unmeasured shape")?;
                gpu.run_map(&plans[index], input, &compiler)
                    .map(|r| {
                        (
                            r.values,
                            f64::from(r.kernel_ms),
                            f64::from(r.h2d_ms) + f64::from(r.d2h_ms),
                        )
                    })
                    .map_err(|e| format!("{e:?}"))
            })?;
            let families = cpu::FAMILIES_BEYOND_MAP
                .iter()
                .map(|f| {
                    let probe = device_family(start, Engine::Cuda, f, |source, program| {
                        let run = crate::schedule::cuda::run_source_with_device(
                            source,
                            program,
                            cpu::FUEL,
                            &mut gpu,
                            &compiler,
                        )
                        .map_err(|e| format!("{}: {}", e.code, e.message))?;
                        Ok(executed_42(&run.executed, &run.observation))
                    });
                    (f.table, probe)
                })
                .collect::<Vec<_>>();
            Ok::<_, String>((map, families))
        })();
        match result {
            Ok((v, families)) => {
                if let Some(row) = v["measurement"].as_object() {
                    measurements.push(Value::Object(row.clone()));
                } else {
                    refusals.push(json!({"engine":"cuda","reason":"calibration-probe-failed","detail":v["refusal"]}));
                }
                probes.push(json!({"engine":"cuda","probe":v}));
                for (family, probe) in families {
                    match probe {
                        Ok(row) => measurements.push(row),
                        Err((reason, detail)) => refusals.push(
                            json!({"engine":"cuda","reason":reason,"family":family,"detail":detail}),
                        ),
                    }
                }
            }
            Err(detail) => refusals
                .push(json!({"engine":"cuda","reason":"calibration-probe-failed","detail":detail})),
        }
    }
    #[cfg(not(feature = "cuda"))]
    refusals.push(json!({"engine":"cuda","reason":"cuda-feature-unavailable"}));
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    refusals.push(json!({"engine":"metal","reason":"metal-feature-or-platform-unavailable"}));
    // Other laws/intrinsics never inherit these probes' timings.
    refusals.push(json!({"engine":"all-other-families","reason":"unmeasured"}));
    let coverage = json!({"families":FAMILIES,"pending":pending(&measurements, &refusals)});
    let result = json!({"schema_version":VERSION,"probe_version":PROBE,"units":"ms,bytes,work",
        "statistics":statistics(),"coverage":coverage,
        "identity":host_identity(host_class,&device,&driver),"provisional":true,"source_revision":source_revision,"source_build_digest":env!("INVAR_BUILD_DIGEST"),
        "observations":{"before":before,"after":observations()},"measurements":measurements,"refusals":refusals,"probes":probes,
        "elapsed_ms":start.elapsed().as_secs_f64()*1000.0});
    Profile::parse(
        &serde_json::to_vec(&result).map_err(|_| Error::Invalid)?,
        &result["identity"],
    )?;
    Ok(result)
}
/// Publish atomically without replacing any existing path (including symlinks).
/// The temporary file is in the same directory; failures remove only that file.
pub fn publish(path: &Path, document: &Value) -> Result<(), Error> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let bytes = serde_json::to_vec_pretty(document).map_err(|_| Error::Invalid)?;
    Profile::parse(&bytes, &document["identity"])?;
    let mut temp = None;
    for n in 0..32 {
        let p = parent.join(format!(".invar-calibration-{}-{n}.tmp", std::process::id()));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&p)
        {
            Ok(file) => {
                temp = Some((p, file));
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(_) => return Err(Error::Io),
        }
    }
    let (temp, mut file) = temp.ok_or(Error::Io)?;
    let result = (|| {
        file.write_all(&bytes).map_err(|_| Error::Io)?;
        file.sync_all().map_err(|_| Error::Io)?;
        std::fs::hard_link(&temp, path).map_err(|_| Error::Io)?;
        Ok(())
    })();
    let _ = std::fs::remove_file(&temp);
    result
}

#[cfg(test)]
mod tests {
    #[test]
    fn map_probe_admits_both_bounded_shapes() {
        for n in super::SIZES {
            let p = super::plan(n, invar_device::Capabilities::cuda_12_9_sm89()).unwrap();
            assert_eq!(p.ir().length, n as u32);
            assert_eq!(p.ir().ops, [invar_device::Op::WrappingAdd(1)]);
        }
    }
    #[test]
    fn protocol_times_thirty_samples_after_ten_untimed_warmups() {
        let mut calls = 0u64;
        let (first, samples) = super::sample(std::time::Instant::now(), || {
            calls += 1;
            Ok((calls, ()))
        })
        .unwrap();
        assert_eq!(calls, 40);
        assert_eq!(first, 1);
        let ns = samples.iter().map(|s| s.0).collect::<Vec<_>>();
        assert_eq!(ns, (11..=40).collect::<Vec<_>>());
        // Each coefficient pass takes the fastest of its own sixth.
        assert_eq!(
            (0..5)
                .map(|p| super::group(&samples, p).0)
                .collect::<Vec<_>>(),
            [11, 17, 23, 29, 35]
        );
        let p = super::point(Some(4), "b", &ns).unwrap();
        assert_eq!(p["summary"], super::summarize(&ns).unwrap());
        assert!(super::point(Some(4), "b", &ns[1..]).is_err());
        let failed = super::sample(std::time::Instant::now(), || {
            Err::<(u64, ()), _>("x".into())
        });
        assert_eq!(failed.unwrap_err(), "x");
    }
    #[test]
    fn measured_empty_call_and_bounded_work_fit() {
        let times = [0.5, 1.0, 2.0];
        let (fixed, work) = super::fit(&times, super::SIZES).unwrap();
        assert_eq!(fixed, 0.5);
        for (n, ms) in super::SIZES.into_iter().zip([1.0, 2.0]) {
            assert!(fixed + work * n as f64 >= ms);
        }
        for bad in [
            vec![],
            vec![1.0, 0.0, 2.0],
            vec![1.0, 0.5, 0.5],
            vec![f64::INFINITY, 1.0, 2.0],
        ] {
            assert!(super::fit(&bad, super::SIZES).is_err());
        }
    }
}
#[cfg(test)]
mod device_cost_tests {
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn changed_device_words_are_refused_before_a_profile() {
        let error =
            super::device_samples(std::time::Instant::now(), super::Engine::Cuda, |n, _| {
                Ok((vec![99; n], 0.0, 0.0))
            })
            .unwrap_err();
        assert_eq!(error, "device output mismatch");
        let result = super::device_samples(
            std::time::Instant::now(),
            super::Engine::Cuda,
            |_, input| Ok((input.iter().map(|x| x.wrapping_add(1)).collect(), 0.0, 0.0)),
        )
        .unwrap();
        assert!(result["measurement"].is_null());
        assert_eq!(result["refusal"], "unresolved device phases");
        // Synthetic phase clocks test the production driver without a GPU.
        let result = super::device_samples(
            std::time::Instant::now(),
            super::Engine::Cuda,
            |n, input| {
                std::thread::sleep(std::time::Duration::from_millis(1));
                Ok((
                    input.iter().map(|x| x.wrapping_add(1)).collect(),
                    0.0000001,
                    n as f64 * 0.0000001,
                ))
            },
        )
        .unwrap();
        assert!(result["measurement"].is_object());
        assert!(result["refusal"].is_null());
    }
    #[test]
    fn measured_device_envelope_keeps_transfers_launches_and_host_overhead() {
        let samples = [[1.0, 0.1, 0.2], [1.6, 0.2, 0.6], [4.0, 0.5, 2.0]];
        let v = super::device_coefficients(&samples).unwrap();
        let get = |key| v[key].as_f64().unwrap();
        for (n, s) in [1, super::SIZES[0], super::SIZES[1]]
            .into_iter()
            .zip(samples)
        {
            let total = get("fixed_ms")
                + get("per_work_ms") * n as f64
                + 8.0 * n as f64 / get("link_bytes_per_ms")
                + 2.0 * get("link_latency_ms")
                + get("launch_floor_ms");
            assert!(total >= s[0], "{n}: {total} < {}", s[0]);
        }
        assert_eq!(get("launch_floor_ms"), 0.1);
        assert_eq!(get("link_latency_ms"), 0.1);
        assert!((get("fixed_ms") - 0.7).abs() < 1e-12);
        for bad in [
            [[0.1, 1.0, 1.0]; 3],
            [[1.0, 0.0, 0.0]; 3],
            [[1.0, 0.1, 0.2]; 3],
            [[f64::NAN, 0.1, 0.2]; 3],
        ] {
            assert!(super::device_coefficients(&bad).is_err());
        }
    }
}
