//! Explicit ordinary-source private ABI5 route. Forced backends never fall back.
use invar::{
    eval::Value,
    front::Type,
    general_net::{
        compiled::{DeviceConfig, Status, Target},
        device_host::{self, Backend, Plan, Progress},
    },
};
use std::{
    ffi::OsString,
    io::Read,
    time::{Duration, Instant},
};

pub fn command(args: &[OsString]) -> i32 {
    let result = execute(args);
    match result {
        Ok((report, success)) => {
            println!("{report}");
            if success { 0 } else { 1 }
        }
        Err(error) => {
            println!(
                "{}",
                serde_json::json!({"schema":"invar-private-gnet-device-v1","error":error})
            );
            1
        }
    }
}
fn execute(args: &[OsString]) -> Result<(serde_json::Value, bool), String> {
    let artifact = match args {
        [_, _, _, _, _] => None,
        [_, _, _, _, _, flag, file] if flag == "--artifact" => Some(file),
        _ => {
            return Err(
                "expected PATH ENTRY BACKEND ARGS_JSON CONFIG_JSON [--artifact FILE]".into(),
            );
        }
    };
    let entry = args[1].to_str().ok_or("invalid entry")?;
    let backend = args[2].to_str().ok_or("invalid backend")?;
    let target = match backend {
        "cpu-model-cuda" | "cuda" => Target::Cuda,
        "cpu-model-metal" | "metal" => Target::Metal,
        _ => return Err("backend must be cpu-model-cuda, cpu-model-metal, cuda or metal".into()),
    };
    if artifact.is_some() && !matches!(backend, "cuda" | "metal") {
        return Err("--artifact applies to the cuda and metal backends only".into());
    }
    let source = read_capped(&args[0], 1024 * 1024).map_err(|e| format!("source:{e}"))?;
    let plan = match Plan::from_source(&source, entry, target, device_host::Limits::certified()) {
        Ok(plan) => plan,
        // Forced device mode: a typed refusal is the whole outcome, never a
        // substituted engine.
        Err(e) => {
            return Ok((
                serde_json::json!({"schema":"invar-private-gnet-device-v1","backend":backend,
                    "error":format!("source admission:{e:?}"),
                    "placement":{"mode":"forced-device","executed":false,
                        "refusal":device_host::refusal_class(&e),"fallback":"none"}}),
                false,
            ));
        }
    };
    let inputs: serde_json::Value =
        serde_json::from_str(args[3].to_str().ok_or("invalid argument JSON")?)
            .map_err(|e| e.to_string())?;
    let inputs = inputs.as_array().ok_or("arguments must be an array")?;
    if inputs.len() != plan.artifact().params().len() {
        return Err("argument count mismatch".into());
    }
    let values: Vec<Value> = plan
        .artifact()
        .params()
        .iter()
        .zip(inputs)
        .map(|(ty, value)| argument(plan.artifact(), ty, value))
        .collect::<Result<_, _>>()?;
    let cfg: serde_json::Value =
        serde_json::from_str(args[4].to_str().ok_or("invalid configuration JSON")?)
            .map_err(|e| e.to_string())?;
    let cfg = cfg.as_object().ok_or("configuration must be an object")?;
    if cfg.len() != 6 {
        return Err("configuration requires exactly fuel, transitions, trace_events, quantum, rounds and capacity".into());
    }
    let word = |name| {
        cfg.get(name)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| format!("invalid configuration {name}"))
    };
    let config = DeviceConfig {
        fuel: word("fuel")?,
        transitions: word("transitions")?,
        trace_events: usize::try_from(word("trace_events")?).map_err(|_| "trace capacity")?,
        quantum: word("quantum")?,
        rounds: word("rounds")?,
        capacity: usize::try_from(word("capacity")?).map_err(|_| "private capacity")?,
    };
    match backend {
        "cpu-model-cuda" | "cpu-model-metal" => drive(
            &plan,
            &mut device_host::cpu::CpuModel::default(),
            &values,
            config,
            backend,
            serde_json::Value::Null,
        ),
        "cuda" => cuda(&plan, &values, config, artifact),
        "metal" => metal(&plan, &values, config, artifact),
        _ => Err("invalid backend".into()),
    }
}
fn drive<B: Backend>(
    plan: &Plan,
    backend: &mut B,
    values: &[Value],
    config: DeviceConfig,
    label: &str,
    image: serde_json::Value,
) -> Result<(serde_json::Value, bool), String> {
    let mut session = plan
        .prepare(backend, values, config)
        .map_err(|e| format!("prepare:{e:?}"))?;
    loop {
        session.submit().map_err(|e| format!("submit:{e:?}"))?;
        let begun = Instant::now();
        let progress = loop {
            match session.poll().map_err(|e| format!("completion:{e:?}"))? {
                Progress::Pending if begun.elapsed() < Duration::from_secs(10) => {
                    std::thread::sleep(Duration::from_millis(1))
                }
                Progress::Pending => {
                    session.cancel();
                    return Err(
                        "completion deadline exceeded; retained work cannot be replayed".into(),
                    );
                }
                progress => break progress,
            }
        };
        if let Progress::Complete(observation) = progress {
            let success = observation.status == Status::Success;
            let result = observation
                .result
                .as_ref()
                .map(value_json)
                .unwrap_or(serde_json::Value::Null);
            let checkpoint = session.checkpoint().ok_or("missing completed checkpoint")?;
            let report = serde_json::json!({"schema":"invar-private-gnet-device-v1","backend":label,"entry":plan.entry(),"artifact":plan.compile_receipt(),"image":image,"execution":checkpoint.manifest(),"checkpoint":checkpoint.id(),"status":observation.status as u32,"result":result,"span":observation.span.map(|span| serde_json::json!({"start":span.start,"end":span.end})),"source_steps":observation.steps,"transitions":observation.transitions,"max_frames":observation.max_frames,"events":observation.events,"cleanup":observation.cleanup,"private_state":observation.private_state,"placement":{"mode":"forced-device","executed":true,"target":label,"fallback":"none"},"submitted":session.submitted(),"completed":session.completed()});
            let numerics = session.numeric_receipt().cloned();
            drop(session.reclaim().map_err(|e| format!("reclaim:{e:?}"))?);
            let mut report = report;
            if let Some(numerics) = numerics {
                report["numerics"] = numerics;
            }
            if plan.artifact().entry_layout().is_some() {
                report["arguments_after"] =
                    serde_json::Value::Array(values.iter().map(value_json).collect());
            }
            return Ok((report, success));
        }
    }
}
/// Production CUDA runs a compiled-ahead image through the Driver API alone;
/// this process never loads or spawns NVRTC. Without an image it refuses.
#[cfg(feature = "cuda")]
fn cuda(
    plan: &Plan,
    values: &[Value],
    config: DeviceConfig,
    artifact: Option<&OsString>,
) -> Result<(serde_json::Value, bool), String> {
    use device_host::cuda_nvrtc::{CudaNvrtc, RUNTIME_NVRTC_ENV};
    let Some(artifact) = artifact else {
        if std::env::var_os(RUNTIME_NVRTC_ENV).is_none_or(|v| v != "1") {
            return Err(format!(
                "forced CUDA runs a compiled-ahead image: pass --artifact FILE from `invar build --target cuda` ({RUNTIME_NVRTC_ENV}=1 compiles source for debugging)"
            ));
        }
        let compiler = std::env::current_exe()
            .map_err(|e| e.to_string())?
            .with_file_name(format!("invar-nvrtc{}", std::env::consts::EXE_SUFFIX));
        let mut backend = CudaNvrtc::open(&compiler).map_err(|e| format!("forced CUDA:{e:?}"))?;
        return drive(
            plan,
            &mut backend,
            values,
            config,
            "cuda",
            serde_json::Value::Null,
        );
    };
    let bytes = read_capped(artifact, device_host::cuda_aot::MAX_CONTAINER_BYTES)
        .map_err(|e| format!("artifact:{e}"))?;
    let mut backend = device_host::cuda::Cuda::open().map_err(|e| format!("forced CUDA:{e:?}"))?;
    let image = match backend.load(plan, &bytes) {
        Ok(image) => image,
        Err(e) => {
            return Ok((
                serde_json::json!({"schema":"invar-private-gnet-device-v1","backend":"cuda",
                    "error":format!("image admission:{e:?}"),
                    "placement":{"mode":"forced-device","executed":false,
                        "refusal":device_host::refusal_class(&e),"fallback":"none"}}),
                false,
            ));
        }
    };
    drive(plan, &mut backend, values, config, "cuda", image)
}
#[cfg(not(feature = "cuda"))]
fn cuda(
    _: &Plan,
    _: &[Value],
    _: DeviceConfig,
    _: Option<&OsString>,
) -> Result<(serde_json::Value, bool), String> {
    Err("forced CUDA requires the cuda feature and a compatible device".into())
}
/// `invar build PATH --target cuda --entry ENTRY --sm SM --output FILE`:
/// NVRTC runs here, at build time, in the bounded helper subprocess.
pub fn build(path: &OsString, entry: &OsString, sm: &OsString, output: &OsString) -> i32 {
    match build_image(path, entry, sm, output) {
        Ok(receipt) => {
            println!(
                "{}",
                serde_json::json!({"schema":"invar-gnet-cuda-build-v1","image":receipt})
            );
            0
        }
        Err(error) => {
            println!(
                "{}",
                serde_json::json!({"schema":"invar-gnet-cuda-build-v1","error":error})
            );
            1
        }
    }
}
#[cfg(feature = "cuda")]
fn build_image(
    path: &OsString,
    entry: &OsString,
    sm: &OsString,
    output: &OsString,
) -> Result<serde_json::Value, String> {
    let entry = entry.to_str().ok_or("invalid entry")?;
    let sm: u32 = sm
        .to_str()
        .and_then(|s| s.parse().ok())
        .filter(|sm| (50..=255).contains(sm))
        .ok_or("--sm must be a CUDA architecture such as 120")?;
    let source = read_capped(path, 1024 * 1024).map_err(|e| format!("source:{e}"))?;
    let plan = Plan::from_source(
        &source,
        entry,
        Target::Cuda,
        device_host::Limits::certified(),
    )
    .map_err(|e| format!("source admission:{e:?}"))?;
    let compiler = std::env::current_exe()
        .map_err(|e| e.to_string())?
        .with_file_name(format!("invar-nvrtc{}", std::env::consts::EXE_SUFFIX));
    let bytes = device_host::cuda_aot::build(&plan, &compiler, sm)?;
    let receipt = device_host::cuda_aot::decode(&bytes)
        .map_err(|e| format!("built image:{e:?}"))?
        .receipt();
    std::fs::write(output, &bytes).map_err(|e| format!("output:{e}"))?;
    Ok(receipt)
}
#[cfg(not(feature = "cuda"))]
fn build_image(
    _: &OsString,
    _: &OsString,
    _: &OsString,
    _: &OsString,
) -> Result<serde_json::Value, String> {
    Err("CUDA build requires the cuda feature and NVRTC".into())
}
fn read_capped(path: &OsString, cap: usize) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .map_err(|e| e.to_string())?
        .take(cap as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > cap {
        return Err("exceeds its admission budget".into());
    }
    Ok(bytes)
}
/// Production Metal runs a compiled-ahead library; this process never compiles
/// MSL. Without `--artifact` it refuses unless the developer flag is set.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal(
    plan: &Plan,
    values: &[Value],
    config: DeviceConfig,
    artifact: Option<&OsString>,
) -> Result<(serde_json::Value, bool), String> {
    use device_host::metal::Metal;
    use invar::schedule::metal::{Kernels, RUNTIME_SOURCE_ENV};
    let Some(artifact) = artifact else {
        if !matches!(Kernels::developer(), Kernels::RuntimeSource) {
            return Err(format!(
                "forced Metal runs a compiled-ahead library: pass --artifact FILE from `invar build PATH --target metal --entry ENTRY --output FILE` ({RUNTIME_SOURCE_ENV}=1 compiles source for debugging)"
            ));
        }
        let mut backend =
            Metal::open_runtime_source().map_err(|e| format!("forced Metal:{e:?}"))?;
        let (mut report, success) = drive(
            plan,
            &mut backend,
            values,
            config,
            "metal",
            serde_json::Value::Null,
        )?;
        report["metal_archive"] = archive_receipt(backend.cache_report().archive);
        return Ok((report, success));
    };
    let mut backend = Metal::open().map_err(|e| format!("forced Metal:{e:?}"))?;
    let loaded = invar::metal_aot::build::read(std::path::Path::new(artifact))
        .and_then(|image| backend.load(plan, &image));
    let image = match loaded {
        Ok(image) => image,
        Err(e) => {
            return Ok((
                serde_json::json!({"schema":"invar-private-gnet-device-v1","backend":"metal",
                    "error":format!("library admission:{e:?}"),
                    "placement":{"mode":"forced-device","executed":false,
                        "refusal":"library","fallback":"none"}}),
                false,
            ));
        }
    };
    let (mut report, success) = drive(plan, &mut backend, values, config, "metal", image)?;
    report["metal_archive"] = archive_receipt(backend.cache_report().archive);
    Ok((report, success))
}
#[cfg(all(feature = "metal", target_os = "macos"))]
fn archive_receipt(report: invar_metal::ArchiveReport) -> serde_json::Value {
    serde_json::json!({"state":format!("{:?}",report.last),"key":report.key.iter().map(|b|format!("{b:02x}")).collect::<String>(),"hits":report.hits,"misses":report.misses,"rebuilds":report.rebuilds,"writes":report.writes,"write_refusal":report.write_refusal.map(|r|format!("{r:?}")),"bytes":report.bytes,"pipeline_creation_ns":report.pipeline_creation_ns})
}
#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal(
    _: &Plan,
    _: &[Value],
    _: DeviceConfig,
    _: Option<&OsString>,
) -> Result<(serde_json::Value, bool), String> {
    Err("forced Metal requires the metal feature and an Apple Silicon Mac".into())
}

fn argument(
    artifact: &invar::general_net::compiled::Artifact,
    ty: &Type,
    v: &serde_json::Value,
) -> Result<Value, String> {
    use std::{cell::RefCell, rc::Rc};
    let mismatch = || "typed argument/layout mismatch".to_string();
    Ok(match ty {
        Type::I32 => Value::I32(
            v.as_i64()
                .and_then(|n| i32::try_from(n).ok())
                .ok_or_else(mismatch)?,
        ),
        Type::U32 => Value::U32(
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(mismatch)?,
        ),
        Type::F32 => {
            let o = v
                .as_object()
                .filter(|o| o.len() == 1)
                .ok_or_else(mismatch)?;
            Value::F32(invar::numeric_contract::F32Bits::from_bits(
                o.get("bits")
                    .and_then(|v| v.as_u64())
                    .and_then(|v| u32::try_from(v).ok())
                    .ok_or_else(mismatch)?,
            ))
        }
        Type::Bool => Value::Bool(v.as_bool().ok_or_else(mismatch)?),
        Type::Unit if v.is_null() => Value::Unit,
        Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 => {
            let values = v
                .as_array()
                .filter(|v| v.len() <= invar::array_limits::MAX_ARRAY_LEN)
                .ok_or_else(mismatch)?;
            match ty {
                Type::ArrayI32 => Value::ArrayI32(Rc::new(RefCell::new(
                    values
                        .iter()
                        .map(|v| {
                            v.as_i64()
                                .and_then(|v| i32::try_from(v).ok())
                                .ok_or_else(mismatch)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ))),
                Type::ArrayU32 => Value::ArrayU32(Rc::new(RefCell::new(
                    values
                        .iter()
                        .map(|v| {
                            v.as_u64()
                                .and_then(|v| u32::try_from(v).ok())
                                .ok_or_else(mismatch)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ))),
                _ => Value::ArrayF32(Rc::new(RefCell::new(
                    values
                        .iter()
                        .map(|v| match argument(artifact, &Type::F32, v)? {
                            Value::F32(v) => Ok(v),
                            _ => Err(mismatch()),
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ))),
            }
        }
        Type::Named(name) => {
            let kind = artifact
                .aggregate_layout()
                .and_then(|a| a.kinds().iter().find(|k| k.name() == name))
                .ok_or_else(mismatch)?;
            let invar::general_net::compiled::AggregateForm::Record(fields) = kind.form() else {
                return Err(mismatch());
            };
            let values = v
                .as_array()
                .filter(|v| v.len() == fields.len())
                .ok_or_else(mismatch)?;
            Value::Record(Rc::new(RefCell::new(
                fields
                    .iter()
                    .zip(values)
                    .map(|(t, v)| argument(artifact, t, v))
                    .collect::<Result<Vec<_>, _>>()?,
            )))
        }
        _ => return Err(mismatch()),
    })
}
fn value_json(value: &Value) -> serde_json::Value {
    match value {
        Value::I32(v) => serde_json::json!(v),
        Value::U32(v) => serde_json::json!(v),
        Value::Bool(v) => serde_json::json!(v),
        Value::F32(v) => serde_json::json!({"bits":v.to_bits()}),
        Value::ArrayI32(v) => serde_json::json!(*v.borrow()),
        Value::ArrayU32(v) => serde_json::json!(*v.borrow()),
        Value::ArrayF32(v) => serde_json::json!(
            v.borrow()
                .iter()
                .map(|v| serde_json::json!({"bits":v.to_bits()}))
                .collect::<Vec<_>>()
        ),
        Value::Record(v) => serde_json::Value::Array(v.borrow().iter().map(value_json).collect()),
        _ => serde_json::Value::Null,
    }
}
