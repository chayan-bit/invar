//! Bounded M5-02 device qualification. Reuse the reviewed, read-only M4
//! corpora so an adapter cannot quietly qualify a different family or oracle.
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
use super::agree;
use super::{Failure, digest, semantic};
use invar::eval;
use invar_device::{Capabilities, indexed, segmented, stencil::Boundary};
use serde_json::{Value, json};
use std::path::Path;

#[allow(dead_code)]
#[path = "../src/contraction_source.rs"]
mod contraction_source;
#[path = "../tests/issue_m4_01_device/ops.rs"]
mod floating;
#[allow(dead_code)]
#[path = "../tests/support/m402.rs"]
mod grid;
#[allow(dead_code)]
#[path = "../tests/support/m4_05_device.rs"]
mod indices;
#[path = "../tests/issue_m4_03_device/corpus.rs"]
mod matrix;
#[path = "../tests/support/pair_reduction.rs"]
mod pair;
#[allow(dead_code)]
#[path = "../tests/support/m4_04_cases.rs"]
mod segments;
#[path = "../tests/issue_m4_07_device/corpus.rs"]
mod summaries;

pub const FAMILIES: [&str; 10] = [
    "map",
    "reduction",
    "scan",
    "segmented",
    "compaction",
    "transpose",
    "stencil",
    "matrix",
    "gather_scatter",
    "frontier",
];
pub const CUDA: Capabilities = Capabilities::Cuda {
    major: 12,
    minor: 9,
    sm: 120,
};
pub const METAL: Capabilities = Capabilities::Metal {
    apple_family: 9,
    msl_major: 3,
    msl_minor: 1,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Cuda,
    Metal,
}
impl Target {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }
    fn caps(self) -> Capabilities {
        match self {
            Self::Cuda => CUDA,
            Self::Metal => METAL,
        }
    }
}

/// Admission checks are exact for this lane's pinned qualification targets.
/// This is not a new general device eligibility or placement policy.
/// The exact device name each target is pinned to (ruling Q2).
pub fn pinned_device(target: Target) -> &'static str {
    match target {
        Target::Cuda => "NVIDIA GeForce RTX 5070 Ti",
        Target::Metal => "Apple M4",
    }
}

pub fn preflight(target: Target, name: &str, caps: Capabilities) -> Result<(), Failure> {
    let expected_name = pinned_device(target);
    if caps != target.caps() || name != expected_name {
        return Err(Failure::new(
            "M5_PREFLIGHT_MISMATCH",
            format!(
                "{name} {caps:?}; expected {expected_name} {:?}",
                target.caps()
            ),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Refusal {
    Subnormal,
    NoRegion,
    Limit,
    Oom,
    Launch,
    Pending,
    Toolchain,
    Device,
}
impl Refusal {
    fn code(self) -> &'static str {
        match self {
            Self::Subnormal => "M5_STRICT_F32_CPU_FALLBACK",
            Self::Limit => "M5_DEVICE_INPUT_LIMIT",
            Self::NoRegion => "M5_SOURCE_DEVICE_INELIGIBLE",
            Self::Oom => "M5_DEVICE_OOM",
            Self::Launch => "M5_DEVICE_LAUNCH",
            Self::Pending => "M5_DEVICE_COMPLETION_UNKNOWN",
            Self::Toolchain => "M5_DEVICE_TOOLCHAIN",
            Self::Device => "M5_DEVICE_REFUSAL",
        }
    }
}
#[derive(Debug)]
struct DeviceError {
    reason: Refusal,
    detail: String,
}
impl From<DeviceError> for Failure {
    fn from(error: DeviceError) -> Self {
        Self::new(error.reason.code(), error.detail)
    }
}
#[cfg(feature = "cuda")]
impl From<invar_cuda::Error> for DeviceError {
    fn from(e: invar_cuda::Error) -> Self {
        use invar_cuda::Error;
        let reason = match &e {
            Error::Limit => Refusal::Limit,
            Error::OutOfMemory => Refusal::Oom,
            Error::SubnormalFallback => Refusal::Subnormal,
            Error::Timeout => Refusal::Pending,
            Error::Compile(_) => Refusal::Toolchain,
            _ => Refusal::Device,
        };
        Self {
            reason,
            detail: format!("{e:?}"),
        }
    }
}
#[cfg(all(feature = "metal", target_os = "macos"))]
impl From<invar_metal::Error> for DeviceError {
    fn from(e: invar_metal::Error) -> Self {
        use invar_metal::Error;
        let reason = match &e {
            Error::SubnormalFallback => Refusal::Subnormal,
            Error::Limit => Refusal::Limit,
            Error::OutOfMemory => Refusal::Oom,
            Error::Pending => Refusal::Pending,
            Error::Toolchain(_) => Refusal::Toolchain,
            _ => Refusal::Device,
        };
        Self {
            reason,
            detail: format!("{e:?}"),
        }
    }
}

enum Work {
    Source(Vec<u8>, u64),
    F32Source(Vec<u8>, u64),
    Transpose(invar_device::transpose::Verified, Vec<u32>),
    Stencil(invar_device::stencil_u32::Verified, Vec<u32>),
    Segmented(segmented::Verified, Vec<u32>, Vec<u32>),
    Indexed(indexed::Bound),
    Summary(invar_device::summary::Verified, Vec<u32>, Vec<u32>),
    Floating(invar_device::v2::Verified, Vec<u32>),
    Matrix(Box<contraction_source::DeviceEmission>, Vec<u32>, Vec<u32>),
}
enum Expected {
    Semantic(Value),
    Words(Vec<u32>),
    Indexed(indexed::Output),
    Tc([u32; 3], Vec<u32>, Vec<u32>),
}
struct Case {
    id: String,
    family: &'static str,
    work: Work,
    expected: Expected,
}

fn cases(caps: Capabilities) -> Result<Vec<Case>, Failure> {
    let mut out = Vec::new();
    for c in super::corpus::corpus(super::corpus::SEED)
        .into_iter()
        .filter(|c| {
            c.id.starts_with("map-") || c.id.starts_with("scan-") || c.id.starts_with("reduce-")
        })
    {
        let family = if c.id.starts_with("map") {
            "map"
        } else if c.id.starts_with("scan") {
            "scan"
        } else {
            "reduction"
        };
        let checked = super::front_check(&c.source).map_err(|e| Failure::new(e.code, e.message))?;
        let expected = semantic(&eval::observe(&checked, c.fuel));
        out.push(Case {
            id: c.id,
            family,
            work: Work::Source(c.source, c.fuel),
            expected: Expected::Semantic(expected),
        });
    }
    for n in [0, 1, 2, 31, 32, 33, 255, 256, 257] {
        let (a, b) = pair::inputs(n);
        let source = pair::source(&a, &b, [u32::MAX, 17]);
        let (checked, _) = pair::compile(&source);
        out.push(Case {
            id: format!("pair-{n}"),
            family: "reduction",
            expected: Expected::Semantic(semantic(&eval::observe(&checked, 100_000))),
            work: Work::Source(source.into_bytes(), 100_000),
        });
    }
    for (i, source) in floating::sources().into_iter().enumerate() {
        let family = if i < 42 && i % 7 == 5 {
            "reduction"
        } else if i < 42 && i % 7 == 6 {
            "scan"
        } else {
            "map"
        };
        let checked =
            super::front_check(source.as_bytes()).map_err(|e| Failure::new(e.code, e.message))?;
        let reference = eval::observe(&checked, 100_000);
        if reference.result != Ok(eval::Value::I32(7)) {
            return Err(Failure::new(
                "M5_INDEPENDENT_EXPECTATION",
                format!("f32 source {i}: {:?}", reference.result),
            ));
        }
        let expected = semantic(&reference);
        out.push(Case {
            id: format!("f32-source-{i}"),
            family,
            work: Work::F32Source(source.into_bytes(), 100_000),
            expected: Expected::Semantic(expected),
        });
    }
    for shape in [[0, 33], [33, 0], [1, 1], [31, 33], [33, 31]] {
        let input = grid::input(shape);
        let [r, c] = shape.map(|v| v as usize);
        let expected = (0..c)
            .flat_map(|j| (0..r).map(move |i| i * c + j))
            .map(|i| input[i])
            .collect();
        out.push(Case {
            id: format!("transpose-{shape:?}"),
            family: "transpose",
            work: Work::Transpose(grid::transpose_plan(shape, caps), input.clone()),
            expected: Expected::Words(expected),
        });
        for boundary in [Boundary::Clamp, Boundary::ConstantZero] {
            for steps in [0, 1, 2, 3] {
                out.push(Case {
                    id: format!("stencil-{shape:?}-{boundary:?}-{steps}"),
                    family: "stencil",
                    expected: Expected::Words(grid::stencil_oracle(shape, steps, boundary, &input)),
                    work: Work::Stencil(
                        grid::stencil_plan(shape, steps, boundary, caps),
                        input.clone(),
                    ),
                });
            }
        }
    }
    // Bounded source + net-oracle cases, including empty/ragged/selectivity
    // boundaries. Maximum-size physical sweeps remain a separate gate.
    for c in segments::cases(false) {
        let family = if matches!(c.kind, segments::Kind::Compact(_)) {
            "compaction"
        } else {
            "segmented"
        };
        let expected = segments::expected(&c);
        out.push(Case {
            id: c.name.clone(),
            family,
            work: Work::Segmented(segments::plan(&c, caps), c.input, c.offsets),
            expected: Expected::Words(expected),
        });
    }
    for c in indices::cases(caps) {
        let family = if matches!(
            c.bound.plan().ir().operation,
            indexed::Operation::Frontier { .. }
        ) {
            "frontier"
        } else {
            "gather_scatter"
        };
        out.push(Case {
            id: c.name.into(),
            family,
            work: Work::Indexed(c.bound),
            expected: Expected::Indexed(c.expected),
        });
    }
    for family in summaries::FAMILIES {
        for prefix in [false, true] {
            let contract = summaries::contract(family, prefix);
            for steps in [0, 1, 31, 32, 33, 255, 256, 257] {
                let (coefficients, seed) = summaries::random(family, steps, super::corpus::SEED);
                let p = contract
                    .prepare(&coefficients, &seed, steps, summaries::BUDGET)
                    .map_err(|e| Failure::new("M5_SUMMARY", format!("{e:?}")))?;
                let expected = summaries::independent(family, prefix, &coefficients, &seed);
                compare_words(&expected, &p.device_words())?;
                out.push(Case {
                    id: format!("summary-{family:?}-{prefix}-{steps}"),
                    family: "scan",
                    work: Work::Summary(
                        p.device_plan(caps)
                            .map_err(|e| Failure::new("M5_SUMMARY", format!("{e:?}")))?,
                        coefficients,
                        seed,
                    ),
                    expected: Expected::Words(expected),
                });
            }
        }
    }
    let mut f32_cases = floating::programs()
        .into_iter()
        .map(|p| {
            let input = floating::packed(&p);
            (p, input)
        })
        .collect::<Vec<_>>();
    for kind in [
        invar_device::f32_ops::Kind::Reduce { initial: 0 },
        invar_device::f32_ops::Kind::Scan { initial: 0 },
    ] {
        f32_cases.push((
            invar_device::f32_ops::Program {
                length: 3,
                inputs: vec![invar_device::f32_ops::Scalar::F32],
                nodes: vec![invar_device::f32_ops::Node::Input(0)],
                kind,
            },
            vec![0x4b800000, 0x3f800000, 0xcb800000],
        ));
    }
    for (p, packed) in f32_cases {
        let family = match p.kind {
            invar_device::f32_ops::Kind::Map => "map",
            invar_device::f32_ops::Kind::Reduce { .. } => "reduction",
            _ => "scan",
        };
        let plan = floating::plan(p.clone(), caps);
        let expected = invar::f32_source::cpu(&plan, &packed, 1)
            .map_err(|e| Failure::new("M5_F32_ORACLE", format!("{e:?}")))?;
        out.push(Case {
            id: format!("f32-{p:?}-{}", digest_words(&packed)),
            family,
            work: Work::Floating(plan, packed),
            expected: Expected::Words(expected),
        });
    }
    for dims in [[1, 1, 1], [5, 0, 7], [17, 13, 33], [3, 17, 5], [0, 3, 5]] {
        for tc in [false, true] {
            let source = if tc {
                matrix::TC_DIRECT
            } else {
                matrix::DIRECT
            };
            let [r, k, c] = dims.map(|v| v as usize);
            let a = matrix::words(
                if tc {
                    &[0x3f800000, 0x33800000]
                } else {
                    &matrix::NORMAL
                },
                r * k,
                3,
            );
            let b = matrix::words(
                if tc {
                    &[0xbf800000, 0x33800000]
                } else {
                    &matrix::NORMAL
                },
                k * c,
                5,
            );
            let emission = contraction_source::Contract::from_source(source.as_bytes(), "mat")
                .map_err(|e| Failure::new("M5_MATRIX", format!("{e:?}")))?
                .emit_device(
                    source.as_bytes(),
                    dims,
                    [(r * k) as u64, (k * c) as u64],
                    caps,
                )
                .map_err(|e| Failure::new("M5_MATRIX", format!("{e:?}")))?;
            let expected = if tc {
                Expected::Tc(dims, a.clone(), b.clone())
            } else {
                Expected::Words(matrix::oracle(dims, &a, &b))
            };
            out.push(Case {
                id: format!("matrix-{dims:?}-tc={tc}"),
                family: "matrix",
                work: Work::Matrix(Box::new(emission), a, b),
                expected,
            });
        }
    }
    if out.len() > 512 {
        return Err(Failure::new(
            "M5_CORPUS_LIMIT",
            "device corpus exceeds 512 cases",
        ));
    }
    Ok(out)
}

pub fn compare_words(expected: &[u32], actual: &[u32]) -> Result<(), Failure> {
    if expected != actual {
        return Err(Failure::new(
            "M5_DEVICE_DISAGREEMENT",
            format!(
                "{} expected words, {} observed; first difference {:?}",
                expected.len(),
                actual.len(),
                expected.iter().zip(actual).position(|(a, b)| a != b)
            ),
        ));
    }
    Ok(())
}

enum Output {
    Semantic(Value, usize, Vec<&'static str>),
    Words(Vec<u32>),
    Indexed(indexed::Output),
}

fn invocation(work: &Work) -> Value {
    let descriptor = |ir: &dyn std::fmt::Debug| digest(format!("{ir:?}").as_bytes());
    match work {
        Work::Source(source, fuel) | Work::F32Source(source, fuel) => {
            json!({"source_sha256":digest(source),"semantic_fuel":fuel})
        }
        Work::Transpose(plan, input) => {
            json!({"descriptor_debug_sha256":descriptor(plan.ir()),"input_sha256":[digest_words(input)],"shape":plan.ir().dimensions})
        }
        Work::Stencil(plan, input) => {
            json!({"descriptor_debug_sha256":descriptor(plan.ir()),"input_sha256":[digest_words(input)],"steps":plan.ir().steps})
        }
        Work::Segmented(plan, input, offsets) => {
            json!({"descriptor_debug_sha256":descriptor(plan.ir()),"input_sha256":[digest_words(input),digest_words(offsets)]})
        }
        Work::Indexed(bound) => {
            json!({"descriptor_debug_sha256":descriptor(bound.plan().ir()),"input_sha256":[digest_words(bound.values()),digest_words(bound.indices())],"start":bound.start()})
        }
        Work::Summary(plan, coefficients, seed) => {
            json!({"descriptor_debug_sha256":descriptor(&plan.plan()),"input_sha256":[digest_words(coefficients),digest_words(seed)]})
        }
        Work::Floating(plan, input) => {
            json!({"descriptor_debug_sha256":descriptor(plan.ir()),"input_sha256":[digest_words(input)],"numeric_policy":"strict-f32-v1","comparison":"bit-exact non-NaN words; D1 NaN equivalence"})
        }
        Work::Matrix(emission, a, b) => {
            json!({"source_receipt":emission.receipt(),"input_sha256":[digest_words(a),digest_words(b)]})
        }
    }
}

// The two APIs differ only by the CUDA compiler argument. Keep one adapter
// body; do not introduce a public driver trait or duplicate each family loop.
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
macro_rules! adapter {
    ($module:ident,$device:ty,$target:expr,$compiler:ident,$invoke:ident) => {
        mod $module {
            use super::*;
            pub(super) fn run(mut device:$device,$compiler:&Path)->Result<Value,Failure> {
                preflight($target,&device.name(),device.capabilities())?;
                let name=device.name().to_owned();
                let caps=device.capabilities();
                let mut rows=Vec::new();
                for case in cases(caps)? {
                    let scope=match &case.work {Work::Source(..)|Work::F32Source(..)=>"reached_source_publication",Work::Floating(..)=>"verified_f32_ir_without_reached_source_publication",_=>"source_bound_device_plan"};
                    let invocation=invocation(&case.work);
                    let actual:Result<Output,DeviceError>=(|| {Ok(match &case.work {
                        Work::Source(source,fuel)=>{
                            let checked=invar::check::check(invar::front::parse(source).map_err(|e|DeviceError{reason:Refusal::Device,detail:format!("{e:?}")})?,true).map_err(|e|DeviceError{reason:Refusal::Device,detail:format!("{e:?}")})?;
                            let program=invar::mir::lower(&checked).map_err(|e|DeviceError{reason:Refusal::Device,detail:format!("{e:?}")})?;
                            let run=$invoke!(source,&mut device,source,&program,*fuel,$compiler).map_err(|e|DeviceError{reason:if e.message=="program has no Metal-eligible region"||e.message=="program has no CUDA-eligible region"||e.message=="no Metal result was published"||e.message=="no CUDA result was published"{Refusal::NoRegion}else{Refusal::Device},detail:format!("{e:?}")})?;
                            Output::Semantic(semantic(&run.observation),run.executed.len(),Vec::new())
                        }
                        Work::F32Source(source,fuel)=>{
                            let run=$invoke!(f32_source,&mut device,source,*fuel,$compiler).map_err(|e|DeviceError{reason:Refusal::Device,detail:format!("{e:?}")})?;
                            Output::Semantic(semantic(&run.observation),run.completed,run.refusals)
                        }
                        Work::Transpose(plan,input)=>Output::Words($invoke!(device,run_transpose,plan,input,$compiler)?.output),
                        Work::Stencil(plan,input)=>Output::Words($invoke!(device,run_stencil_u32,plan,input,$compiler)?.output),
                        Work::Segmented(plan,input,offsets)=>Output::Words($invoke!(device,run_segmented,plan,input,offsets,$compiler)?.output),
                        Work::Indexed(bound)=>Output::Indexed($invoke!(device,run_indexed,bound,$compiler)?.output),
                        Work::Summary(plan,coefficients,seed)=>Output::Words($invoke!(device,run_summary,plan,coefficients,seed,$compiler)?.output),
                        Work::Floating(plan,input)=>Output::Words($invoke!(device,run_f32_ops,plan,input,$compiler)?),
                        Work::Matrix(emission,a,b)=>Output::Words($invoke!(device,run_tiled_contraction,emission.plan(),[a.as_slice(),b.as_slice()],$compiler)?.output),
                    })})();
                    let (status,diagnostic,observation)=match actual {
                        Ok(actual)=>{
                            let observation=match (&case.expected,actual) {
                                (Expected::Semantic(expected),Output::Semantic(actual,regions,refusals))=>{
                                    agree(expected,&actual)?;
                                    json!({"published_regions":regions,"semantic":actual,"fallback_reasons":refusals})
                                }
                                (Expected::Words(expected),Output::Words(actual))=>{
                                    // D1 NaNs are equivalent; all non-NaN words must be bit exact.
                                    let correct=if matches!(case.work,Work::Floating(..)){matrix::strict_equal(&actual,expected)}else{expected==&actual};
                                    if !correct {compare_words(expected,&actual)?;}
                                    json!({"words":actual.len(),"output_sha256":digest_words(&actual)})
                                }
                                (Expected::Indexed(expected),Output::Indexed(actual))=>{
                                    if &actual!=expected {return Err(Failure::new("M5_DEVICE_DISAGREEMENT",&case.id));}
                                    json!({"words":actual.values.len(),"output_sha256":digest_words(&actual.values),"frontier_masks":actual.frontier_masks,"compaction_maps":actual.compaction_maps})
                                }
                                (Expected::Tc(dims,a,b),Output::Words(actual))=>{
                                    let checked=invar_device::tc::verify(*dims,a,b,&actual).map_err(|e|Failure::new("M5_TC_BOUND",format!("{} {e:?}",case.id)))?;
                                    json!({"words":actual.len(),"output_sha256":digest_words(&actual),"numeric_policy":"tc-f16-f32-v1","max_d5_ratio":checked.max_ratio})
                                }
                                _=>return Err(Failure::new("M5_ADAPTER_OUTPUT","output type differs")),
                            };
                            if scope=="reached_source_publication" && (observation["published_regions"]==0 || observation["fallback_reasons"].as_array().is_some_and(|r|!r.is_empty())) {
                                ("unavailable",Some("M5_SOURCE_DEVICE_INELIGIBLE"),observation)
                            } else {("qualified",None,observation)}
                        }
                        Err(e) if matches!(e.reason,Refusal::Subnormal|Refusal::NoRegion)=>("unavailable",Some(e.reason.code()),json!({"fallback":"cpu","publication":false,"detail":e.detail})),
                        Err(e)=>return Err(Failure::new(e.reason.code(),format!("{}: {}",case.id,e.detail))),
                    };
                    rows.push(json!({"id":case.id,"family":case.family,"target":$target.name(),"track":"generated","status":status,"diagnostic":diagnostic,"scope":scope,"invocation":invocation,"observation":observation}));
                }
                let adversarial=adversarial(&mut device,$compiler)?;
                Ok(json!({"target":$target.name(),"status":"bounded_device_corpus_passed","device":name,"capabilities":format!("{caps:?}"),"preflight":"exact_identity_and_capabilities","rows":rows,"adversarial":adversarial}))
            }

            fn adversarial(device:&mut $device,$compiler:&Path)->Result<Value,Failure> {
                let caps=device.capabilities();
                let mut checks=Vec::new();
                let c=super::super::corpus::corpus(super::super::corpus::SEED).into_iter().find(|c|c.id=="map-257").ok_or_else(||Failure::new("M5_CORPUS_LIMIT","fuel control missing"))?;
                let checked=super::super::front_check(&c.source).map_err(|e|Failure::new(e.code,e.message))?;
                let program=invar::mir::lower(&checked).map_err(|e|Failure::new(e.code,e.message))?;
                let complete=eval::observe(&checked,c.fuel);
                for fuel in [0,1,complete.steps_used-1,complete.steps_used] {
                    let expected=semantic(&eval::observe(&checked,fuel));
                    let run=$invoke!(source,device,&c.source,&program,fuel,$compiler).map_err(|e|Failure::new(e.code,e.message))?;
                    agree(&expected,&semantic(&run.observation))?;
                    checks.push(json!({"case":"source-fuel-boundary","fuel":fuel,"steps_used":run.observation.steps_used,"published_regions":run.executed.len(),"status":"exact"}));
                }
                let plan=grid::stencil_plan([31,33],3,Boundary::Clamp,caps);
                let input=grid::input([31,33]);
                let error=$invoke!(device,run_stencil_u32,&plan,&input[1..],$compiler).err().ok_or_else(||Failure::new("M5_MISSING_REFUSAL","short stencil input"))?;
                checks.push(json!({"case":"malformed-input","diagnostic":error.reason.code(),"published":false}));
                let case=indices::cases(caps).remove(12);
                for fault in [0,1,2] {
                    let error=$invoke!(fault,device,&case.bound,$compiler,fault).err().ok_or_else(||Failure::new("M5_MISSING_REFUSAL","device fault hook published"))?;
                    checks.push(json!({"case":(["oom","launch","partial-completion"][fault]),"diagnostic":error.reason.code(),"published":false}));
                }
                if device.quarantine_len()==0 {return Err(Failure::new("M5_QUARANTINE","unconfirmed work was not quarantined"));}
                let end=std::time::Instant::now()+std::time::Duration::from_secs(10);
                while device.quarantine_len()>0 && std::time::Instant::now()<end {device.reap();std::thread::sleep(std::time::Duration::from_millis(5));}
                if device.is_poisoned(){return Err(Failure::new("M5_QUARANTINE","completion not reaped before deadline"));}
                let recovered=$invoke!(device,run_indexed,&case.bound,$compiler)?;
                if recovered.output!=case.expected {return Err(Failure::new("M5_DEVICE_DISAGREEMENT","post-quarantine recovery"));}
                Ok(json!({"checks":checks,"recovery":"exact","quarantine_after":device.quarantine_len()}))
            }
        }
    };
}

#[cfg(feature = "cuda")]
macro_rules! cuda_call {
    (f32_source,$device:expr,$source:expr,$fuel:expr,$compiler:expr)=>{invar::f32_source::run_cuda_source($source,$fuel,$device,$compiler)};
    (source,$device:expr,$source:expr,$program:expr,$fuel:expr,$compiler:expr)=>{invar::schedule::cuda::run_source_with_device($source,$program,$fuel,$device,$compiler)};
    (fault,$device:expr,$bound:expr,$compiler:expr,$fault:expr)=>{$device.run_indexed_with_fault($bound,$compiler,[invar_cuda::cuda::CollectiveFault::OutOfMemory,invar_cuda::cuda::CollectiveFault::LaunchFailure,invar_cuda::cuda::CollectiveFault::PartialCompletion][$fault]).map_err(DeviceError::from)};
    ($device:ident,$method:ident,$($args:expr),+)=>{cuda_call!(@split $device,$method,[], $($args),+)};
    (@split $device:ident,$method:ident,[$($args:expr,)*],$last:expr)=>{$device.$method($($args,)*$last).map_err(DeviceError::from)};
    (@split $device:ident,$method:ident,[$($args:expr,)*],$head:expr,$($tail:expr),+)=>{cuda_call!(@split $device,$method,[$($args,)*$head,],$($tail),+)};
}
#[cfg(all(feature = "metal", target_os = "macos"))]
macro_rules! metal_call {
    (f32_source,$device:expr,$source:expr,$fuel:expr,$compiler:expr)=>{invar::f32_source::run_metal_source($source,$fuel,$device)};
    (source,$device:expr,$source:expr,$program:expr,$fuel:expr,$compiler:expr)=>{invar::schedule::metal::run_source_with_device($source,$program,$fuel,$device)};
    (fault,$device:expr,$bound:expr,$compiler:expr,$fault:expr)=>{$device.run_indexed_with_fault($bound,[invar_metal::Fault::OutOfMemory,invar_metal::Fault::LaunchFailure,invar_metal::Fault::PartialCompletion][$fault]).map_err(DeviceError::from)};
    ($device:ident,$method:ident,$($args:expr),+)=>{metal_call!(@split $device,$method,[], $($args),+)};
    (@split $device:ident,$method:ident,[$($args:expr,)*],$last:expr)=>{$device.$method($($args),*).map_err(DeviceError::from)};
    (@split $device:ident,$method:ident,[$($args:expr,)*],$head:expr,$($tail:expr),+)=>{metal_call!(@split $device,$method,[$($args,)*$head,],$($tail),+)};
}
#[cfg(feature = "cuda")]
adapter!(
    cuda,
    invar_cuda::cuda::CudaDevice,
    Target::Cuda,
    compiler,
    cuda_call
);
#[cfg(all(feature = "metal", target_os = "macos"))]
adapter!(
    metal,
    invar_metal::MetalDevice,
    Target::Metal,
    _compiler,
    metal_call
);

fn digest_words(words: &[u32]) -> String {
    digest(
        &words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

pub fn run(target: Target, compiler: Option<&Path>) -> Result<Value, Failure> {
    match target {
        #[cfg(feature = "cuda")]
        Target::Cuda => {
            let compiler = compiler
                .filter(|p| p.is_absolute() && p.is_file())
                .ok_or_else(|| {
                    Failure::new(
                        "M5_NVRTC_REQUIRED",
                        "absolute existing --nvrtc helper required",
                    )
                })?;
            let device = invar_cuda::cuda::CudaDevice::open()
                .map_err(|e| Failure::new("M5_DEVICE_UNAVAILABLE", format!("{e:?}")))?;
            cuda::run(device, compiler)
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Target::Metal => {
            // The qualification corpus is arbitrary generated source with no
            // compiled-ahead library (#129), so it compiles MSL at run time,
            // as main's Metal GPU tests do since #206.
            let device = invar_metal::MetalDevice::open_runtime_source()
                .map_err(|e| Failure::new("M5_DEVICE_UNAVAILABLE", format!("{e:?}")))?;
            metal::run(device, Path::new(""))
        }
        #[allow(unreachable_patterns)]
        _ => {
            let _ = compiler;
            Err(Failure::new(
                "M5_DEVICE_FEATURE_REQUIRED",
                format!("{} feature and host required", target.name()),
            ))
        }
    }
}

pub fn pending(target: Target, reason: &str) -> Value {
    json!({"target":target.name(),"status":"pending","diagnostic":reason,"rows":FAMILIES.iter().flat_map(|family|["generated","library-assisted"].map(|track|json!({"family":family,"target":target.name(),"track":track,"status":"pending","diagnostic":reason}))).collect::<Vec<_>>()})
}

/// Pure boundary checks run on staging, including the same comparator and
/// preflight refusal used by hardware adapters. No mock counts as execution.
pub fn boundary_checks() -> Result<Value, Failure> {
    let mut checks = Vec::new();
    for target in [Target::Cuda, Target::Metal] {
        let wrong = preflight(target, "unqualified-device", target.caps());
        if wrong.is_ok() {
            return Err(Failure::new("M5_MISSING_REFUSAL", "wrong device admitted"));
        }
        let mut caps = target.caps();
        match &mut caps {
            Capabilities::Cuda { minor, .. } => *minor -= 1,
            Capabilities::Metal { msl_minor, .. } => *msl_minor = 0,
        }
        if preflight(
            target,
            if target == Target::Cuda {
                "NVIDIA GeForce RTX 5070 Ti"
            } else {
                "Apple M4"
            },
            caps,
        )
        .is_ok()
        {
            return Err(Failure::new(
                "M5_MISSING_REFUSAL",
                "wrong compiler admitted",
            ));
        }
        let transpose = grid::transpose_plan([31, 33], target.caps());
        let original = transpose.ir();
        let origin = invar_device::Provenance {
            source_identity: original.source_identity,
            graph_revision: original.graph_revision,
        };
        for mutation in [0, 1, 2] {
            let mut ir = original.clone();
            match mutation {
                0 => ir.input.strides = [0, 1],
                1 => ir.output.id = ir.input.id,
                _ => ir.graph_revision ^= 1,
            }
            if invar_device::transpose::verify(ir, target.caps(), origin).is_ok() {
                return Err(Failure::new(
                    "M5_MISSING_REFUSAL",
                    "invalid layout, alias or stale device IR admitted",
                ));
            }
        }
        let source = grid::stencil_source(Boundary::Clamp);
        let contract =
            invar::stencil_source::Contract::from_checked(&grid::checked(&source), "step")
                .map_err(|e| Failure::new("M5_STENCIL", format!("{e:?}")))?;
        if contract.device_u32([1, 1], 64, target.caps()).is_err()
            || contract.device_u32([1, 1], 65, target.caps()).is_ok()
        {
            return Err(Failure::new(
                "M5_DEVICE_INPUT_LIMIT",
                "stencil step limit edge",
            ));
        }
        checks.push(json!({"target":target.name(),"preflight_refusals":2,"indexed_refusals":indices::refusals(target.caps()),"physical_ir_refusals":["strides","output_alias","stale_graph"],"step_limit_edge":[64,65]}));
    }
    if compare_words(&[1, 2], &[2, 1]).is_ok() || compare_words(&[], &[0]).is_ok() {
        return Err(Failure::new("M5_MUTATION_SURVIVED", "adapter comparison"));
    }
    Ok(
        json!({"status":"passed","rows":checks,"comparison_mutants_detected":2,"scope":"pure_boundaries_no_device_execution"}),
    )
}

pub fn inventory(target: Target) -> Result<Value, Failure> {
    let cases = cases(target.caps())?;
    if cases
        .iter()
        .map(|c| &c.id)
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != cases.len()
    {
        return Err(Failure::new(
            "M5_CORPUS_DUPLICATE",
            "duplicate invocation id",
        ));
    }
    let mut families = std::collections::BTreeMap::<&str, usize>::new();
    for case in &cases {
        *families.entry(case.family).or_default() += 1;
    }
    if FAMILIES.iter().any(|f| !families.contains_key(f)) {
        return Err(Failure::new("M5_CORPUS_LIMIT", "missing family"));
    }
    Ok(
        json!({"target":target.name(),"case_count":cases.len(),"families":families,"scope":"prepared_not_executed"}),
    )
}

// Existing independently implemented baseline fixtures, with their original
// full-output validators. No handwritten baseline is called a library.
#[allow(dead_code)]
#[path = "../bench/baseline.rs"]
mod baseline;
#[allow(dead_code)]
#[path = "../bench/baseline_families.rs"]
mod baseline_families;
#[allow(dead_code)]
#[path = "../bench/family_references.rs"]
mod family_references;
mod hash {
    pub fn sha256(bytes: &[u8]) -> String {
        super::digest(bytes)
    }
}

pub fn libraries(target: Target, directory: Option<&Path>) -> Result<Vec<Value>, Failure> {
    let mut rows = Vec::new();
    for family in FAMILIES {
        let specification = match (target, family) {
            (Target::Metal, "map") => Some(("metal-u32", "map", false, false)),
            (Target::Metal, "reduction") => Some(("metal-reduce", "sum", true, false)),
            (Target::Metal, "scan") => Some(("metal-affine", "affine", false, false)),
            (Target::Metal, "matrix") => Some(("metal-gemm", "f32gemm", false, false)),
            (Target::Metal, other) => Some((
                "metal-families",
                match other {
                    "segmented" => "segsum",
                    "compaction" => "compact",
                    "gather_scatter" => "gather_hw",
                    s => s,
                },
                false,
                true,
            )),
            (Target::Cuda, "map" | "reduction" | "scan") => Some((
                "cuda-cub",
                match family {
                    "reduction" => "sum",
                    "scan" => "affine",
                    s => s,
                },
                false,
                false,
            )),
            (Target::Cuda, "matrix") => Some(("cuda-gemm", "f32gemm", false, false)),
            (Target::Cuda, other) => Some((
                "cuda-families",
                match other {
                    "segmented" => "segsum",
                    "compaction" => "compact",
                    "gather_scatter" => "gather",
                    s => s,
                },
                false,
                true,
            )),
        };
        let (status, diagnostic, receipt) = if let Some((file, fixture, v2, families)) =
            specification
        {
            let executable = directory.map(|p| p.join(file));
            if let Some(executable) = executable.filter(|p| p.is_absolute() && p.is_file()) {
                // Frontier fixtures count dense adjacency cells (32 vertices).
                let n = if family == "frontier" {
                    1024
                } else if family == "matrix" {
                    8
                } else {
                    256
                };
                let result = if families {
                    baseline_families::run(
                        &executable,
                        fixture,
                        n,
                        match target {
                            Target::Metal => baseline_families::Backend::Metal,
                            Target::Cuda => baseline_families::Backend::Cuda,
                        },
                    )
                } else {
                    baseline::run(&executable, fixture, n, v2, false)
                }
                .map_err(|e| Failure::new("M5_LIBRARY_REFUSAL", e))?;
                if result["full_output_correctness"] == "exact_full_output_fail" {
                    return Err(Failure::new(
                        "M5_LIBRARY_DISAGREEMENT",
                        format!("{family}: {result}"),
                    ));
                }
                let diagnostic = if result["full_output_correctness"] == "exact_full_output_pass" {
                    None
                } else {
                    Some("M5_LIBRARY_NUMERIC_POLICY_PENDING")
                };
                // Fixture outputs are independently checked by the existing
                // validator. Keep identity and coverage, discard timing.
                (
                    if diagnostic.is_none() {
                        "qualified"
                    } else {
                        "unavailable"
                    },
                    diagnostic,
                    json!({"baseline_name":result["baseline_name"],"kind":result["kind"],"fixture_source_sha256":result["fixture_source_sha256"],"executable_sha256":result["executable_sha256"],"output_sha256":result["output_sha256"],"vendor_versions":result["vendor_versions"],"numeric_comparison":result["numeric_comparison"],"scope":"independent_fixture_inputs_only","elements":n}),
                )
            } else {
                (
                    "unavailable",
                    Some("M5_LIBRARY_RUNNER_REQUIRED"),
                    Value::Null,
                )
            }
        } else {
            (
                "unavailable",
                Some("M5_LIBRARY_FAMILY_API_MISSING"),
                Value::Null,
            )
        };
        rows.push(json!({"family":family,"target":target.name(),"track":"library-assisted","status":status,"diagnostic":diagnostic,"receipt":receipt}));
    }
    Ok(rows)
}

pub fn attach(
    report: &mut Value,
    target: Target,
    compiler: Option<&Path>,
    libraries_dir: Option<&Path>,
) -> Result<(), Failure> {
    let mut device = run(target, compiler)?;
    let rows = device["rows"]
        .as_array_mut()
        .ok_or_else(|| Failure::new("M5_REPORT_INVALID", "device rows missing"))?;
    rows.extend(libraries(target, libraries_dir)?);
    // Only identical reached-source cases can update the CPU corpus engine
    // rows. Independent fixture recipes cannot qualify those invocations.
    let engine = format!("{}-generated", target.name());
    if let Some(programs) = report["programs"].as_array_mut() {
        for program in programs {
            if let Some(observed) = rows
                .iter()
                .find(|r| r["id"] == program["id"] && r["scope"] == "reached_source_publication")
                && let Some(row) = program["rows"]
                    .as_array_mut()
                    .and_then(|rs| rs.iter_mut().find(|r| r["engine"] == engine))
            {
                row["status"] = observed["status"].clone();
                row["diagnostic"] = observed["diagnostic"].clone();
                row["observation"] = observed["observation"]["semantic"].clone();
                row["dispatch"] = observed["observation"].clone();
            }
        }
    }
    if let Some(programs) = report["programs"].as_array() {
        let mut count = json!({"denominator":0,"qualified":0,"unavailable":0,"blocked":0});
        for row in programs
            .iter()
            .flat_map(|p| p["rows"].as_array().into_iter().flatten())
            .filter(|r| r["engine"] == engine)
        {
            let status = row["status"]
                .as_str()
                .ok_or_else(|| Failure::new("M5_REPORT_INVALID", "engine status missing"))?;
            let old = count[status]
                .as_u64()
                .ok_or_else(|| Failure::new("M5_REPORT_INVALID", "unknown engine status"))?;
            count[status] = json!(old + 1);
            count["denominator"] = json!(count["denominator"].as_u64().unwrap_or(0) + 1);
        }
        report["engines"][&engine] = count;
    }
    report["device_qualification"] = json!([
        device,
        pending(
            if target == Target::Metal {
                Target::Cuda
            } else {
                Target::Metal
            },
            "M5_TARGET_NOT_EXECUTED"
        )
    ]);
    report["device_boundary_checks"] = boundary_checks()?;
    report["pending"] = json!([
        "M4-06 / M5-01 freeze / hardware performance",
        "full frozen manifest shapes and large physical limit sweeps",
        "all required library subtypes beyond independent fixture recipes"
    ]);
    Ok(())
}

/// Every family/target/track stays in the denominator. These bounded corpus
/// checks alone cannot close the prerequisite freeze or the complete issue.
pub fn require(report: &Value) -> Result<(), Failure> {
    for target in [Target::Cuda, Target::Metal] {
        for family in FAMILIES {
            for track in ["generated", "library-assisted"] {
                let rows = report["device_qualification"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|d| d["target"] == target.name())
                    .flat_map(|d| d["rows"].as_array().into_iter().flatten())
                    .filter(|r| r["family"] == family && r["track"] == track)
                    .collect::<Vec<_>>();
                if rows.is_empty() || rows.iter().any(|r| r["status"] != "qualified") {
                    return Err(Failure::new(
                        "M5_GPU_UNAVAILABLE",
                        format!("{} {family} {track}", target.name()),
                    ));
                }
            }
        }
    }
    Err(Failure::new(
        "M5_FREEZE_PREREQUISITES_PENDING",
        "bounded adapters do not close criterion (c) or all frozen shapes",
    ))
}
