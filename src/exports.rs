//! Selected scalar functions with retained source provenance and a bounded,
//! returning native C boundary. The public L1-04 wire format is unchanged.
//! HA-1.2 admits `read`/`mut` `Array<i32|u32>` parameters of selected
//! functions through the private host-array protocol (`arrays`).
use crate::hex::hex;
use crate::{
    check, eval,
    front::{self, FrontError, Span, Type},
    mir,
};
use sha2::{Digest, Sha256};
#[cfg(feature = "native")]
use std::path::PathBuf;
use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportEngine {
    Reference,
    NativeSerial,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarType {
    I32,
    U32,
    /// Array element only; public scalar f32 parameters/results remain refused.
    F32,
    /// Array element only (HA-1.4); public scalar f64 remains refused.
    F64,
    /// Array element only (HA-1.4): one byte, exactly 0 or 1.
    Bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarValue {
    I32(i32),
    U32(u32),
}
/// Borrow mode of an admitted array parameter (HA-1.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrayMode {
    Read,
    Mut,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportParameter {
    pub name: String,
    /// The scalar type, or the element type of an array parameter.
    pub ty: ScalarType,
    /// `Some` for a `read`/`mut` `Array<i32|u32>` parameter.
    pub array: Option<ArrayMode>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportSignature {
    pub source_name: String,
    pub symbol: String,
    pub span: Span,
    pub params: Vec<ExportParameter>,
    pub result: ScalarType,
    pub artifact_id: String,
}
#[derive(Debug)]
pub enum ExportError {
    Source(FrontError),
    Evaluation(eval::EvalError),
    Boundary { code: &'static str, message: String },
}
fn boundary(code: &'static str, message: impl Into<String>) -> ExportError {
    ExportError::Boundary {
        code,
        message: message.into(),
    }
}
impl ExportError {
    pub fn to_diagnostics_json(&self, path: &str, source: &[u8]) -> serde_json::Value {
        match self {
            Self::Source(error) => error.to_diagnostics_json(path, source),
            Self::Evaluation(error) => error.to_diagnostics_json(path, source),
            Self::Boundary { code, message } => eval::EvalError::Tool {
                code,
                message: message.clone(),
            }
            .to_diagnostics_json(path, source),
        }
    }
}
impl ScalarType {
    fn from_type(ty: &Type) -> Option<Self> {
        match ty {
            Type::I32 => Some(Self::I32),
            Type::U32 => Some(Self::U32),
            _ => None,
        }
    }
    #[cfg(feature = "native")]
    fn tag(self) -> u32 {
        match self {
            Self::I32 => 1,
            Self::U32 => 2,
            // Only used as a private array element; signature admission excludes
            // scalar f32, f64 and bool.
            Self::F32 => 3,
            Self::F64 => 4,
            Self::Bool => 5,
        }
    }
}
impl ScalarValue {
    fn ty(self) -> ScalarType {
        match self {
            Self::I32(_) => ScalarType::I32,
            Self::U32(_) => ScalarType::U32,
        }
    }
    fn reference(self) -> eval::Value {
        match self {
            Self::I32(v) => eval::Value::I32(v),
            Self::U32(v) => eval::Value::U32(v),
        }
    }
    #[cfg(feature = "native")]
    fn foreign(self) -> invar_foreign::Arg<'static> {
        match self {
            Self::I32(v) => invar_foreign::Arg::I32(v),
            Self::U32(v) => invar_foreign::Arg::U32(v),
        }
    }
}

enum Implementation {
    Reference,
    #[cfg(feature = "native")]
    Native(Box<NativeArtifact>),
}
#[cfg(feature = "native")]
struct NativeArtifact {
    loaded: invar_foreign::exports::LoadedExports,
    descriptors: Vec<invar_foreign::exports::Descriptor>,
    directory: PathBuf,
    #[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
    spec: invar_foreign::exports::ArtifactSpec,
}
#[cfg(feature = "native")]
impl NativeArtifact {
    fn descriptor(&self, symbol: &str) -> Result<&invar_foreign::exports::Descriptor, ExportError> {
        self.descriptors
            .iter()
            .find(|descriptor| descriptor.symbol == symbol)
            .ok_or_else(|| {
                boundary(
                    "NATIVE_ABI",
                    "export symbol does not belong to this artifact",
                )
            })
    }
}
#[cfg(feature = "native")]
impl Drop for NativeArtifact {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}
pub struct CompiledExports {
    source: Vec<u8>,
    path: String,
    checked: check::CheckedProgram,
    signatures: Vec<ExportSignature>,
    artifact_id: String,
    implementation: Implementation,
    // Explicitly thread-confined for both engines and any future loader change.
    _thread: std::marker::PhantomData<Rc<()>>,
}

fn field(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}
/// Check the whole source, select exact concrete names and admit their complete
/// reachable scalar MIR. Does not manufacture a main/wrapper source program.
pub fn compile_exports(
    source: &[u8],
    path: &str,
    requested: &[&str],
    engine: ExportEngine,
) -> Result<CompiledExports, ExportError> {
    let parsed = front::parse(source).map_err(ExportError::Source)?;
    let original = parsed.functions.clone();
    let checked = check::check(parsed, false).map_err(ExportError::Source)?;
    crate::ownership::check(&checked).map_err(ExportError::Source)?;
    if requested.is_empty() || requested.len() > mir::MAX_FUNCTIONS || path.len() > 4096 {
        return Err(boundary(
            "NATIVE_ABI",
            "invalid export selection or source path length",
        ));
    }
    let program = mir::lower(&checked).map_err(|e| boundary("NATIVE_BACKEND", e.message))?;
    mir::verify(&program).map_err(|e| boundary("NATIVE_BACKEND", e.message))?;
    let by_name = program
        .functions
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name.as_str(), i))
        .collect::<HashMap<_, _>>();
    let mut seen = HashSet::new();
    let mut selected = Vec::new();
    let mut signatures = Vec::new();
    for (ordinal, name) in requested.iter().enumerate() {
        if !seen.insert(*name) {
            return Err(boundary("NATIVE_ABI", "duplicate export name"));
        }
        let declaration = original.iter().find(|f| f.name == *name).ok_or_else(|| {
            boundary("NATIVE_ABI", "export is not an original source declaration")
        })?;
        if !declaration.generic_params.is_empty()
            || (declaration.module.is_some() && !declaration.public)
            || declaration.params.len() > 8
        {
            return Err(boundary(
                "NATIVE_ABI",
                "export must be a visible concrete scalar function",
            ));
        }
        let result = ScalarType::from_type(&declaration.result)
            .ok_or_else(|| boundary("NATIVE_ABI", "export result is not original i32/u32"))?;
        let params = declaration
            .params
            .iter()
            .map(arrays::export_parameter)
            .collect::<Result<Vec<_>, _>>()?;
        selected.push(
            *by_name
                .get(name)
                .ok_or_else(|| boundary("NATIVE_ABI", "selected checked function missing"))?,
        );
        signatures.push(ExportSignature {
            source_name: name.to_string(),
            symbol: format!("inv_export_{ordinal}"),
            span: declaration.span,
            params,
            result,
            artifact_id: String::new(),
        });
    }
    let reachable = arrays::admit(&program, &selected)?;
    let mut hash = Sha256::new();
    field(&mut hash, b"invar-scalar-exports-v1");
    field(&mut hash, source);
    field(&mut hash, program.numeric_policy.id().as_bytes());
    field(&mut hash, &program.layout_version.to_le_bytes());
    field(&mut hash, path.as_bytes());
    field(&mut hash, env!("INVAR_BUILD_DIGEST").as_bytes());
    field(&mut hash, format!("{engine:?}").as_bytes());
    for signature in &signatures {
        field(&mut hash, format!("{signature:?}").as_bytes());
    }
    let (implementation, identity): (Implementation, [u8; 32]) = match engine {
        ExportEngine::Reference => (Implementation::Reference, hash.finalize().into()),
        ExportEngine::NativeSerial => {
            #[cfg(not(feature = "native"))]
            {
                let _ = reachable;
                return Err(boundary(
                    "NATIVE_UNAVAILABLE",
                    "native feature is not built",
                ));
            }
            #[cfg(feature = "native")]
            {
                build_native(&program, &selected, &reachable, &signatures, hash)?
            }
        }
    };
    let artifact_id = hex(&identity);
    for signature in &mut signatures {
        signature.artifact_id.clone_from(&artifact_id);
    }
    Ok(CompiledExports {
        source: source.to_vec(),
        path: path.to_owned(),
        checked,
        signatures,
        artifact_id,
        implementation,
        _thread: std::marker::PhantomData,
    })
}

impl CompiledExports {
    pub fn signatures(&self) -> &[ExportSignature] {
        &self.signatures
    }
    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }
    pub fn diagnostics(&self, error: &ExportError) -> serde_json::Value {
        error.to_diagnostics_json(&self.path, &self.source)
    }
    pub fn invoke(
        &mut self,
        name: &str,
        args: &[ScalarValue],
        fuel: u64,
    ) -> Result<ScalarValue, ExportError> {
        let descriptor = self
            .signatures
            .iter()
            .find(|s| s.source_name == name)
            .cloned()
            .ok_or_else(|| boundary("NATIVE_ABI", "unknown exported function"))?;
        self.invoke_descriptor(&descriptor, args, fuel)
    }
    pub fn invoke_descriptor(
        &mut self,
        descriptor: &ExportSignature,
        args: &[ScalarValue],
        fuel: u64,
    ) -> Result<ScalarValue, ExportError> {
        if fuel == 0 {
            return Err(ExportError::Evaluation(eval::EvalError::Tool {
                code: "INVALID_FUEL",
                message: "fuel must be positive".into(),
            }));
        }
        if !self.signatures.contains(descriptor) {
            return Err(boundary(
                "NATIVE_ABI",
                "descriptor does not belong to this artifact",
            ));
        }
        if args.len() != descriptor.params.len()
            || args
                .iter()
                .zip(&descriptor.params)
                .any(|(a, p)| p.array.is_some() || a.ty() != p.ty)
        {
            return Err(boundary(
                "NATIVE_ABI",
                "export argument arity/type mismatch",
            ));
        }
        let result = match &mut self.implementation {
            Implementation::Reference => {
                // Checked source call depth remains 64. Host interpreter frames
                // are larger than compiled frames; execute this admitted scalar
                // subset on a bounded stack independent of Python/test stacks.
                // Only Copy scalars cross the thread boundary, never Rc values.
                let checked = &self.checked;
                let name = &descriptor.source_name;
                std::thread::scope(|scope| {
                    let worker = std::thread::Builder::new()
                        .name("invar-scalar-reference".into())
                        .stack_size(16 * 1024 * 1024)
                        .spawn_scoped(scope, move || {
                            let values = args.iter().map(|v| v.reference()).collect::<Vec<_>>();
                            let observation =
                                eval::named::observe_named(checked, name, &values, fuel);
                            match observation.result.map_err(ExportError::Evaluation)? {
                                eval::Value::I32(v) => Ok(ScalarValue::I32(v)),
                                eval::Value::U32(v) => Ok(ScalarValue::U32(v)),
                                _ => Err(boundary(
                                    "NATIVE_ABI",
                                    "reference export returned a non-scalar",
                                )),
                            }
                        })
                        .map_err(|e| {
                            boundary(
                                "RESOURCE_LIMIT",
                                format!("reference worker allocation failed: {e}"),
                            )
                        })?;
                    worker
                        .join()
                        .map_err(|_| boundary("NATIVE_ABI", "reference worker invariant failed"))?
                })?
            }
            #[cfg(feature = "native")]
            Implementation::Native(artifact) => {
                let wire = args.iter().map(|v| v.foreign()).collect::<Vec<_>>();
                let descriptor = artifact.descriptor(&descriptor.symbol)?.clone();
                let invocation = artifact
                    .loaded
                    .invoke(&descriptor, &wire, fuel)
                    .map_err(|e| {
                        boundary(
                            "NATIVE_ABI",
                            format!("export loader rejected invocation: {e:?}"),
                        )
                    })?;
                decode_native(invocation, &self.source)?
            }
        };
        if result.ty() != descriptor.result {
            return Err(boundary(
                "NATIVE_ABI",
                "export returned the wrong scalar tag",
            ));
        }
        Ok(result)
    }
}

#[cfg(feature = "native")]
fn build_native(
    program: &mir::Program,
    selected: &[usize],
    reachable: &HashSet<usize>,
    signatures: &[ExportSignature],
    mut hash: Sha256,
) -> Result<(Implementation, [u8; 32]), ExportError> {
    use invar_foreign::exports::{ArtifactSpec, Descriptor, LoadedExports};
    let (triple, layout, compiler) =
        crate::native::export::configuration().map_err(|e| boundary("NATIVE_BACKEND", e))?;
    for data in [
        triple.as_bytes(),
        layout.as_bytes(),
        compiler.as_bytes(),
        b"generic;PIC;LLVM-Default;rust-opt2;panic-abort;heap256x64;v1",
        crate::native::export::linkage().as_bytes(),
    ] {
        field(&mut hash, data);
    }
    let input_identity: [u8; 32] = hash.finalize().into();
    let directory = crate::native::unique_directory().map_err(|e| boundary("NATIVE_BACKEND", e))?;
    let outcome = (|| {
        let library = directory.join(format!("scalar.{}", std::env::consts::DLL_EXTENSION));
        crate::native::export::build(program, selected, reachable, &input_identity, &library)
            .map_err(|e| boundary("NATIVE_BACKEND", e))?;
        let bytes =
            std::fs::read(&library).map_err(|e| boundary("NATIVE_BACKEND", e.to_string()))?;
        let library_sha256: [u8; 32] = Sha256::digest(bytes).into();
        let mut hash = Sha256::new();
        field(&mut hash, &input_identity);
        field(&mut hash, &library_sha256);
        let identity: [u8; 32] = hash.finalize().into();
        let descriptors = signatures
            .iter()
            .map(|s| Descriptor {
                artifact_id: identity,
                symbol: s.symbol.clone(),
                parameter_tags: s.params.iter().map(arrays::parameter_tag).collect(),
                result_tag: s.result.tag(),
            })
            .collect::<Vec<_>>();
        let spec = ArtifactSpec {
            identity,
            input_identity,
            library_sha256,
            descriptors: descriptors.clone(),
        };
        let loaded = LoadedExports::load(&library, &spec).map_err(|e| {
            boundary(
                "NATIVE_ABI",
                format!("compiler artifact load failed: {e:?}"),
            )
        })?;
        Ok((
            Implementation::Native(Box::new(NativeArtifact {
                loaded,
                descriptors,
                directory: directory.clone(),
                #[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
                spec,
            })),
            identity,
        ))
    })();
    if outcome.is_err() {
        let _ = std::fs::remove_dir_all(directory);
    }
    outcome
}

mod arrays;
#[cfg(all(
    test,
    feature = "native",
    any(target_os = "linux", target_os = "macos")
))]
mod artifact_tests;
pub use arrays::ArrayCallOutcome;

#[cfg(feature = "native")]
fn decode_native(
    invocation: invar_foreign::exports::Invocation,
    source: &[u8],
) -> Result<ScalarValue, ExportError> {
    use eval::{EvalError, TrapCategory};
    let failure = invocation.failure;
    match failure.kind {
        0 => match invocation.output {
            Some(invar_foreign::Output::I32(v)) => Ok(ScalarValue::I32(v)),
            Some(invar_foreign::Output::U32(v)) => Ok(ScalarValue::U32(v)),
            _ => Err(boundary("NATIVE_ABI", "missing canonical scalar output")),
        },
        1 => {
            let text = std::str::from_utf8(source)
                .map_err(|_| boundary("NATIVE_ABI", "retained source encoding changed"))?;
            let span = Span {
                start: failure.span_start as usize,
                end: failure.span_end as usize,
            };
            if span.start >= span.end
                || span.end > source.len()
                || !text.is_char_boundary(span.start)
                || !text.is_char_boundary(span.end)
            {
                return Err(boundary(
                    "NATIVE_ABI",
                    "native trap span is outside original source",
                ));
            }
            let category = match failure.reason {
                1 => TrapCategory::IntegerOverflow,
                2 => TrapCategory::DivisionByZero,
                3 => TrapCategory::IndexOutOfBounds,
                _ => return Err(boundary("NATIVE_ABI", "unknown scalar native trap")),
            };
            Err(ExportError::Evaluation(EvalError::Trap { category, span }))
        }
        2 => {
            let (code, message) = match failure.reason {
                1 => ("NATIVE_ABI", "native export invariant failed"),
                3 => ("RESOURCE_LIMIT", "native export allocation limit exceeded"),
                4 => ("MIR_FOR", "native loop control invariant failed"),
                5 => ("CALL_DEPTH_LIMIT", "reference call depth limit exceeded"),
                _ => return Err(boundary("NATIVE_ABI", "unknown scalar native tool reason")),
            };
            Err(ExportError::Evaluation(EvalError::Tool {
                code,
                message: message.into(),
            }))
        }
        3 => Err(ExportError::Evaluation(EvalError::EvaluationLimit {
            fuel_limit: failure.fuel_limit,
            steps_used: failure.steps_used,
        })),
        _ => Err(boundary("NATIVE_ABI", "native export ABI rejection")),
    }
}
