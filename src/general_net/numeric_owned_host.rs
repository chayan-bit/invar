//! Source-derived, synchronous loaded CPU ABI10 execution. The independent net
//! oracle is a comparator only; its interpreter never supplies this result.
use super::compiler_child::bounded;
use super::{BlockSite, Token, compiled, coverage, oracle, provenance};
use crate::hex::hex;
use crate::{
    eval,
    front::{Span, Type},
    numeric_contract::F32Bits,
};
use invar_foreign::numeric_owned::{self as numeric, Driver};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};
const MAX_SOURCE: usize = 2 * 1024 * 1024;
const MAX_FRAMES: usize = 128 * 1024;
const DEADLINE: Duration = Duration::from_secs(30);
const OPTIONS: &[&str] = &[
    "-std=c++17",
    "-O2",
    "-fPIC",
    "-fvisibility=hidden",
    "-fno-fast-math",
    "-ffp-contract=off",
];
#[derive(Debug)]
pub enum Error {
    Provenance(provenance::Error),
    Loader(numeric::Error),
    Load {
        cause: numeric::Error,
        evidence: PathBuf,
    },
    Input,
    Limit,
    Allocation,
    Io(String),
    Compile {
        message: String,
        evidence: PathBuf,
    },
    Malformed,
    Closed,
}
struct Directory {
    path: PathBuf,
    remove: bool,
}
impl Drop for Directory {
    fn drop(&mut self) {
        if self.remove {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}
/// A source-issued plan and persistent thread-confined library. Dropping unloads
/// the driver before removing its directory. Repeated calls use fresh frames.
/// ```compile_fail
/// fn send<T: Send>() {} send::<invar::general_net::numeric_owned_host::Plan>();
/// ```
/// ```compile_fail
/// fn sync<T: Sync>() {} sync::<invar::general_net::numeric_owned_host::Plan>();
/// ```
pub struct Plan {
    driver: Driver,
    artifact: provenance::ProvenancedArtifact,
    receipt: Json,
    sites: BTreeMap<TokenKey, (coverage::Rule, Span)>,
    program: crate::mir::Program,
    storage: compiled::OwnedLayout,
    cleanup_name_bytes: usize,
    directory: Directory,
    closed: bool,
}
type TokenKey = (usize, usize, usize);
fn hash(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn io(e: std::io::Error) -> Error {
    Error::Io(e.to_string())
}
fn reserve<T>(n: usize) -> Result<Vec<T>, Error> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| Error::Allocation)?;
    Ok(v)
}
fn private_directory(prefix: &str) -> Result<PathBuf, Error> {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    // PIDs can be reused while preserved failure evidence survives. Atomic
    // creation refuses collisions and never adopts an existing directory.
    let epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| Error::Io(e.to_string()))?
        .as_nanos();
    for _ in 0..128 {
        let nonce = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| Error::Limit)?;
        let path =
            std::env::temp_dir().join(format!("{prefix}-{}-{epoch}-{nonce}", std::process::id()));
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        match builder.create(&path) {
            Ok(()) => return Ok(path),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(io(e)),
        }
    }
    Err(Error::Io("private artifact directory collisions".into()))
}
fn compiler_path(path: &Path) -> Result<PathBuf, Error> {
    if path.components().count() > 1 || path.is_absolute() {
        return fs::canonicalize(path).map_err(io);
    }
    let paths = std::env::var_os("PATH").ok_or(Error::Input)?;
    for dir in std::env::split_paths(&paths) {
        let candidate = dir.join(path);
        if candidate.is_file() {
            return fs::canonicalize(candidate).map_err(io);
        }
    }
    Err(Error::Input)
}
fn wrapper(
    artifact: &compiled::Artifact,
    symbol: &str,
    identity: [u8; 32],
) -> Result<String, Error> {
    // Headers must remain outside the namespace. Hide all emitted helpers, even
    // inline names, so two simultaneously loaded artifacts cannot interpose.
    let source = artifact.source();
    let body = source
        .lines()
        .filter(|l| !l.starts_with("#include"))
        .collect::<Vec<_>>()
        .join("\n");
    let id = identity
        .iter()
        .map(|b| b.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let text = format!(
        r#"#include <climits>
#include <cstddef>
#include <new>
namespace {symbol}_private {{
{body}
}}
#if defined(_WIN32)
#define INV_EXPORT __declspec(dllexport)
#else
#define INV_EXPORT __attribute__((visibility("default")))
#endif
extern "C" INV_EXPORT unsigned int {symbol}_identity(unsigned char* out,std::size_t len) noexcept {{
 if(!out||len!=32) return 2; const unsigned char id[32]={{{id}}};for(std::size_t i=0;i<32;++i)out[i]=id[i];return 0;
}}
extern "C" INV_EXPORT unsigned int {symbol}(const long long* input,std::size_t n,long long* output,std::size_t out_n,unsigned long long* trace,std::size_t trace_n,long long* heap,std::size_t heap_n,unsigned long long* cleanup,std::size_t cleanup_n,unsigned long long fuel,unsigned long long transitions,unsigned long long cap) noexcept {{
 if(n!={params}||!input||!output||out_n!=10||!trace||!heap||!cleanup||heap_n!={heap}||cleanup_n!={cleanup}||fuel==0||fuel>1000000||transitions>1000000||cap>1000000||trace_n!=cap*7) return 2;
 try {{ {symbol}_private::gnet_run(input,output,trace,fuel,transitions,cap,heap,cleanup);return 0; }}
 catch(const std::bad_alloc&) {{return 1;}} catch(...) {{return 3;}}
}}
"#,
        params = artifact.params().len(),
        heap = artifact.owned_layout().ok_or(Error::Input)?.heap_words(),
        cleanup = artifact.owned_layout().ok_or(Error::Input)?.cleanup_words()
    );
    if text.len() > MAX_SOURCE {
        return Err(Error::Limit);
    }
    Ok(text)
}
impl Plan {
    pub fn from_source(
        source: &[u8],
        entry: &str,
        limits: provenance::Limits,
        storage: compiled::OwnedLimits,
        compiler: &Path,
    ) -> Result<Self, Error> {
        if limits.compiler.frame_bytes > MAX_FRAMES {
            return Err(Error::Limit);
        }
        let artifact = provenance::compile_owned_numeric(
            source,
            entry,
            compiled::Target::Cpu,
            limits,
            storage,
        )
        .map_err(Error::Provenance)?;
        if artifact.artifact().abi_version() != 10
            || artifact.artifact().frame_bytes() > MAX_FRAMES
            || artifact.artifact().params().len() > 128
        {
            return Err(Error::Limit);
        }
        let storage = artifact
            .artifact()
            .owned_layout()
            .cloned()
            .ok_or(Error::Input)?;
        let compiler = compiler_path(compiler)?;
        let compiler_hash = numeric::file_digest(&compiler).map_err(Error::Loader)?;
        let mut directory = Directory {
            path: private_directory("invar-gnet10-host")?,
            remove: false,
        };
        let compile_error = |message: String| Error::Compile {
            message,
            evidence: directory.path.clone(),
        };
        let (version_ok, version_out, version_err) = bounded(
            Command::new(&compiler).arg("--version"),
            &directory.path.join("compiler-version"),
            DEADLINE,
        )
        .map_err(&compile_error)?;
        fs::write(directory.path.join("compiler-version.stdout"), &version_out).map_err(io)?;
        fs::write(directory.path.join("compiler-version.stderr"), &version_err).map_err(io)?;
        if !version_ok {
            return Err(compile_error("compiler version failed".into()));
        }
        let prelink = json!({"version":1,"abi":10,"heap_words":storage.heap_words(),"cleanup_words":storage.cleanup_words(),"elements":storage.elements(),"allocations":storage.allocations(),"artifact":artifact.id(),"compiler_sha256":hex(&compiler_hash),"compiler_path":compiler,"compiler_version":hex(&hash(&version_out)),"compiler_version_stderr":hex(&hash(&version_err)),"options":OPTIONS,"shared":if cfg!(target_os="macos"){"-dynamiclib"}else{"-shared"},"host_build":env!("INVAR_BUILD_DIGEST"),"frame_bound":MAX_FRAMES,"source_bound":MAX_SOURCE,"library_bound":64*1024*1024});
        let identity = hash(prelink.to_string().as_bytes());
        let symbol = format!("invar_{}_numeric_owned_v1", hex(&identity));
        let text = wrapper(artifact.artifact(), &symbol, identity)?;
        let source_path = directory.path.join("numeric.cpp");
        let library = directory
            .path
            .join(format!("numeric.{}", std::env::consts::DLL_EXTENSION));
        fs::write(&source_path, &text).map_err(io)?;
        fs::write(directory.path.join("prelink.json"), prelink.to_string()).map_err(io)?;
        let mut command = Command::new(&compiler);
        command
            .args(OPTIONS)
            .arg(if cfg!(target_os = "macos") {
                "-dynamiclib"
            } else {
                "-shared"
            })
            .arg(&source_path)
            .arg("-o")
            .arg(&library);
        let compile = bounded(&mut command, &directory.path.join("compiler"), DEADLINE);
        let (success, stdout, stderr) = compile.map_err(&compile_error)?;
        fs::write(directory.path.join("compiler.stdout"), &stdout).map_err(io)?;
        fs::write(directory.path.join("compiler.stderr"), &stderr).map_err(io)?;
        if !success {
            return Err(compile_error(format!(
                "compiler failed: {}",
                String::from_utf8_lossy(&stderr)
            )));
        }
        if numeric::file_digest(&compiler).map_err(Error::Loader)? != compiler_hash {
            return Err(Error::Loader(numeric::Error::Identity));
        }
        let library_hash = numeric::file_digest(&library).map_err(Error::Loader)?;
        let receipt = json!({"scope":"source_issued_abi10_loaded_cpu","artifact":artifact.manifest(),"prelink":prelink,"identity":hex(&identity),"wrapper_sha256":hex(&hash(text.as_bytes())),"wrapper_bytes":text.len(),"library_sha256":hex(&library_hash),"library_bytes":fs::metadata(&library).map_err(io)?.len(),"symbol":symbol,"bounds":{"fuel":1000000,"transitions":1000000,"trace_events":1000000,"complete_frames_bytes":MAX_FRAMES}});
        fs::write(directory.path.join("receipt.json"), receipt.to_string()).map_err(io)?;
        let mut carriers = reserve(artifact.artifact().params().len())?;
        for ty in artifact.artifact().params() {
            carriers.push(match ty {
                Type::I32 => numeric::Carrier::I32,
                Type::Bool => numeric::Carrier::Bool,
                _ => numeric::Carrier::Word,
            });
        }
        let driver = Driver::load(
            &library,
            library_hash,
            identity,
            &symbol,
            carriers,
            [storage.heap_words(), storage.cleanup_words()],
        )
        .map_err(|cause| Error::Load {
            cause,
            evidence: directory.path.clone(),
        })?;
        let (net, proof) = artifact.checked_net();
        let program = net
            .compiled_mir(proof)
            .map_err(|e| Error::Provenance(provenance::Error::Net(e)))?;
        let mut sites = BTreeMap::new();
        for (token, rule, span) in net
            .agents
            .iter()
            .map(|a| (a.token, a.rule, a.span))
            .chain(net.control.iter().map(|a| (a.token, a.rule, a.span)))
        {
            let operator = program
                .functions
                .get(token.block.function)
                .and_then(|f| f.blocks.get(token.block.block))
                .and_then(|b| b.instructions.get(token.position))
                .and_then(|i| match i.kind {
                    crate::mir::Operation::Binary { operator, .. }
                    | crate::mir::Operation::Unary { operator, .. } => Some(operator),
                    _ => None,
                });
            sites.insert(
                (token.block.function, token.block.block, token.position),
                (rule, operator.unwrap_or(span)),
            );
        }
        let longest_name = program
            .functions
            .iter()
            .flat_map(|f| f.declaration_order.iter())
            .map(|n| n.len())
            .max()
            .unwrap_or(0);
        let cleanup_name_bytes = longest_name
            .checked_mul(storage.cleanup_events())
            .ok_or(Error::Limit)?;
        if cleanup_name_bytes > 16 * 1024 * 1024 {
            return Err(Error::Limit);
        }
        directory.remove = true;
        Ok(Self {
            driver,
            artifact,
            receipt,
            sites,
            program,
            storage,
            cleanup_name_bytes,
            directory,
            closed: false,
        })
    }
    pub fn artifact(&self) -> &provenance::ProvenancedArtifact {
        &self.artifact
    }
    pub fn receipt(&self) -> &Json {
        &self.receipt
    }
    pub fn evidence_directory(&self) -> &Path {
        &self.directory.path
    }
    pub fn run(
        &mut self,
        args: &[eval::Value],
        limits: oracle::Limits,
    ) -> Result<oracle::Observation, Error> {
        if self.closed {
            return Err(Error::Closed);
        }
        if limits.fuel == 0
            || limits.fuel > 1_000_000
            || limits.transitions > 1_000_000
            || limits.trace_events > 1_000_000
        {
            return Err(Error::Limit);
        }
        // The artifact fixes native heap capacity. Refuse a tighter caller cap
        // instead of silently exceeding it; native storage failures stay distinct.
        if limits.array_elements < self.storage.elements() {
            return Err(Error::Limit);
        }
        let bytes = limits
            .trace_events
            .checked_mul(56 + std::mem::size_of::<oracle::Event>())
            .and_then(|n| n.checked_add(self.storage.storage_bytes()))
            .and_then(|n| n.checked_add(self.cleanup_name_bytes))
            .and_then(|n| {
                n.checked_add(
                    self.storage.cleanup_events() * std::mem::size_of::<eval::CleanupEvent>(),
                )
            })
            .and_then(|n| n.checked_add(self.storage.elements() * 4))
            .ok_or(Error::Limit)?;
        if bytes > 128 * 1024 * 1024 {
            return Err(Error::Limit);
        }
        let input = encode(self.artifact.artifact().params(), args)?;
        let words = match self.driver.run(
            &input,
            [limits.fuel, limits.transitions, limits.trace_events as u64],
        ) {
            Ok(words) => words,
            Err(cause) => {
                self.closed = true;
                self.directory.remove = false;
                let _ = fs::write(
                    self.directory.path.join("driver-failure.txt"),
                    format!("{cause:?}"),
                );
                return Err(Error::Load {
                    cause,
                    evidence: self.directory.path.clone(),
                });
            }
        };
        match decode(
            self.artifact.artifact().result(),
            self.artifact.source().len(),
            &self.sites,
            &self.program,
            &self.storage,
            limits,
            &words,
        ) {
            Ok(o) => Ok(o),
            Err(e) => {
                self.closed = true;
                self.directory.remove = false;
                let _ = fs::write(
                    self.directory.path.join("malformed-output.json"),
                    json!(words.output).to_string(),
                );
                if let Ok(file) = fs::File::create(self.directory.path.join("malformed-trace.bin"))
                {
                    let mut writer = std::io::BufWriter::new(file);
                    for word in &words.trace {
                        if writer.write_all(&word.to_le_bytes()).is_err() {
                            break;
                        }
                    }
                }
                // Stream fixed-width initialized buffers rather than allocating a
                // second, potentially much larger JSON representation on failure.
                for (name, words) in [("malformed-heap.bin", &words.heap)] {
                    if let Ok(file) = fs::File::create(self.directory.path.join(name)) {
                        let mut writer = std::io::BufWriter::new(file);
                        for word in words {
                            if writer.write_all(&word.to_le_bytes()).is_err() {
                                break;
                            }
                        }
                    }
                }
                if let Ok(file) =
                    fs::File::create(self.directory.path.join("malformed-cleanup.bin"))
                {
                    let mut writer = std::io::BufWriter::new(file);
                    for word in &words.cleanup {
                        if writer.write_all(&word.to_le_bytes()).is_err() {
                            break;
                        }
                    }
                }
                Err(e)
            }
        }
    }
}
fn encode(types: &[Type], args: &[eval::Value]) -> Result<Vec<i64>, Error> {
    if types.len() != args.len() {
        return Err(Error::Input);
    }
    let mut values = reserve(args.len())?;
    for (ty, arg) in types.iter().zip(args) {
        values.push(match (ty, arg) {
            (Type::F32, eval::Value::F32(x)) => i64::from(x.to_bits()),
            (Type::I32, eval::Value::I32(x)) => i64::from(*x),
            (Type::U32, eval::Value::U32(x)) => i64::from(*x),
            (Type::Bool, eval::Value::Bool(x)) => i64::from(*x),
            _ => return Err(Error::Input),
        });
    }
    Ok(values)
}
pub(super) fn decode(
    result: &Type,
    source_len: usize,
    sites: &BTreeMap<TokenKey, (coverage::Rule, Span)>,
    program: &crate::mir::Program,
    storage: &compiled::OwnedLayout,
    l: oracle::Limits,
    raw: &numeric::Invocation,
) -> Result<oracle::Observation, Error> {
    let h = raw.output;
    let [status, start, end, steps, transitions, frames, count] =
        [h[0], h[2], h[3], h[4], h[5], h[6], h[7]]
            .map(|n| u64::try_from(n).map_err(|_| Error::Malformed));
    let (status, start, end, steps, transitions, frames, count) =
        (status?, start?, end?, steps?, transitions?, frames?, count?);
    let used = usize::try_from(count)
        .ok()
        .and_then(|n| n.checked_mul(7))
        .ok_or(Error::Malformed)?;
    if status > 13
        || status == 8
        || frames == 0
        || frames > 64
        || steps > l.fuel
        || transitions > l.transitions
        || count > l.trace_events as u64
        || start > end
        || end > source_len as u64
        || raw.trace.len() != l.trace_events * 7
        || used > raw.trace.len()
        || raw.trace[used..].iter().any(|n| *n != 0)
    {
        return Err(Error::Malformed);
    }
    if (!matches!(status, 1 | 2 | 9) && (start != 0 || end != 0))
        || (matches!(status, 1 | 2 | 9) && start == end)
        || (status != 0 && h[1] != 0)
        || (status == 3 && steps != l.fuel)
        || (status == 4 && frames != 64)
        || (status == 5 && transitions != l.transitions)
        || (status == 6 && count != l.trace_events as u64)
    {
        return Err(Error::Malformed);
    }
    let mut events = reserve(count as usize)?;
    let mut pending: Vec<(u64, u64, TokenKey, coverage::Rule)> = reserve(64)?;
    for row in raw.trace[..used].chunks_exact(7) {
        let function = usize::try_from(row[3]).map_err(|_| Error::Malformed)?;
        let block = usize::try_from(row[4]).map_err(|_| Error::Malformed)?;
        let position = usize::try_from(row[5]).map_err(|_| Error::Malformed)?;
        let (rule, _) = sites
            .get(&(function, block, position))
            .ok_or(Error::Malformed)?;
        if row[0] > 1 || row[1] > transitions || row[2] > transitions || row[6] != *rule as u64 {
            return Err(Error::Malformed);
        }
        let key = (row[1], row[2], (function, block, position), *rule);
        if row[0] == 0 {
            if pending.len() == 64
                || pending
                    .last()
                    .is_some_and(|last| last.3 != coverage::Rule::Call || row[1] <= last.0)
            {
                return Err(Error::Malformed);
            }
            pending.push(key);
        } else if pending.pop() != Some(key) {
            return Err(Error::Malformed);
        }
        events.push(oracle::Event {
            kind: if row[0] == 0 {
                oracle::EventKind::Demand
            } else {
                oracle::EventKind::Complete
            },
            frame: row[1],
            activation: row[2],
            token: Token {
                block: BlockSite { function, block },
                position,
            },
            rule: *rule,
        });
    }
    if status == 0
        && (!pending.is_empty()
            || !events.last().is_some_and(|e| {
                e.kind == oracle::EventKind::Complete
                    && e.rule == coverage::Rule::Return
                    && e.frame == 0
            }))
    {
        return Err(Error::Malformed);
    }
    if matches!(status, 1 | 2 | 9) {
        let last = events.last().ok_or(Error::Malformed)?;
        let (_, span) = sites
            .get(&(
                last.token.block.function,
                last.token.block.block,
                last.token.position,
            ))
            .ok_or(Error::Malformed)?;
        if last.kind != oracle::EventKind::Demand
            || span.start as u64 != start
            || span.end as u64 != end
        {
            return Err(Error::Malformed);
        }
    }
    let failure = match status {
        5 => Some(oracle::ToolFailure::TransitionLimit),
        6 => Some(oracle::ToolFailure::TraceLimit),
        7 => Some(oracle::ToolFailure::InvalidState),
        11..=13 => Some(oracle::ToolFailure::Allocation),
        _ => None,
    };
    let (root, cleanup_trace) = decode_storage(result, program, storage, &events, status, raw)?;
    let outcome = match status {
        0 => Ok(if let Some(root) = root {
            root
        } else {
            match result {
                Type::F32 => eval::Value::F32(F32Bits::from_bits(
                    u32::try_from(h[1]).map_err(|_| Error::Malformed)?,
                )),
                Type::I32 => eval::Value::I32(i32::try_from(h[1]).map_err(|_| Error::Malformed)?),
                Type::U32 => eval::Value::U32(u32::try_from(h[1]).map_err(|_| Error::Malformed)?),
                Type::Bool if h[1] == 0 || h[1] == 1 => eval::Value::Bool(h[1] != 0),
                Type::Unit if h[1] == 0 => eval::Value::Unit,
                _ => return Err(Error::Malformed),
            }
        }),
        1 | 2 | 9 => Err(eval::EvalError::Trap {
            category: if status == 1 {
                eval::TrapCategory::IntegerOverflow
            } else if status == 2 {
                eval::TrapCategory::DivisionByZero
            } else {
                eval::TrapCategory::IndexOutOfBounds
            },
            span: Span {
                start: start as usize,
                end: end as usize,
            },
        }),
        3 => Err(eval::EvalError::EvaluationLimit {
            fuel_limit: l.fuel,
            steps_used: steps,
        }),
        4 => Err(eval::EvalError::Tool {
            code: "CALL_DEPTH_LIMIT",
            message: "compiled ABI10 call-depth limit".into(),
        }),
        5 | 6 => Err(eval::EvalError::Tool {
            code: "RESOURCE_LIMIT",
            message: "compiled ABI10 transition/trace limit".into(),
        }),
        7 => Err(eval::EvalError::Tool {
            code: "MIR_INVALID",
            message: "compiled ABI10 invalid state".into(),
        }),
        10 => Err(eval::EvalError::Tool {
            code: "ARRAY_SIZE_LIMIT",
            message: "compiled ABI10 source array-size limit".into(),
        }),
        11..=13 => Err(eval::EvalError::Tool {
            code: "RESOURCE_LIMIT",
            message: format!("compiled ABI10 storage limit {status}"),
        }),
        _ => return Err(Error::Malformed),
    };
    Ok(oracle::Observation {
        semantic: eval::Observation {
            result: outcome,
            steps_used: steps,
            cleanup_trace,
        },
        failure,
        events,
        transitions,
        max_frames: frames as usize,
    })
}

fn is_array(ty: &Type) -> bool {
    matches!(ty, Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32)
}
fn allocating(program: &crate::mir::Program, event: &oracle::Event) -> bool {
    let Some(i) = program.functions[event.token.block.function].blocks[event.token.block.block]
        .instructions
        .get(event.token.position)
    else {
        return false;
    };
    matches!(i.kind, crate::mir::Operation::Array(_))
        || matches!(&i.kind, crate::mir::Operation::Call { name, .. }
            if matches!(name.as_str(), "filled_i32" | "filled_u32" | "filled_f32"))
}
fn cleanup_names<'a>(program: &'a crate::mir::Program, event: &oracle::Event) -> &'a [String] {
    if event.kind != oracle::EventKind::Demand {
        return &[];
    }
    let block = &program.functions[event.token.block.function].blocks[event.token.block.block];
    match block
        .instructions
        .get(event.token.position)
        .map(|i| &i.kind)
    {
        Some(crate::mir::Operation::Drop(names)) => names,
        Some(
            crate::mir::Operation::Bind { name, .. } | crate::mir::Operation::Store { name, .. },
        ) => std::slice::from_ref(name),
        None => match &block.end {
            Some(crate::mir::Terminator::Return { cleanup, .. }) => cleanup,
            _ => &[],
        },
        _ => &[],
    }
}
fn decode_storage(
    result: &Type,
    program: &crate::mir::Program,
    layout: &compiled::OwnedLayout,
    events: &[oracle::Event],
    status: u64,
    raw: &numeric::Invocation,
) -> Result<(Option<eval::Value>, Vec<eval::CleanupEvent>), Error> {
    use std::{cell::RefCell, rc::Rc};
    let allocated = usize::try_from(raw.output[9]).map_err(|_| Error::Malformed)?;
    let count = usize::try_from(raw.output[8]).map_err(|_| Error::Malformed)?;
    if allocated > layout.allocations()
        || count > layout.cleanup_events()
        || raw.heap.len() != layout.heap_words()
        || raw.cleanup.len() != layout.cleanup_words()
        || raw.cleanup[count * 4..].iter().any(|n| *n != 0)
        || raw.heap[allocated * 4..layout.allocations() * 4]
            .iter()
            .any(|n| *n != 0)
        || (status == 12 && allocated != layout.allocations())
        || (status == 13 && count != layout.cleanup_events())
    {
        return Err(Error::Malformed);
    }
    let mut allocations = events
        .iter()
        .filter(|e| e.kind == oracle::EventKind::Complete && allocating(program, e))
        .count();
    if status == 6
        && events
            .last()
            .is_some_and(|e| e.kind == oracle::EventKind::Demand && allocating(program, e))
    {
        allocations += 1;
    }
    if allocations != allocated {
        return Err(Error::Malformed);
    }
    let pool = &raw.heap[layout.allocations() * 4..];
    // Reclaimed pool words may remain as initialized stale values after stable-ID
    // compaction. They cannot publish a root; every word still has a 32-bit carrier.
    if pool
        .iter()
        .any(|w| *w < i64::from(i32::MIN) || *w > i64::from(u32::MAX))
    {
        return Err(Error::Malformed);
    }
    let mut live = reserve(allocated)?;
    let mut dead_lengths = BTreeMap::<usize, usize>::new();
    let mut ever_words = 0usize;
    for (index, d) in raw.heap[..allocated * 4].chunks_exact(4).enumerate() {
        let offset = usize::try_from(d[0]).map_err(|_| Error::Malformed)?;
        let len = usize::try_from(d[1]).map_err(|_| Error::Malformed)?;
        let end = offset.checked_add(len).ok_or(Error::Malformed)?;
        if len > crate::array_limits::MAX_ARRAY_LEN
            || end > pool.len()
            || !matches!(d[2], 0 | 1 | 258)
            || !matches!(d[3], 0 | 1)
        {
            return Err(Error::Malformed);
        }
        ever_words = ever_words.saturating_add(len).min(pool.len());
        if d[3] == 1 {
            if pool[offset..end].iter().any(|w| {
                if d[2] == 0 {
                    i32::try_from(*w).is_err()
                } else {
                    u32::try_from(*w).is_err()
                }
            }) {
                return Err(Error::Malformed);
            }
            live.push((offset, end, index, d[2]));
        } else {
            *dead_lengths.entry(len).or_default() += 1;
        }
    }
    // Compaction can leave stale words, but no write can reach beyond the sum
    // of every stable allocation length ever issued in this invocation.
    if pool[ever_words..].iter().any(|w| *w != 0) {
        return Err(Error::Malformed);
    }
    live.sort_unstable();
    if live.windows(2).any(|pair| pair[0].1 > pair[1].0) || allocated - live.len() != count {
        return Err(Error::Malformed);
    }
    let mut cleanup = reserve(count)?;
    let (mut event_at, mut name_at) = (0usize, 0usize);
    for row in raw.cleanup[..count * 4].chunks_exact(4) {
        let function = usize::try_from(row[0]).map_err(|_| Error::Malformed)?;
        let place = usize::try_from(row[1]).map_err(|_| Error::Malformed)?;
        let len = usize::try_from(row[2]).map_err(|_| Error::Malformed)?;
        let f = program.functions.get(function).ok_or(Error::Malformed)?;
        let name = f.declaration_order.get(place).ok_or(Error::Malformed)?;
        if !f.locals.get(name).is_some_and(|v| is_array(&v.ty)) {
            return Err(Error::Malformed);
        }
        let remaining = dead_lengths.get_mut(&len).ok_or(Error::Malformed)?;
        if *remaining == 0 {
            return Err(Error::Malformed);
        }
        *remaining -= 1;
        let mut found = false;
        while let Some(event) = events.get(event_at) {
            let names = cleanup_names(program, event);
            if event.frame == row[3] && event.token.block.function == function {
                while name_at < names.len() {
                    let candidate = &names[name_at];
                    name_at += 1;
                    if candidate == name {
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }
            event_at += 1;
            name_at = 0;
        }
        if !found {
            return Err(Error::Malformed);
        }
        cleanup.push(eval::CleanupEvent {
            name: name.clone(),
            length: len,
        });
    }
    if dead_lengths.values().any(|n| *n != 0) {
        return Err(Error::Malformed);
    }
    if status != 0 {
        return Ok((None, cleanup));
    }
    if !is_array(result) {
        if !live.is_empty() {
            return Err(Error::Malformed);
        }
        return Ok((None, cleanup));
    }
    let handle = usize::try_from(raw.output[1])
        .ok()
        .and_then(|n| n.checked_sub(1))
        .ok_or(Error::Malformed)?;
    if live.len() != 1 || live[0].2 != handle {
        return Err(Error::Malformed);
    }
    let (start, end, _, kind) = live[0];
    let words = &pool[start..end];
    let value = match (result, kind) {
        (Type::ArrayI32, 0) => {
            let mut xs = reserve(words.len())?;
            for x in words {
                xs.push(i32::try_from(*x).map_err(|_| Error::Malformed)?);
            }
            eval::Value::ArrayI32(Rc::new(RefCell::new(xs)))
        }
        (Type::ArrayU32, 1) => {
            let mut xs = reserve(words.len())?;
            for x in words {
                xs.push(u32::try_from(*x).map_err(|_| Error::Malformed)?);
            }
            eval::Value::ArrayU32(Rc::new(RefCell::new(xs)))
        }
        (Type::ArrayF32, 258) => {
            let mut xs = reserve(words.len())?;
            for x in words {
                xs.push(F32Bits::from_bits(
                    u32::try_from(*x).map_err(|_| Error::Malformed)?,
                ));
            }
            eval::Value::ArrayF32(Rc::new(RefCell::new(xs)))
        }
        _ => return Err(Error::Malformed),
    };
    Ok((Some(value), cleanup))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn prepared() -> (Plan, oracle::Limits, numeric::Invocation) {
        let source = b"fn run()->Array<f32>{let gone:Array<f32> = [1.0,2.0];let root:Array<f32> = [-0.0];return root;}";
        let mut plan = Plan::from_source(
            source,
            "run",
            Default::default(),
            compiled::OwnedLimits {
                elements: 16,
                allocations: 8,
                cleanup_events: 8,
            },
            Path::new("c++"),
        )
        .unwrap();
        let limits = oracle::Limits {
            fuel: 100,
            transitions: 1000,
            trace_events: 200,
            array_elements: 16,
        };
        let raw = plan
            .driver
            .run(
                &[],
                [limits.fuel, limits.transitions, limits.trace_events as u64],
            )
            .unwrap();
        (plan, limits, raw)
    }
    fn observe(
        p: &Plan,
        l: oracle::Limits,
        r: &numeric::Invocation,
    ) -> Result<oracle::Observation, Error> {
        decode(
            p.artifact.artifact().result(),
            p.artifact.source().len(),
            &p.sites,
            &p.program,
            &p.storage,
            l,
            r,
        )
    }
    fn copy(r: &numeric::Invocation) -> numeric::Invocation {
        numeric::Invocation {
            output: r.output,
            trace: r.trace.clone(),
            heap: r.heap.clone(),
            cleanup: r.cleanup.clone(),
        }
    }
    #[test]
    fn issue_gnet_10_loaded_malformed_headers_storage_and_events() {
        let (p, l, raw) = prepared();
        observe(&p, l, &raw).unwrap();
        assert_eq!(raw.output[8], 1);
        assert_eq!(raw.output[9], 2);
        for (i, value) in [
            (0, -1),
            (0, 8),
            (0, 14),
            (1, 0),
            (1, 9),
            (2, 1),
            (3, 9999),
            (4, -1),
            (4, 101),
            (5, 1001),
            (6, 0),
            (6, 65),
            (7, -1),
            (7, 201),
            (8, -1),
            (8, 9),
            (9, -1),
            (9, 9),
        ] {
            let mut r = copy(&raw);
            r.output[i] = value;
            assert!(
                matches!(observe(&p, l, &r), Err(Error::Malformed)),
                "header[{i}]={value}"
            );
        }
        // Descriptor offsets, lengths, unsupported nominal kind, live flag,
        // unused descriptors, live f32 carriers, and unused pool tail.
        for (i, value) in [
            (0, -1),
            (1, 17),
            (2, 2),
            (3, 2),
            (4, 17),
            (5, -1),
            (6, 257),
            (7, 0),
            (8, 1),
            (34, -1),
            (47, 1),
        ] {
            let mut r = copy(&raw);
            r.heap[i] = value;
            assert!(
                matches!(observe(&p, l, &r), Err(Error::Malformed)),
                "heap[{i}]={value}"
            );
        }
        for (i, value) in [(0, 99), (1, 99), (2, 99), (3, 99), (4, 1)] {
            let mut r = copy(&raw);
            r.cleanup[i] = value;
            assert!(
                matches!(observe(&p, l, &r), Err(Error::Malformed)),
                "cleanup[{i}]={value}"
            );
        }
        for (i, value) in [
            (0, 2),
            (1, 1001),
            (2, 1001),
            (3, 99),
            (4, 99),
            (5, 99),
            (6, u64::MAX),
            (1399, 1),
        ] {
            let mut r = copy(&raw);
            r.trace[i] = value;
            assert!(
                matches!(observe(&p, l, &r), Err(Error::Malformed)),
                "trace[{i}]={value}"
            );
        }
        let mut r = copy(&raw);
        r.heap.pop();
        assert!(matches!(observe(&p, l, &r), Err(Error::Malformed)));
        let mut r = copy(&raw);
        r.cleanup.pop();
        assert!(matches!(observe(&p, l, &r), Err(Error::Malformed)));
        let mut r = copy(&raw);
        r.trace.pop();
        assert!(matches!(observe(&p, l, &r), Err(Error::Malformed)));
        // An extra live root and overlapping live extents are never observable.
        let mut r = copy(&raw);
        r.heap[3] = 1;
        assert!(matches!(observe(&p, l, &r), Err(Error::Malformed)));
    }
    #[test]
    fn issue_gnet_10_loaded_returning_faults_poison_and_retain_evidence() {
        for protocol in [0, 1, 2, 3] {
            let (mut p, l, _) = prepared();
            let dir = private_directory("gnet10-fault-library").unwrap();
            let cpp = dir.join("fault.cpp");
            let lib = dir.join(format!("fault.{}", std::env::consts::DLL_EXTENSION));
            fs::write(&cpp,format!(r#"#include <cstddef>
extern "C" unsigned int invar_fault_identity(unsigned char* p,std::size_t n){{if(n!=32)return 1;for(int i=0;i<32;i++)p[i]=0;return 0;}}
extern "C" unsigned int invar_fault(const long long*,std::size_t,long long*,std::size_t,unsigned long long*,std::size_t,long long*,std::size_t,unsigned long long*,std::size_t,unsigned long long,unsigned long long,unsigned long long){{return {protocol};}}
"#)).unwrap();
            let o = Command::new("c++")
                .args([
                    "-std=c++17",
                    "-fPIC",
                    if cfg!(target_os = "macos") {
                        "-dynamiclib"
                    } else {
                        "-shared"
                    },
                ])
                .arg(&cpp)
                .arg("-o")
                .arg(&lib)
                .output()
                .unwrap();
            assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
            p.driver = Driver::load(
                &lib,
                numeric::file_digest(&lib).unwrap(),
                [0; 32],
                "invar_fault",
                vec![],
                [p.storage.heap_words(), p.storage.cleanup_words()],
            )
            .unwrap();
            let failure = p.run(&[], l);
            if protocol == 0 {
                assert!(matches!(failure, Err(Error::Malformed)));
            } else {
                assert!(matches!(
                    failure,
                    Err(Error::Load {
                        cause: numeric::Error::Protocol(code),
                        ..
                    }) if code == protocol
                ));
            }
            assert!(matches!(p.run(&[], l), Err(Error::Closed)));
            let evidence = p.evidence_directory().to_owned();
            drop(p);
            assert!(evidence.is_dir());
            assert!(
                evidence
                    .join(if protocol == 0 {
                        "malformed-heap.bin"
                    } else {
                        "driver-failure.txt"
                    })
                    .is_file()
            );
            // Original generated artifacts and injected fault fixture both survive.
        }
    }
    #[test]
    fn issue_gnet_10_loaded_compiler_deadline_and_log_caps() {
        #[cfg(unix)]
        {
            let path = private_directory("gnet10-capture").unwrap();
            let r = bounded(
                Command::new("sh").args(["-c", "printf 'partial compiler log' >&2;exec sleep 10"]),
                &path.join("deadline"),
                Duration::from_millis(100),
            );
            assert!(r.unwrap_err().contains("deadline"));
            assert_eq!(
                fs::read(path.join("deadline.stderr")).unwrap(),
                b"partial compiler log"
            );
            let r = bounded(
                Command::new("head").args(["-c", "65537", "/dev/zero"]),
                &path.join("cap"),
                Duration::from_secs(1),
            );
            assert!(r.unwrap_err().contains("output limit"));
            assert_eq!(fs::metadata(path.join("cap.stdout")).unwrap().len(), 65536);
            // Preserve bounded failed-attempt evidence for inspection.
        }
    }
}
