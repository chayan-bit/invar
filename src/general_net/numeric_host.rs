//! Source-derived, synchronous loaded CPU ABI8 execution. The independent net
//! oracle is a comparator only; its interpreter never supplies this result.
use super::compiler_child::bounded;
use super::{BlockSite, Token, compiled, coverage, oracle, provenance};
use crate::hex::hex;
use crate::{
    eval,
    front::{Span, Type},
    numeric_contract::F32Bits,
};
use invar_foreign::numeric::{self, Driver};
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
pub struct Plan {
    driver: Driver,
    artifact: provenance::ProvenancedArtifact,
    receipt: Json,
    sites: BTreeMap<TokenKey, (coverage::Rule, Span)>,
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
extern "C" INV_EXPORT unsigned int {symbol}(const long long* input,std::size_t n,long long* output,std::size_t out_n,unsigned long long* trace,std::size_t trace_n,unsigned long long fuel,unsigned long long transitions,unsigned long long cap) noexcept {{
 if(n!={params}||!input||!output||out_n!=8||!trace||fuel==0||fuel>1000000||transitions>1000000||cap>1000000||trace_n!=cap*7) return 2;
 try {{ {symbol}_private::gnet_run(input,output,trace,fuel,transitions,cap);return 0; }}
 catch(const std::bad_alloc&) {{return 1;}} catch(...) {{return 3;}}
}}
"#,
        params = artifact.params().len()
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
        compiler: &Path,
    ) -> Result<Self, Error> {
        if limits.compiler.frame_bytes > MAX_FRAMES {
            return Err(Error::Limit);
        }
        let artifact = provenance::compile_numeric(source, entry, compiled::Target::Cpu, limits)
            .map_err(Error::Provenance)?;
        if artifact.artifact().abi_version() != 8
            || artifact.artifact().frame_bytes() > MAX_FRAMES
            || artifact.artifact().params().len() > 128
        {
            return Err(Error::Limit);
        }
        let compiler = compiler_path(compiler)?;
        let compiler_hash = numeric::file_digest(&compiler).map_err(Error::Loader)?;
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let nonce = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| Error::Limit)?;
        let mut directory = Directory {
            path: std::env::temp_dir()
                .join(format!("invar-gnet8-host-{}-{nonce}", std::process::id())),
            remove: false,
        };
        fs::create_dir(&directory.path).map_err(io)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&directory.path, fs::Permissions::from_mode(0o700)).map_err(io)?;
        }
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
        let prelink = json!({"version":1,"abi":8,"artifact":artifact.id(),"compiler_sha256":hex(&compiler_hash),"compiler_path":compiler,"compiler_version":hex(&hash(&version_out)),"compiler_version_stderr":hex(&hash(&version_err)),"options":OPTIONS,"shared":if cfg!(target_os="macos"){"-dynamiclib"}else{"-shared"},"host_build":env!("INVAR_BUILD_DIGEST"),"frame_bound":MAX_FRAMES,"source_bound":MAX_SOURCE,"library_bound":64*1024*1024});
        let identity = hash(prelink.to_string().as_bytes());
        let symbol = format!("invar_{}_numeric_v1", hex(&identity));
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
        let receipt = json!({"scope":"source_issued_abi8_loaded_cpu","artifact":artifact.manifest(),"prelink":prelink,"identity":hex(&identity),"wrapper_sha256":hex(&hash(text.as_bytes())),"wrapper_bytes":text.len(),"library_sha256":hex(&library_hash),"library_bytes":fs::metadata(&library).map_err(io)?.len(),"symbol":symbol,"bounds":{"fuel":1000000,"transitions":1000000,"trace_events":1000000,"complete_frames_bytes":MAX_FRAMES}});
        fs::write(directory.path.join("receipt.json"), receipt.to_string()).map_err(io)?;
        let mut carriers = reserve(artifact.artifact().params().len())?;
        for ty in artifact.artifact().params() {
            carriers.push(match ty {
                Type::I32 => numeric::Carrier::I32,
                Type::Bool => numeric::Carrier::Bool,
                _ => numeric::Carrier::Word,
            });
        }
        let driver =
            Driver::load(&library, library_hash, identity, &symbol, carriers).map_err(|cause| {
                Error::Load {
                    cause,
                    evidence: directory.path.clone(),
                }
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
        directory.remove = true;
        Ok(Self {
            driver,
            artifact,
            receipt,
            sites,
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
        let input = encode(self.artifact.artifact().params(), args)?;
        let words = self
            .driver
            .run(
                &input,
                [limits.fuel, limits.transitions, limits.trace_events as u64],
            )
            .map_err(Error::Loader)?;
        match decode(
            self.artifact.artifact().result(),
            self.artifact.source().len(),
            &self.sites,
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
    if status > 7
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
    if (status != 1 && status != 2 && (start != 0 || end != 0))
        || (matches!(status, 1 | 2) && start == end)
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
    if matches!(status, 1 | 2) {
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
        _ => None,
    };
    let outcome = match status {
        0 => Ok(match result {
            Type::F32 => eval::Value::F32(F32Bits::from_bits(
                u32::try_from(h[1]).map_err(|_| Error::Malformed)?,
            )),
            Type::I32 => eval::Value::I32(i32::try_from(h[1]).map_err(|_| Error::Malformed)?),
            Type::U32 => eval::Value::U32(u32::try_from(h[1]).map_err(|_| Error::Malformed)?),
            Type::Bool if h[1] == 0 || h[1] == 1 => eval::Value::Bool(h[1] != 0),
            Type::Unit if h[1] == 0 => eval::Value::Unit,
            _ => return Err(Error::Malformed),
        }),
        1 | 2 => Err(eval::EvalError::Trap {
            category: if status == 1 {
                eval::TrapCategory::IntegerOverflow
            } else {
                eval::TrapCategory::DivisionByZero
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
            message: "compiled ABI8 call-depth limit".into(),
        }),
        5 | 6 => Err(eval::EvalError::Tool {
            code: "RESOURCE_LIMIT",
            message: "compiled ABI8 transition/trace limit".into(),
        }),
        7 => Err(eval::EvalError::Tool {
            code: "MIR_INVALID",
            message: "compiled ABI8 invalid state".into(),
        }),
        _ => return Err(Error::Malformed),
    };
    Ok(oracle::Observation {
        semantic: eval::Observation {
            result: outcome,
            steps_used: steps,
            cleanup_trace: Vec::new(),
        },
        failure,
        events,
        transitions,
        max_frames: frames as usize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn packet() -> (
        BTreeMap<TokenKey, (coverage::Rule, Span)>,
        oracle::Limits,
        numeric::Invocation,
    ) {
        let sites = BTreeMap::from([(
            (0, 0, 0),
            (coverage::Rule::Return, Span { start: 1, end: 3 }),
        )]);
        let l = oracle::Limits {
            fuel: 10,
            transitions: 10,
            trace_events: 3,
            ..Default::default()
        };
        let raw = numeric::Invocation {
            output: [0, 1, 0, 0, 0, 1, 1, 2],
            trace: vec![
                0,
                0,
                0,
                0,
                0,
                0,
                coverage::Rule::Return as u64,
                1,
                0,
                0,
                0,
                0,
                0,
                coverage::Rule::Return as u64,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        };
        (sites, l, raw)
    }
    #[test]
    fn issue_gnet_08_numeric_host_malformed_status_span_count_carrier_and_event_guards() {
        let (sites, l, raw) = packet();
        decode(&Type::F32, 30, &sites, l, &raw).unwrap();
        for (index, value) in [
            (0, -1),
            (0, 8),
            (0, 9),
            (1, -1),
            (1, 4294967296),
            (2, 1),
            (3, 31),
            (4, 11),
            (5, 11),
            (6, 0),
            (6, 65),
            (7, 4),
            (7, -1),
        ] {
            let mut changed = numeric::Invocation {
                output: raw.output,
                trace: raw.trace.clone(),
            };
            changed.output[index] = value;
            assert!(
                matches!(
                    decode(&Type::F32, 30, &sites, l, &changed),
                    Err(Error::Malformed)
                ),
                "header[{index}]={value}"
            );
        }
        for (index, value) in [
            (0, 2),
            (1, 11),
            (2, 11),
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 0),
            (7, 0),
            (8, 1),
            (14, 1),
        ] {
            let mut changed = numeric::Invocation {
                output: raw.output,
                trace: raw.trace.clone(),
            };
            changed.trace[index] = value;
            assert!(
                matches!(
                    decode(&Type::F32, 30, &sites, l, &changed),
                    Err(Error::Malformed)
                ),
                "trace[{index}]={value}"
            );
        }
        let mut short = numeric::Invocation {
            output: raw.output,
            trace: raw.trace.clone(),
        };
        short.trace.pop();
        assert!(matches!(
            decode(&Type::F32, 30, &sites, l, &short),
            Err(Error::Malformed)
        ));
        let mut invalid = numeric::Invocation {
            output: raw.output,
            trace: raw.trace.clone(),
        };
        invalid.output[1] = 2;
        assert!(matches!(
            decode(&Type::Bool, 30, &sites, l, &invalid),
            Err(Error::Malformed)
        ));
        assert!(matches!(
            decode(&Type::Unit, 30, &sites, l, &invalid),
            Err(Error::Malformed)
        ));
        let mut trap = numeric::Invocation {
            output: [2, 0, 1, 3, 1, 1, 1, 1],
            trace: vec![0; 21],
        };
        trap.trace[6] = coverage::Rule::Binary as u64;
        let sites = BTreeMap::from([(
            (0, 0, 0),
            (coverage::Rule::Binary, Span { start: 1, end: 3 }),
        )]);
        decode(&Type::F32, 30, &sites, l, &trap).unwrap();
        trap.output[2] = 2;
        assert!(matches!(
            decode(&Type::F32, 30, &sites, l, &trap),
            Err(Error::Malformed)
        ));
    }
    #[test]
    fn issue_gnet_08_numeric_host_compiler_caps_and_deadline_keep_partial_logs() {
        #[cfg(unix)]
        {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "gnet8-capture-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            let result = bounded(
                Command::new("sh").args(["-c", "printf 'original partial log' >&2;exec sleep 10"]),
                &path.join("deadline"),
                Duration::from_millis(50),
            );
            assert!(result.unwrap_err().contains("deadline"));
            assert_eq!(
                fs::read(path.join("deadline.stderr")).unwrap(),
                b"original partial log"
            );
            let result = bounded(
                Command::new("head").args(["-c", "65537", "/dev/zero"]),
                &path.join("cap"),
                Duration::from_secs(1),
            );
            assert!(result.unwrap_err().contains("output limit"));
            assert_eq!(fs::metadata(path.join("cap.stdout")).unwrap().len(), 65536);
            fs::remove_dir_all(path).unwrap();
        }
    }
}
