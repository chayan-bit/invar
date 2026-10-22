//! Source-issued loaded ABI9 continuation execution with exclusive, opaque tasks.
//! Native state is never reconstructed from caller-supplied snapshot bytes.
use super::compiler_child::bounded;
use super::{compiled, coverage, numeric_host, oracle, provenance};
use crate::hex::hex;
use crate::{
    eval,
    front::{Span, Type},
};
use invar_foreign::{
    numeric,
    numeric_resume::{Driver, Snapshot},
};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    rc::Rc,
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
    Decode(numeric_host::Error),
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
/// Persistent thread-confined code mapping. Each Run exclusively borrows it.
/// ```compile_fail
/// fn send<T: Send>() {} send::<invar::general_net::numeric_resume_host::Plan>();
/// ```
/// ```compile_fail
/// fn sync<T: Sync>() {} sync::<invar::general_net::numeric_resume_host::Plan>();
/// ```
pub struct Plan {
    driver: Driver,
    artifact: provenance::ProvenancedArtifact,
    receipt: Json,
    sites: BTreeMap<TokenKey, (coverage::Rule, Span)>,
    frame_values: usize,
    state_functions: Vec<usize>,
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
    let source = artifact.source();
    let headers = source
        .lines()
        .filter(|l| l.starts_with("#include"))
        .collect::<Vec<_>>()
        .join("\n");
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
#include <cstring>
#include <new>
#include <type_traits>
{headers}
namespace {symbol}_private {{
{body}
static_assert(std::is_trivially_copyable<GnetTask>::value,"copyable task");
static_assert(offsetof(GnetTask,started)=={frame_bytes},"frame extent");
struct Owner {{
 GnetTask task{{}};
 gnet_w* input=nullptr;
 gnet_w output[8]{{LLONG_MIN,LLONG_MIN,LLONG_MIN,LLONG_MIN,LLONG_MIN,LLONG_MIN,LLONG_MIN,LLONG_MIN}};
 gnet_u* trace=nullptr;
 std::size_t input_n,trace_n;
 gnet_w heap[1]{{}}; gnet_u cleanup[1]{{}};
 gnet_u fuel,transitions,cap,calls=0;
 Owner(const gnet_w* in,std::size_t n,gnet_u f,gnet_u t,gnet_u c):input_n(n),trace_n(c*7),fuel(f),transitions(t),cap(c) {{
  input=new gnet_w[n?n:1]();
  try {{trace=new gnet_u[trace_n?trace_n:1]();}}catch(...){{delete[] input;throw;}}
  if(n)std::memcpy(input,in,n*8);
 }}
 ~Owner() noexcept {{delete[] trace;delete[] input;}}
}};
}}
#if defined(_WIN32)
#define INV_EXPORT __declspec(dllexport)
#else
#define INV_EXPORT __attribute__((visibility("default")))
#endif
extern "C" INV_EXPORT unsigned int {symbol}_identity(unsigned char* out,std::size_t n) noexcept {{
 if(!out||n!=32)return 2; const unsigned char id[32]={{{id}}};std::memcpy(out,id,32);return 0;
}}
extern "C" INV_EXPORT unsigned int {symbol}_create(const long long* input,std::size_t n,unsigned long long fuel,unsigned long long transitions,unsigned long long cap,void** handle) noexcept {{
 if(!handle)return 2;*handle=nullptr;
 if(!input||n!={params}||fuel==0||fuel>1000000||transitions>1000000||cap>1000000)return 2;
 try {{*handle=new {symbol}_private::Owner(input,n,fuel,transitions,cap);return 0;}}
 catch(const std::bad_alloc&){{return 1;}}catch(...){{return 3;}}
}}
extern "C" INV_EXPORT unsigned int {symbol}_resume(void* handle,unsigned long long quantum) noexcept {{
 if(!handle||quantum==0||quantum>1000000)return 2;
 try {{auto& o=*static_cast<{symbol}_private::Owner*>(handle);if(o.task.done)return 0;++o.calls;
 {symbol}_private::gnet_resume(o.input,o.output,o.trace,o.fuel,o.transitions,o.cap,o.heap,o.cleanup,&o.task,quantum);return 0;}}
 catch(const std::bad_alloc&){{return 1;}}catch(...){{return 3;}}
}}
extern "C" INV_EXPORT unsigned int {symbol}_capture(void* handle,long long* input,std::size_t n,long long* output,std::size_t out_n,unsigned long long* trace,std::size_t trace_n,unsigned char* task,std::size_t task_n,unsigned long long* counters,std::size_t counter_n,unsigned long long* calls) noexcept {{
 if(!handle||!input||!output||!trace||!task||!counters||!calls)return 2;
 auto& o=*static_cast<{symbol}_private::Owner*>(handle);
 if(n!=o.input_n||out_n!=8||trace_n!=o.trace_n||task_n!=sizeof(o.task)||counter_n!=18)return 2;
 if(n)std::memcpy(input,o.input,n*8);std::memcpy(output,o.output,64);
 if(trace_n)std::memcpy(trace,o.trace,trace_n*8);
 std::memcpy(task,&o.task,sizeof(o.task));
 const unsigned long long c[18]={{o.task.started,o.task.done,o.task.depth,o.task.max_depth,o.task.steps,o.task.transitions,o.task.used,o.task.next_frame,o.task.allocated,o.task.live,o.task.cleanup_used,o.task.fuel,o.task.transition_cap,o.task.trace_cap,(unsigned long long)o.task.status,(unsigned long long)o.task.result,(unsigned long long)o.task.start,(unsigned long long)o.task.end}};
 std::memcpy(counters,c,sizeof(c));*calls=o.calls;return 0;
}}
extern "C" INV_EXPORT unsigned int {symbol}_destroy(void* handle) noexcept {{
 if(!handle)return 2;try{{delete static_cast<{symbol}_private::Owner*>(handle);return 0;}}catch(...){{return 3;}}
}}
"#,
        params = artifact.params().len(),
        frame_bytes = artifact.frame_bytes()
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
        let artifact =
            provenance::compile_resumable_numeric(source, entry, compiled::Target::Cpu, limits)
                .map_err(Error::Provenance)?;
        if artifact.artifact().abi_version() != 9
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
                .join(format!("invar-gnet9-host-{}-{nonce}", std::process::id())),
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
        let prelink = json!({"version":1,"abi":9,"artifact":artifact.id(),"compiler_sha256":hex(&compiler_hash),"compiler_path":compiler,"compiler_version":hex(&hash(&version_out)),"compiler_version_stderr":hex(&hash(&version_err)),"options":OPTIONS,"shared":if cfg!(target_os="macos"){"-dynamiclib"}else{"-shared"},"host_build":env!("INVAR_BUILD_DIGEST"),"frame_bound":MAX_FRAMES,"task_bytes":artifact.artifact().continuation_layout().ok_or(Error::Limit)?.task_bytes(),"config_words":5,"output_words":8,"task_counters":18,"params":artifact.artifact().params().iter().map(|t|format!("{t:?}")).collect::<Vec<_>>(),"source_bound":MAX_SOURCE,"library_bound":64*1024*1024});
        let identity = hash(prelink.to_string().as_bytes());
        let symbol = format!("invar_{}_resume_v1", hex(&identity));
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
        let receipt = json!({"scope":"source_issued_abi9_loaded_cpu","artifact":artifact.manifest(),"prelink":prelink,"identity":hex(&identity),"wrapper_sha256":hex(&hash(text.as_bytes())),"wrapper_bytes":text.len(),"library_sha256":hex(&library_hash),"library_bytes":fs::metadata(&library).map_err(io)?.len(),"symbol":symbol,"bounds":{"fuel":1000000,"transitions":1000000,"trace_events":1000000,"complete_frames_bytes":MAX_FRAMES}});
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
            artifact
                .artifact()
                .continuation_layout()
                .ok_or(Error::Limit)?
                .task_bytes(),
        )
        .map_err(|cause| Error::Load {
            cause,
            evidence: directory.path.clone(),
        })?;
        let (net, proof) = artifact.checked_net();
        let program = net
            .compiled_mir(proof)
            .map_err(|e| Error::Provenance(provenance::Error::Net(e)))?;
        // ABI9 assigns one PC to each instruction and terminator, in MIR order.
        // Derive the exact dense frame and PC ranges from the same checked MIR.
        let mut state_functions = reserve(artifact.artifact().states())?;
        let mut frame_values = 1;
        let mut frame_places = 1;
        for (function_id, function) in program.functions.iter().enumerate() {
            let mut values = BTreeSet::new();
            for block in &function.blocks {
                for (id, _) in block
                    .params
                    .iter()
                    .chain(block.instructions.iter().filter_map(|i| i.output.as_ref()))
                {
                    values.insert(*id);
                }
                for _ in 0..=block.instructions.len() {
                    state_functions.push(function_id);
                }
            }
            frame_values = frame_values.max(values.len());
            frame_places = frame_places.max(function.locals.len());
        }
        if state_functions.len() != artifact.artifact().states()
            || (frame_values + frame_places + 4) * 64 * 8 != artifact.artifact().frame_bytes()
        {
            return Err(Error::Malformed);
        }
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
            frame_values,
            state_functions,
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
    pub fn start(
        &mut self,
        args: &[eval::Value],
        limits: oracle::Limits,
    ) -> Result<Run<'_>, Error> {
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
        if let Err(cause) = self.driver.start(
            &input,
            [limits.fuel, limits.transitions, limits.trace_events as u64],
        ) {
            self.closed = true;
            self.directory.remove = false;
            let _ = fs::write(self.directory.path.join("failure-start.json"), json!({"input":input,"limits":[limits.fuel,limits.transitions,limits.trace_events],"cause":format!("{cause:?}")}).to_string());
            return Err(Error::Loader(cause));
        }
        let snapshot = match self.driver.snapshot() {
            Ok(s) => s,
            Err(e) => {
                self.closed = true;
                self.directory.remove = false;
                let _ = fs::write(self.directory.path.join("failure-start.json"), json!({"input":input,"limits":[limits.fuel,limits.transitions,limits.trace_events],"cause":format!("{e:?}"),"snapshot_observed":false}).to_string());
                let _ = self.driver.finish();
                return Err(Error::Loader(e));
            }
        };
        if snapshot.input != input
            || snapshot.output != [i64::MIN; 8]
            || snapshot.counters != [0; 18]
            || snapshot.task.iter().any(|b| *b != 0)
            || snapshot.trace.iter().any(|b| *b != 0)
            || snapshot.native_calls != 0
        {
            retain(self, &snapshot, "initial task storage");
            let _ = self.driver.finish();
            return Err(Error::Malformed);
        }
        Ok(Run {
            plan: self,
            limits,
            snapshot,
            terminal: None,
            failed: false,
        })
    }
}
/// Safe counters from a yielded continuation; no partial source result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pending {
    pub steps: u64,
    pub transitions: u64,
    pub events: usize,
    pub depth: usize,
    pub max_frames: usize,
}
#[derive(Debug)]
pub enum Progress {
    Pending(Pending),
    Complete(Rc<oracle::Observation>),
}
/// Exclusively borrows a plan. Dropping destroys its native task before unmapping.
/// ```compile_fail
/// fn send<T: Send>() {} send::<invar::general_net::numeric_resume_host::Run<'static>>();
/// ```
/// ```compile_fail
/// fn sync<T: Sync>() {} sync::<invar::general_net::numeric_resume_host::Run<'static>>();
/// ```
pub struct Run<'a> {
    plan: &'a mut Plan,
    limits: oracle::Limits,
    snapshot: Snapshot,
    terminal: Option<Rc<oracle::Observation>>,
    failed: bool,
}
impl Run<'_> {
    pub fn resume(&mut self, quantum: u64) -> Result<Progress, Error> {
        if quantum == 0 || quantum > 1_000_000 {
            return Err(Error::Limit);
        }
        if self.failed {
            return Err(Error::Closed);
        }
        if let Some(observation) = &self.terminal {
            return Ok(Progress::Complete(Rc::clone(observation)));
        }
        let snapshot = match self.plan.driver.resume(quantum) {
            Ok(s) => s,
            Err(e) => {
                self.failed = true;
                if let Ok(s) = self.plan.driver.snapshot() {
                    retain(self.plan, &s, "native resume/capture");
                } else {
                    retain(
                        self.plan,
                        &self.snapshot,
                        "native resume/capture (last validated snapshot)",
                    );
                }
                return Err(Error::Loader(e));
            }
        };
        if let Err(error) = validate(self.plan, self.limits, &self.snapshot, &snapshot, quantum) {
            retain(self.plan, &snapshot, "task/prefix validation");
            self.failed = true;
            return Err(error);
        }
        self.snapshot = snapshot;
        if self.snapshot.counters[1] == 0 {
            let c = self.snapshot.counters;
            return Ok(Progress::Pending(Pending {
                steps: c[4],
                transitions: c[5],
                events: c[6] as usize,
                depth: c[2] as usize,
                max_frames: c[3] as usize,
            }));
        }
        // Reuse the qualified ABI8 terminal decoder only on actual completed ABI9
        // output. Pending prefixes never fabricate an observation or Return.
        let raw = numeric::Invocation {
            output: self.snapshot.output,
            trace: std::mem::take(&mut self.snapshot.trace),
        };
        let decoded = numeric_host::decode(
            self.plan.artifact.artifact().result(),
            self.plan.artifact.source().len(),
            &self.plan.sites,
            self.limits,
            &raw,
        );
        self.snapshot.trace = raw.trace;
        match decoded {
            Ok(observation) => {
                let observation = Rc::new(observation);
                self.terminal = Some(Rc::clone(&observation));
                Ok(Progress::Complete(observation))
            }
            Err(e) => {
                retain(self.plan, &self.snapshot, "terminal decoder");
                self.failed = true;
                Err(Error::Decode(e))
            }
        }
    }
}
impl Drop for Run<'_> {
    fn drop(&mut self) {
        if self.plan.driver.finish().is_err() {
            retain(self.plan, &self.snapshot, "native destruction");
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
fn retain(plan: &mut Plan, snapshot: &Snapshot, reason: &str) {
    plan.closed = true;
    plan.directory.remove = false;
    let path = &plan.directory.path;
    let _ = fs::write(path.join("failure.json"), json!({"reason":reason,"limits":[snapshot.counters[11],snapshot.counters[12],snapshot.counters[13]],"counters":snapshot.counters,"native_calls":snapshot.native_calls}).to_string());
    let _ = fs::write(path.join("failure-task.bin"), &snapshot.task);
    for (name, words) in [
        (
            "failure-input.bin",
            snapshot.input.iter().map(|w| *w as u64).collect::<Vec<_>>(),
        ),
        (
            "failure-output.bin",
            snapshot.output.iter().map(|w| *w as u64).collect(),
        ),
    ] {
        write_words(&path.join(name), &words);
    }
    write_words(&path.join("failure-trace.bin"), &snapshot.trace);
}
fn write_words(path: &Path, words: &[u64]) {
    if let Ok(file) = fs::File::create(path) {
        let mut out = std::io::BufWriter::new(file);
        for word in words {
            if out.write_all(&word.to_le_bytes()).is_err() {
                break;
            }
        }
    }
}
fn validate(
    plan: &Plan,
    l: oracle::Limits,
    old: &Snapshot,
    new: &Snapshot,
    quantum: u64,
) -> Result<(), Error> {
    let c = new.counters;
    let p = old.counters;
    let artifact = plan.artifact.artifact();
    let frame_bytes = artifact.frame_bytes();
    let expected_task = artifact
        .continuation_layout()
        .ok_or(Error::Malformed)?
        .task_bytes();
    if new.task.len() != expected_task
        || expected_task != frame_bytes + 18 * 8
        || new.input != old.input
        || new.trace.len() != l.trace_events * 7
        || c[0] != 1
        || c[1] > 1
        || c[2] > 64
        || c[3] == 0
        || c[3] > 64
        || c[2] > c[3]
        || c[4] > l.fuel
        || c[5] > l.transitions
        || c[6] > l.trace_events as u64
        || c[7] == 0
        || c[7] > c[5] + 1
        || c[8..11] != [0, 0, 0]
        || c[11..14] != [l.fuel, l.transitions, l.trace_events as u64]
        || c[14] > 6
        || (c[1] == 0 && (c[2] == 0 || c[14..18] != [0, 0, 0, 0]))
        || (c[1] == 1 && c[14] == 0 && c[2] != 0)
        || new.native_calls != old.native_calls + 1
        || c[4] < p[4]
        || c[5] < p[5]
        || c[6] < p[6]
        || c[7] < p[7]
        || c[3] < p[3]
        || c[5] - p[5] > quantum
        || (c[1] == 0 && c[5] - p[5] != quantum)
    {
        return Err(Error::Malformed);
    }
    for (bytes, word) in new.task[frame_bytes..].chunks_exact(8).zip(c) {
        if u64::from_ne_bytes(bytes.try_into().map_err(|_| Error::Malformed)?) != word {
            return Err(Error::Malformed);
        }
    }
    let frame_len = frame_bytes / 64;
    if !frame_bytes.is_multiple_of(64) || frame_len < 32 || !frame_len.is_multiple_of(8) {
        return Err(Error::Malformed);
    }
    let mut last_id = None;
    let mut active = reserve(64)?;
    for frame in new.task[..frame_bytes]
        .chunks_exact(frame_len)
        .take(c[2] as usize)
    {
        let mut m = [0; 4];
        for (slot, bytes) in m.iter_mut().zip(frame[frame_len - 32..].chunks_exact(8)) {
            *slot = u64::from_ne_bytes(bytes.try_into().map_err(|_| Error::Malformed)?);
        }
        if m[0] >= artifact.states() as u64
            || m[1] > plan.frame_values as u64
            || m[2] >= c[7]
            || m[3] > c[5]
            || last_id.is_some_and(|id| m[2] <= id)
            || (last_id.is_none() && m[2] != 0)
        {
            return Err(Error::Malformed);
        }
        active.push((m[2], m[3], plan.state_functions[m[0] as usize]));
        last_id = Some(m[2]);
    }
    let used = c[6] as usize * 7;
    let previous = p[6] as usize * 7;
    if previous > old.trace.len()
        || new.trace[..previous] != old.trace[..previous]
        || new.trace[used..].iter().any(|w| *w != 0)
    {
        return Err(Error::Malformed);
    }
    let mut stack: Vec<(u64, u64, TokenKey, coverage::Rule)> = reserve(64)?;
    let mut history = reserve(c[7] as usize)?;
    history.resize(
        c[7] as usize,
        None::<(u64, usize, usize, coverage::Rule, bool)>,
    );
    for row in new.trace[..used].chunks_exact(7) {
        let key = (
            usize::try_from(row[3]).map_err(|_| Error::Malformed)?,
            usize::try_from(row[4]).map_err(|_| Error::Malformed)?,
            usize::try_from(row[5]).map_err(|_| Error::Malformed)?,
        );
        let (rule, _) = plan.sites.get(&key).ok_or(Error::Malformed)?;
        if row[0] > 1
            || row[1] >= c[7]
            || row[1] > c[5]
            || row[2] > c[5]
            || row[6] != *rule as u64
            || *rule == coverage::Rule::Tick
        {
            return Err(Error::Malformed);
        }
        let last = &mut history[row[1] as usize];
        if let Some((activation, function, block, previous_rule, complete)) = *last {
            let expected = activation
                + u64::from(
                    complete
                        && matches!(previous_rule, coverage::Rule::Goto | coverage::Rule::Branch),
                );
            if row[2] != expected
                || key.0 != function
                || (row[2] == activation && key.1 != block)
                || (complete && previous_rule == coverage::Rule::Return)
            {
                return Err(Error::Malformed);
            }
        } else if row[2] != 0 {
            return Err(Error::Malformed);
        }
        *last = Some((row[2], key.0, key.1, *rule, row[0] == 1));
        let event = (row[1], row[2], key, *rule);
        if row[0] == 0 {
            if stack.len() == 64
                || stack
                    .last()
                    .is_some_and(|parent| parent.3 != coverage::Rule::Call || row[1] <= parent.0)
            {
                return Err(Error::Malformed);
            }
            stack.push(event);
        } else if stack.pop() != Some(event) {
            return Err(Error::Malformed);
        }
    }
    for (frame, activation, function) in active {
        if let Some((last_activation, last_function, _, rule, complete)) = history[frame as usize] {
            let expected = last_activation
                + u64::from(
                    complete && matches!(rule, coverage::Rule::Goto | coverage::Rule::Branch),
                );
            if activation != expected || function != last_function {
                return Err(Error::Malformed);
            }
        } else if activation != 0 {
            return Err(Error::Malformed);
        }
    }
    if c[1] == 0 {
        if new.output != [i64::MIN; 8]
            || stack.iter().any(|e| e.3 != coverage::Rule::Call)
            || stack.len() + 1 != c[2] as usize
        {
            return Err(Error::Malformed);
        }
    } else {
        let o = new.output;
        if [
            o[0] as u64,
            o[1] as u64,
            o[2] as u64,
            o[3] as u64,
            o[4] as u64,
            o[5] as u64,
            o[6] as u64,
            o[7] as u64,
        ] != [c[14], c[15], c[16], c[17], c[4], c[5], c[3], c[6]]
        {
            return Err(Error::Malformed);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const SOURCE: &[u8] = b"fn recur(a:f32,b:f32,c:f32,n:u32)->f32{if n==0u32{return a;}return recur(fma(a,b,c),b,c,n-1u32);}fn run(a:f32,b:f32,c:f32,n:u32)->f32{var x=a;for i in 0u32..n{x=recur(x,b,c,2u32);}return x;}";
    fn plan() -> Plan {
        Plan::from_source(SOURCE, "run", Default::default(), Path::new("c++")).unwrap()
    }
    fn limits() -> oracle::Limits {
        oracle::Limits {
            fuel: 1000,
            transitions: 10000,
            trace_events: 1000,
            ..Default::default()
        }
    }
    fn args() -> [eval::Value; 4] {
        use crate::numeric_contract::F32Bits;
        [
            eval::Value::F32(F32Bits::from_bits(0x3f800001)),
            eval::Value::F32(F32Bits::from_bits(0x3f7ffffe)),
            eval::Value::F32(F32Bits::from_bits(0xbf800000)),
            eval::Value::U32(3),
        ]
    }
    #[test]
    fn issue_gnet_09_numeric_host_exact_task_and_prefix_at_each_quantum() {
        let mut p = plan();
        let l = limits();
        let a = args();
        let (net, proof) = p.artifact.checked_net();
        let expected = oracle::observe_call(net, proof, "run", &a, l).unwrap();
        let expected_words: Vec<_> = expected
            .events
            .iter()
            .flat_map(|e| {
                [
                    u64::from(e.kind == oracle::EventKind::Complete),
                    e.frame,
                    e.activation,
                    e.token.block.function as u64,
                    e.token.block.block as u64,
                    e.token.position as u64,
                    e.rule as u64,
                ]
            })
            .collect();
        let mut states = BTreeMap::new();
        let mut yields = 0;
        let mut matches = 0;
        for quantum in [1, 3, 17] {
            let mut r = p.start(&a, l).unwrap();
            loop {
                let progress = r.resume(quantum).unwrap();
                let s = &r.snapshot;
                let key = (s.counters[5], s.counters[1]);
                let used = s.counters[6] as usize * 7;
                assert_eq!(&s.trace[..used], &expected_words[..used]);
                assert_eq!(s.counters[8..11], [0, 0, 0]);
                assert_eq!(
                    s.counters[11..14],
                    [l.fuel, l.transitions, l.trace_events as u64]
                );
                if s.native_calls == 1 && s.counters[1] == 0 {
                    let before = r.plan.driver.snapshot().unwrap();
                    for invalid in [0, 1_000_001, u64::MAX] {
                        assert!(matches!(r.resume(invalid), Err(Error::Limit)));
                    }
                    let after = r.plan.driver.snapshot().unwrap();
                    assert_eq!(before.task, after.task);
                    assert_eq!(before.trace, after.trace);
                    assert_eq!(before.output, after.output);
                    assert_eq!(before.native_calls, after.native_calls);
                }
                let s = &r.snapshot;
                if quantum == 1 {
                    states.insert(key, (s.task.clone(), s.trace.clone(), s.output));
                } else if let Some((task, trace, output)) = states.get(&key) {
                    assert_eq!(&s.task, task);
                    assert_eq!(&s.trace, trace);
                    assert_eq!(&s.output, output);
                    matches += 1;
                }
                match progress {
                    Progress::Pending(_) => {
                        assert_eq!(s.output, [i64::MIN; 8]);
                        yields += 1;
                    }
                    Progress::Complete(o) => {
                        assert_eq!(o.events, expected.events);
                        assert_eq!(o.transitions, expected.transitions);
                        let calls = r.snapshot.native_calls;
                        for q in [1, 3, 17, 1_000_000] {
                            let Progress::Complete(replay) = r.resume(q).unwrap() else {
                                panic!("replay yielded")
                            };
                            assert!(Rc::ptr_eq(&o, &replay));
                            assert_eq!(r.plan.driver.snapshot().unwrap().native_calls, calls);
                        }
                        break;
                    }
                }
            }
        }
        assert!(yields > 100);
        assert!(matches > 20);
        eprintln!(
            "actual ABI9 exact task/prefix yields={yields}, matching cross-quantum states={matches}"
        );
    }
    fn sync_task(s: &mut Snapshot, frame_bytes: usize) {
        for (bytes, counter) in s.task[frame_bytes..].chunks_exact_mut(8).zip(s.counters) {
            bytes.copy_from_slice(&counter.to_ne_bytes());
        }
    }
    #[test]
    fn issue_gnet_09_numeric_host_malformed_pending_task_events_and_output_refused() {
        let mut p = plan();
        let a = args();
        let l = limits();
        p.driver
            .start(
                &encode(p.artifact.artifact().params(), &a).unwrap(),
                [l.fuel, l.transitions, l.trace_events as u64],
            )
            .unwrap();
        let old = p.driver.snapshot().unwrap();
        let good = p.driver.resume(3).unwrap();
        validate(&p, l, &old, &good, 3).unwrap();
        let frame_bytes = p.artifact.artifact().frame_bytes();
        for mutation in 0..24 {
            let mut s = p.driver.snapshot().unwrap();
            match mutation {
                0 => s.counters[0] = 0,
                1 => s.counters[1] = 2,
                2 => s.counters[2] = 65,
                3 => s.counters[3] = 0,
                4 => s.counters[4] = l.fuel + 1,
                5 => s.counters[5] = l.transitions + 1,
                6 => s.counters[6] = l.trace_events as u64 + 1,
                7 => s.counters[7] = s.counters[5] + 2,
                8 => s.counters[8] = 1,
                9 => s.counters[9] = 1,
                10 => s.counters[10] = 1,
                11 => s.counters[11] += 1,
                12 => s.counters[12] += 1,
                13 => s.counters[13] += 1,
                14 => s.counters[14] = 8,
                15 => s.counters[15] = 1,
                16 => s.counters[16] = 1,
                17 => s.counters[17] = 1,
                18 => s.output[1] = 0,
                19 => s.trace[0] = 2,
                20 => s.trace[3] = u64::MAX,
                21 => s.trace[6] = u64::MAX,
                22 => {
                    let n = s.trace.len();
                    s.trace[n - 1] = 1;
                }
                23 => s.input[0] ^= 1,
                _ => unreachable!(),
            }
            sync_task(&mut s, frame_bytes);
            assert!(
                matches!(validate(&p, l, &old, &s, 3), Err(Error::Malformed)),
                "mutation {mutation}"
            );
        }
        for part in [0, 1, 2, 3] {
            let mut s = p.driver.snapshot().unwrap();
            let frame_len = frame_bytes / 64;
            let begin = frame_len - 32 + part * 8;
            s.task[begin..begin + 8].copy_from_slice(&u64::MAX.to_ne_bytes());
            assert!(matches!(
                validate(&p, l, &old, &s, 3),
                Err(Error::Malformed)
            ));
        }
        let old = good;
        let good = p.driver.resume(3).unwrap();
        validate(&p, l, &old, &good, 3).unwrap();
        let mut changed = p.driver.snapshot().unwrap();
        changed.trace[1] = 1;
        assert!(matches!(
            validate(&p, l, &old, &changed, 3),
            Err(Error::Malformed)
        ));
        retain(
            &mut p,
            &changed,
            "deliberate malformed pending packet qualification",
        );
        eprintln!(
            "retained malformed packet {}",
            p.evidence_directory().display()
        );
        p.driver.finish().unwrap();
        assert!(matches!(p.start(&a, l), Err(Error::Closed)));
    }
    #[test]
    fn issue_gnet_09_numeric_host_terminal_task_typed_word_status_and_pairing_refused() {
        let mut p = Plan::from_source(
            b"fn run(x:Bool)->Bool{return x;}",
            "run",
            Default::default(),
            Path::new("c++"),
        )
        .unwrap();
        let l = limits();
        let mut r = p.start(&[eval::Value::Bool(true)], l).unwrap();
        let old = r.plan.driver.snapshot().unwrap();
        let Progress::Complete(_) = r.resume(17).unwrap() else {
            panic!("unexpected pending")
        };
        for mutation in 0..8 {
            let mut s = r.plan.driver.snapshot().unwrap();
            match mutation {
                0 => {
                    s.output[0] = 8;
                    s.counters[14] = 8;
                }
                1 => {
                    s.output[1] = 2;
                    s.counters[15] = 2;
                }
                2 => {
                    s.output[2] = 1;
                    s.counters[16] = 1;
                }
                3 => {
                    s.output[7] = 0;
                    s.counters[6] = 0;
                    s.trace.fill(0);
                }
                4 => s.trace[0] = 1,
                5 => {
                    s.counters[2] = 1;
                }
                6 => s.task.pop().map(|_| ()).unwrap(),
                7 => s.native_calls += 1,
                _ => unreachable!(),
            }
            if mutation != 6 {
                sync_task(&mut s, r.plan.artifact.artifact().frame_bytes());
            }
            let accepted = validate(r.plan, l, &old, &s, 17).and_then(|_| {
                numeric_host::decode(
                    &Type::Bool,
                    r.plan.artifact.source().len(),
                    &r.plan.sites,
                    l,
                    &numeric::Invocation {
                        output: s.output,
                        trace: s.trace,
                    },
                )
                .map_err(Error::Decode)
            });
            assert!(accepted.is_err(), "terminal mutation {mutation}");
        }
    }
    #[cfg(unix)]
    #[test]
    fn issue_gnet_09_numeric_host_compiler_caps_deadline_and_descendant_cleanup() {
        let path = std::env::temp_dir().join(format!("gnet9-capture-{}", std::process::id()));
        fs::create_dir(&path).unwrap();
        let result=bounded(
            Command::new("sh").args(["-c","printf 'original ABI9 partial log' >&2;sleep 10 & echo $! > \"$INVAR_GNET9_CHILD_PID\";exit 0"]).env("INVAR_GNET9_CHILD_PID",path.join("child.pid")),
            &path.join("deadline"),Duration::from_millis(100),
        );
        assert!(result.unwrap_err().contains("deadline"));
        assert_eq!(
            fs::read(path.join("deadline.stderr")).unwrap(),
            b"original ABI9 partial log"
        );
        #[cfg(target_os = "linux")]
        {
            let pid = fs::read_to_string(path.join("child.pid")).unwrap();
            let proc = PathBuf::from(format!("/proc/{}/stat", pid.trim()));
            let begun = std::time::Instant::now();
            loop {
                match fs::read_to_string(&proc) {
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => break,
                    Ok(stat)
                        if stat
                            .split(") ")
                            .nth(1)
                            .is_some_and(|tail| tail.starts_with('Z')) =>
                    {
                        break;
                    }
                    _ => {
                        assert!(
                            begun.elapsed() < Duration::from_secs(2),
                            "compiler descendant stayed live"
                        );
                        std::thread::yield_now();
                    }
                }
            }
        }
        let result = bounded(
            Command::new("head").args(["-c", "65537", "/dev/zero"]),
            &path.join("cap"),
            Duration::from_secs(1),
        );
        assert!(result.unwrap_err().contains("output limit"));
        assert_eq!(fs::metadata(path.join("cap.stdout")).unwrap().len(), 65536);
        eprintln!("retained capped/deadline compiler logs {}", path.display());
    }
}
