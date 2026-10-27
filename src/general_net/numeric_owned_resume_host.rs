//! Source-issued loaded CPU ABI11 owned numeric continuation with an exclusive
//! opaque native task. Snapshots are validation evidence only: native state is
//! never imported or restored from caller bytes. Terminal packets go
//! through the owned-numeric decoder; pending packets never reach it.
use super::compiler_child::bounded;
use super::{
    compiled, coverage, oracle, owned_numeric_decode,
    owned_numeric_task::{
        self, Advance, History, PENDING, Row, TaskError, TaskShape, TaskWords, TokenKey,
    },
    provenance,
};
use crate::hex::hex;
use crate::{
    eval,
    front::{Span, Type},
};
use invar_foreign::numeric_owned_resume::{
    self as numeric, Driver, MAX_BYTES, OUTPUT_WORDS, Snapshot,
};
#[cfg(test)]
use owned_numeric_task::word;
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};
const PROFILE: &str = "cpu-owned-numeric-continuation-v1";
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
    Decode(owned_numeric_decode::Error),
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
/// The driver field drops (destroying any task, then unloading) before the
/// private directory is removed.
/// ```compile_fail
/// fn send<T: Send>() {} send::<invar::general_net::numeric_owned_resume_host::Plan>();
/// ```
/// ```compile_fail
/// fn sync<T: Sync>() {} sync::<invar::general_net::numeric_owned_resume_host::Plan>();
/// ```
pub struct Plan {
    driver: Driver,
    artifact: provenance::ProvenancedArtifact,
    receipt: Json,
    sites: BTreeMap<TokenKey, (coverage::Rule, Span)>,
    program: crate::mir::Program,
    storage: compiled::OwnedLayout,
    frame_values: usize,
    frame_places: usize,
    state_functions: Vec<usize>,
    cleanup_name_bytes: usize,
    directory: Directory,
    closed: bool,
}
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
        let builder = &mut fs::DirBuilder::new();
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
    // `c++` is `c++.exe` on Windows.
    let exe = format!("{}{}", path.display(), std::env::consts::EXE_SUFFIX);
    std::env::split_paths(&paths)
        .flat_map(|dir| [dir.join(path), dir.join(&exe)])
        .find(|candidate| candidate.is_file())
        .ok_or(Error::Input)
        .and_then(|found| fs::canonicalize(found).map_err(io))
}
fn wrapper(
    artifact: &compiled::Artifact,
    storage: &compiled::OwnedLayout,
    symbol: &str,
    identity: [u8; 32],
    frame_len: usize,
) -> Result<String, Error> {
    // Headers stay outside the namespace. All emitted helpers are hidden so two
    // simultaneously loaded artifacts cannot interpose.
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
    let task_bytes = artifact
        .continuation_layout()
        .ok_or(Error::Limit)?
        .task_bytes();
    let text = format!(
        r#"#include <climits>
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
{headers}
namespace {symbol}_private {{
{body}
static_assert(std::is_trivially_copyable<GnetTask>::value,"copyable task");
static_assert(sizeof(GnetFrame)=={frame_len},"frame layout");
static_assert(offsetof(GnetFrame,pc)=={pc},"frame metadata");
static_assert(offsetof(GnetTask,started)=={frame_bytes},"frame extent");
static_assert(sizeof(GnetTask)=={task_bytes},"task layout");
struct Owner {{
 GnetTask task{{}};
 std::size_t input_n,trace_n;
 gnet_u fuel,transitions,cap,calls=0;
 std::unique_ptr<gnet_w[]> input;
 std::unique_ptr<gnet_u[]> trace;
 std::unique_ptr<gnet_w[]> heap;
 std::unique_ptr<gnet_u[]> cleanup;
 gnet_w output[10];
 Owner(const gnet_w* in,std::size_t n,gnet_u f,gnet_u t,gnet_u c):input_n(n),trace_n((std::size_t)c*7),fuel(f),transitions(t),cap(c),
  input(new gnet_w[n?n:1]()),trace(new gnet_u[trace_n?trace_n:1]()),heap(new gnet_w[{heap}]()),cleanup(new gnet_u[{cleanup_alloc}]()) {{
  for(std::size_t i=0;i<10;++i)output[i]=LLONG_MIN;
  if(n)std::memcpy(input.get(),in,n*8);
 }}
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
 {symbol}_private::gnet_resume(o.input.get(),o.output,o.trace.get(),o.fuel,o.transitions,o.cap,o.heap.get(),o.cleanup.get(),&o.task,quantum);return 0;}}
 catch(const std::bad_alloc&){{return 1;}}catch(...){{return 3;}}
}}
extern "C" INV_EXPORT unsigned int {symbol}_capture(void* handle,long long* input,std::size_t n,long long* output,std::size_t out_n,unsigned long long* trace,std::size_t trace_n,unsigned char* task,std::size_t task_n,unsigned long long* counters,std::size_t counter_n,long long* heap,std::size_t heap_n,unsigned long long* cleanup,std::size_t cleanup_n,unsigned long long* calls) noexcept {{
 if(!handle||!input||!output||!trace||!task||!counters||!heap||!cleanup||!calls)return 2;
 auto& o=*static_cast<{symbol}_private::Owner*>(handle);
 if(n!=o.input_n||out_n!=10||trace_n!=o.trace_n||task_n!=sizeof(o.task)||counter_n!=18||heap_n!={heap}||cleanup_n!={cleanup})return 2;
 if(n)std::memcpy(input,o.input.get(),n*8);std::memcpy(output,o.output,80);
 if(trace_n)std::memcpy(trace,o.trace.get(),trace_n*8);
 std::memcpy(task,&o.task,sizeof(o.task));
 std::memcpy(heap,o.heap.get(),heap_n*8);
 if(cleanup_n)std::memcpy(cleanup,o.cleanup.get(),cleanup_n*8);
 const unsigned long long c[18]={{o.task.started,o.task.done,o.task.depth,o.task.max_depth,o.task.steps,o.task.transitions,o.task.used,o.task.next_frame,o.task.allocated,o.task.live,o.task.cleanup_used,o.task.fuel,o.task.transition_cap,o.task.trace_cap,(unsigned long long)o.task.status,(unsigned long long)o.task.result,(unsigned long long)o.task.start,(unsigned long long)o.task.end}};
 std::memcpy(counters,c,sizeof(c));*calls=o.calls;return 0;
}}
extern "C" INV_EXPORT unsigned int {symbol}_destroy(void* handle) noexcept {{
 if(!handle)return 2;try{{delete static_cast<{symbol}_private::Owner*>(handle);return 0;}}catch(...){{return 3;}}
}}
"#,
        params = artifact.params().len(),
        frame_bytes = artifact.frame_bytes(),
        pc = frame_len - 32,
        heap = storage.heap_words(),
        cleanup = storage.cleanup_words(),
        cleanup_alloc = storage.cleanup_words().max(1),
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
        if limits.compiler.frame_bytes > MAX_FRAMES || source.len() > MAX_SOURCE {
            return Err(Error::Limit);
        }
        let artifact = provenance::compile_resumable_owned_numeric(
            source,
            entry,
            compiled::Target::Cpu,
            limits,
            storage,
        )
        .map_err(Error::Provenance)?;
        let compiled = artifact.artifact();
        if compiled.abi_version() != 11
            || compiled.frame_bytes() > MAX_FRAMES
            || compiled.params().len() > 128
            || compiled.numeric_policy().is_none()
        {
            return Err(Error::Limit);
        }
        let storage = compiled.owned_layout().cloned().ok_or(Error::Input)?;
        let task_bytes = compiled
            .continuation_layout()
            .ok_or(Error::Input)?
            .task_bytes();
        let (net, proof) = artifact.checked_net();
        let program = net
            .compiled_mir(proof)
            .map_err(|e| Error::Provenance(provenance::Error::Net(e)))?;
        // One PC per instruction and terminator in MIR order; frame layout is
        // GnetFrame{v[V],p[P],pc,ret,id,activation} with V/P the existing maxima.
        let mut state_functions = reserve(compiled.states())?;
        let (mut frame_values, mut frame_places) = (1, 1);
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
            frame_values = usize::max(frame_values, values.len());
            frame_places = usize::max(frame_places, function.locals.len());
        }
        let frame_len = (frame_values + frame_places + 4) * 8;
        if state_functions.len() != compiled.states()
            || frame_len * 64 != compiled.frame_bytes()
            || task_bytes != compiled.frame_bytes() + 144
        {
            return Err(Error::Malformed);
        }
        let compiler = compiler_path(compiler)?;
        let compiler_hash = numeric::file_digest(&compiler).map_err(Error::Loader)?;
        let mut directory = Directory {
            path: private_directory("invar-gnet11-host")?,
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
        // The prelink identity excludes the final library hash; the receipt binds it.
        let prelink = json!({"version":1,"profile":PROFILE,"abi":11,"policy":"strict-f32-v1","target":"cpu","artifact":artifact.id(),"source_sha256":hex(&hash(source)),"entry":entry,"params":compiled.params().iter().map(|t|format!("{t:?}")).collect::<Vec<_>>(),"result":format!("{:?}",compiled.result()),"frame_values":frame_values,"frame_places":frame_places,"frame_bytes":compiled.frame_bytes(),"task_bytes":task_bytes,"task_counters":18,"output_words":OUTPUT_WORDS,"heap_words":storage.heap_words(),"cleanup_words":storage.cleanup_words(),"elements":storage.elements(),"allocations":storage.allocations(),"cleanup_events":storage.cleanup_events(),"buffer_bound":MAX_BYTES,"compiler_sha256":hex(&compiler_hash),"compiler_path":compiler,"compiler_version":hex(&hash(&version_out)),"compiler_version_stderr":hex(&hash(&version_err)),"options":OPTIONS,"shared":if cfg!(target_os="macos"){"-dynamiclib"}else{"-shared"},"host_build":env!("INVAR_BUILD_DIGEST"),"frame_bound":MAX_FRAMES,"source_bound":MAX_SOURCE,"library_bound":64*1024*1024,"bounds":{"fuel":1000000,"transitions":1000000,"trace_events":1000000,"quantum":1000000}});
        let identity = hash(prelink.to_string().as_bytes());
        let symbol = format!("invar_{}_owned_resume_v1", hex(&identity));
        let text = wrapper(compiled, &storage, &symbol, identity, frame_len)?;
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
        let receipt = json!({"scope":"source_issued_abi11_loaded_cpu","profile":PROFILE,"artifact":artifact.manifest(),"prelink":prelink,"identity":hex(&identity),"wrapper_sha256":hex(&hash(text.as_bytes())),"wrapper_bytes":text.len(),"library_sha256":hex(&library_hash),"library_bytes":fs::metadata(&library).map_err(io)?.len(),"symbol":symbol});
        fs::write(directory.path.join("receipt.json"), receipt.to_string()).map_err(io)?;
        let mut carriers = reserve(compiled.params().len())?;
        for ty in compiled.params() {
            carriers.push(match ty {
                Type::I32 => numeric::Carrier::I32,
                Type::Bool => numeric::Carrier::Bool,
                Type::U32 | Type::F32 => numeric::Carrier::Word,
                _ => return Err(Error::Input),
            });
        }
        let driver = Driver::load(
            &library,
            library_hash,
            identity,
            &symbol,
            carriers,
            task_bytes,
            [storage.heap_words(), storage.cleanup_words()],
        )
        .map_err(|cause| Error::Load {
            cause,
            evidence: directory.path.clone(),
        })?;
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
            frame_values,
            frame_places,
            state_functions,
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
    /// Conservative aggregate of native storage, two host snapshots, pending
    /// validation scratch and terminal decode scratch.
    fn budget(&self, l: oracle::Limits) -> Option<usize> {
        let compiled = self.artifact.artifact();
        let copy = numeric::copy_bytes(
            compiled.params().len(),
            l.trace_events as u64,
            compiled.continuation_layout()?.task_bytes(),
            [self.storage.heap_words(), self.storage.cleanup_words()],
        )?;
        let history = usize::try_from(l.transitions)
            .ok()?
            .checked_add(1)?
            .checked_mul(std::mem::size_of::<Option<History>>())?;
        let events = l
            .trace_events
            .checked_mul(std::mem::size_of::<Row>() + 56 + std::mem::size_of::<oracle::Event>())?;
        let owners = self
            .storage
            .allocations()
            .checked_mul(std::mem::size_of::<(usize, usize, usize, i64)>() + 64)?;
        let decode = self
            .storage
            .cleanup_events()
            .checked_mul(std::mem::size_of::<eval::CleanupEvent>())?
            .checked_add(self.cleanup_name_bytes)?
            .checked_add(self.storage.elements().checked_mul(4)?)?;
        copy.checked_mul(3)?
            .checked_add(history)?
            .checked_add(events)?
            .checked_add(owners)?
            .checked_add(decode)
    }
    /// Capture inputs and fixed budgets in a fresh native task.
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
            // The artifact fixes native heap capacity; never silently exceed a
            // tighter caller cap.
            || limits.array_elements < self.storage.elements()
            || self.budget(limits).is_none_or(|n| n > MAX_BYTES)
        {
            return Err(Error::Limit);
        }
        let input = encode(self.artifact.artifact().params(), args)?;
        let budgets = [limits.fuel, limits.transitions, limits.trace_events as u64];
        if let Err(cause) = self.driver.start(&input, budgets) {
            if cause == numeric::Error::Allocation {
                return Err(Error::Allocation);
            }
            self.closed = true;
            self.directory.remove = false;
            let _ = fs::write(
                self.directory.path.join("failure-start.json"),
                json!({"input":input,"limits":budgets,"cause":format!("{cause:?}")}).to_string(),
            );
            return Err(Error::Loader(cause));
        }
        let snapshot = match self.driver.snapshot() {
            Ok(s) => s,
            Err(e) => {
                self.closed = true;
                self.directory.remove = false;
                let _ = fs::write(self.directory.path.join("failure-start.json"), json!({"input":input,"limits":budgets,"cause":format!("{e:?}"),"snapshot_observed":false}).to_string());
                let _ = self.driver.finish();
                return Err(Error::Loader(e));
            }
        };
        if snapshot.input != input
            || snapshot.output != PENDING
            || snapshot.counters != [0; 18]
            || snapshot.task.iter().any(|b| *b != 0)
            || snapshot.trace.iter().any(|w| *w != 0)
            || snapshot.heap.iter().any(|w| *w != 0)
            || snapshot.cleanup.iter().any(|w| *w != 0)
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
            terminal: false,
            failed: false,
        })
    }
}
/// Safe counters from a validated yielded continuation; no partial result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pending {
    pub steps: u64,
    pub transitions: u64,
    pub events: usize,
    pub depth: usize,
    pub max_frames: usize,
    pub allocations: usize,
    pub live_elements: usize,
    pub cleanup_events: usize,
}
#[derive(Debug)]
pub enum Progress {
    Pending(Pending),
    /// Freshly decoded from the retained terminal packet on every retrieval,
    /// so separate retrievals never share mutable array owners.
    Complete(oracle::Observation),
}
/// Exclusively borrows a plan. Dropping destroys its native task before the
/// plan can unload its code.
/// ```compile_fail
/// fn send<T: Send>() {} send::<invar::general_net::numeric_owned_resume_host::Run<'static>>();
/// ```
/// ```compile_fail
/// fn sync<T: Sync>() {} sync::<invar::general_net::numeric_owned_resume_host::Run<'static>>();
/// ```
pub struct Run<'a> {
    plan: &'a mut Plan,
    limits: oracle::Limits,
    snapshot: Snapshot,
    terminal: bool,
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
        if self.terminal {
            return self.complete();
        }
        let snapshot = match self.plan.driver.resume(quantum) {
            Ok(s) => s,
            Err(e) => {
                self.failed = true;
                match self.plan.driver.snapshot() {
                    Ok(s) => retain(self.plan, &s, "native resume/capture"),
                    Err(_) => retain(
                        self.plan,
                        &self.snapshot,
                        "native resume/capture (last validated snapshot)",
                    ),
                }
                return Err(Error::Loader(e));
            }
        };
        if let Err(error) = validate(self.plan, self.limits, &self.snapshot, &snapshot, quantum) {
            if !matches!(error, Error::Allocation) {
                retain(self.plan, &snapshot, "task/prefix/storage validation");
            }
            self.failed = true;
            return Err(error);
        }
        self.snapshot = snapshot;
        let c = self.snapshot.counters;
        if c[1] == 0 {
            return Ok(Progress::Pending(Pending {
                steps: c[4],
                transitions: c[5],
                events: c[6] as usize,
                depth: c[2] as usize,
                max_frames: c[3] as usize,
                allocations: c[8] as usize,
                live_elements: c[9] as usize,
                cleanup_events: c[10] as usize,
            }));
        }
        self.terminal = true;
        self.complete()
    }
    /// Decode only an actually completed, validated ABI11 task with the
    /// owned-numeric terminal decoder. No native call is made here.
    fn complete(&mut self) -> Result<Progress, Error> {
        let s = &mut self.snapshot;
        let raw = numeric::Invocation {
            output: s.output,
            trace: std::mem::take(&mut s.trace),
            heap: std::mem::take(&mut s.heap),
            cleanup: std::mem::take(&mut s.cleanup),
        };
        let decoded = owned_numeric_decode::decode(
            self.plan.artifact.artifact().result(),
            self.plan.artifact.source().len(),
            &self.plan.sites,
            &self.plan.program,
            &self.plan.storage,
            self.limits,
            &raw,
        );
        s.trace = raw.trace;
        s.heap = raw.heap;
        s.cleanup = raw.cleanup;
        match decoded {
            Ok(observation) => Ok(Progress::Complete(observation)),
            Err(owned_numeric_decode::Error::Allocation) => Err(Error::Allocation),
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
    let _ = fs::write(path.join("failure.json"), json!({"reason":reason,"counters":snapshot.counters,"output":snapshot.output,"native_calls":snapshot.native_calls}).to_string());
    let _ = fs::write(path.join("failure-task.bin"), &snapshot.task);
    write_words(
        path.join("failure-input.bin"),
        snapshot.input.iter().map(|w| *w as u64),
    );
    write_words(
        path.join("failure-trace.bin"),
        snapshot.trace.iter().copied(),
    );
    write_words(
        path.join("failure-heap.bin"),
        snapshot.heap.iter().map(|w| *w as u64),
    );
    write_words(
        path.join("failure-cleanup.bin"),
        snapshot.cleanup.iter().copied(),
    );
}
/// Stream fixed-width words; never allocate a second representation on failure.
fn write_words(path: PathBuf, words: impl Iterator<Item = u64>) {
    if let Ok(file) = fs::File::create(path) {
        let mut out = std::io::BufWriter::new(file);
        for word in words {
            if out.write_all(&word.to_le_bytes()).is_err() {
                break;
            }
        }
    }
}
/// Borrowed checker view of this plan; the checker lives in
/// `owned_numeric_task`, shared with the ABI12 device host design.
fn shape(plan: &Plan) -> Result<TaskShape<'_>, Error> {
    let artifact = plan.artifact.artifact();
    Ok(TaskShape {
        frame_bytes: artifact.frame_bytes(),
        task_bytes: artifact
            .continuation_layout()
            .ok_or(Error::Malformed)?
            .task_bytes(),
        states: artifact.states(),
        frame_values: plan.frame_values,
        frame_places: plan.frame_places,
        state_functions: &plan.state_functions,
        sites: &plan.sites,
        program: &plan.program,
        storage: &plan.storage,
    })
}
fn words(s: &Snapshot) -> TaskWords<'_> {
    TaskWords {
        input: &s.input,
        output: s.output,
        trace: &s.trace,
        task: &s.task,
        counters: s.counters,
        heap: &s.heap,
        cleanup: &s.cleanup,
    }
}
/// Validate a newly captured task against the previous validated snapshot
/// (one native resume call). Behavior is unchanged by the extraction.
fn validate(
    plan: &Plan,
    l: oracle::Limits,
    old: &Snapshot,
    new: &Snapshot,
    quantum: u64,
) -> Result<(), Error> {
    owned_numeric_task::validate(
        &shape(plan)?,
        l,
        &words(old),
        &words(new),
        quantum,
        Advance::NativeCall {
            previous: old.native_calls,
            current: new.native_calls,
        },
    )
    .map_err(|e| match e {
        TaskError::Malformed => Error::Malformed,
        TaskError::Allocation => Error::Allocation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    const SOURCE: &[u8] = b"fn grow(xs:mut Array<f32>)->Unit{let t:Array<f32> = [2.0];xs[0u32]=xs[0u32]*t[0u32];} fn run(n:u32)->Array<f32>{var xs:Array<f32> = [1.0,-0.0];for i in 0u32..n{grow(xs);let gone:Array<i32> = [1];}let ys=move xs;return ys;}";
    fn plan() -> Plan {
        Plan::from_source(
            SOURCE,
            "run",
            Default::default(),
            compiled::OwnedLimits {
                elements: 32,
                allocations: 32,
                cleanup_events: 32,
            },
            Path::new("c++"),
        )
        .unwrap()
    }
    fn limits() -> oracle::Limits {
        oracle::Limits {
            fuel: 1000,
            transitions: 10_000,
            trace_events: 1000,
            array_elements: 32,
        }
    }
    fn args() -> [eval::Value; 1] {
        [eval::Value::U32(3)]
    }
    fn budgets(l: oracle::Limits) -> [u64; 3] {
        [l.fuel, l.transitions, l.trace_events as u64]
    }
    fn copy(s: &Snapshot) -> Snapshot {
        Snapshot {
            input: s.input.clone(),
            output: s.output,
            trace: s.trace.clone(),
            task: s.task.clone(),
            counters: s.counters,
            heap: s.heap.clone(),
            cleanup: s.cleanup.clone(),
            native_calls: s.native_calls,
        }
    }
    fn sync_task(s: &mut Snapshot, frame_bytes: usize) {
        for (bytes, counter) in s.task[frame_bytes..].chunks_exact_mut(8).zip(s.counters) {
            bytes.copy_from_slice(&counter.to_ne_bytes());
        }
    }
    #[test]
    fn issue_gnet_11_loaded_exact_task_storage_at_each_quantum() {
        let mut p = plan();
        let (l, a) = (limits(), args());
        let mut states = BTreeMap::new();
        let (mut yields, mut matches, mut nested, mut owned) = (0, 0, 0, 0);
        for quantum in [1, 2, 3, 17] {
            let mut r = p.start(&a, l).unwrap();
            loop {
                let progress = r.resume(quantum).unwrap();
                let s = &r.snapshot;
                let key = (s.counters[5], s.counters[1]);
                let state = (
                    s.task.clone(),
                    s.trace.clone(),
                    s.heap.clone(),
                    s.cleanup.clone(),
                    s.output,
                );
                if quantum == 1 {
                    states.insert(key, state);
                } else {
                    assert_eq!(states.get(&key), Some(&state), "quantum {quantum} {key:?}");
                    matches += 1;
                }
                match progress {
                    Progress::Pending(pending) => {
                        yields += 1;
                        nested += usize::from(pending.depth > 1);
                        owned += usize::from(pending.allocations > 0 && pending.cleanup_events > 0);
                    }
                    Progress::Complete(o) => {
                        // Replay decodes the retained packet without a native call.
                        let calls = r.plan.driver.snapshot().unwrap().native_calls;
                        let Progress::Complete(again) = r.resume(1).unwrap() else {
                            panic!("terminal replay yielded")
                        };
                        assert_eq!(r.plan.driver.snapshot().unwrap().native_calls, calls);
                        assert_eq!(again.events, o.events);
                        let (net, proof) = r.plan.artifact.checked_net();
                        let oracle = oracle::observe_call(net, proof, "run", &a, l).unwrap();
                        assert_eq!(o.events, oracle.events);
                        assert_eq!(o.semantic.cleanup_trace, oracle.semantic.cleanup_trace);
                        let Ok(eval::Value::ArrayF32(xs)) = &o.semantic.result else {
                            panic!("{:?}", o.semantic.result)
                        };
                        let bits: Vec<_> = xs.borrow().iter().map(|x| x.to_bits()).collect();
                        assert_eq!(bits, [0x41000000, 0x80000000]);
                        break;
                    }
                }
            }
        }
        // Measured on the qualified artifact: 300 yields, 144 cross-quantum
        // matches, 145 nested-frame and 209 owner-plus-cleanup pending states.
        assert!(yields >= 300 && matches >= 144 && nested >= 145 && owned >= 209);
        eprintln!("ABI11 yields={yields} matches={matches} nested={nested} owned={owned}");
    }
    /// Advance by single transitions until a pending task has nested frames,
    /// several issued owners and at least one cleanup row.
    fn rich_pending(p: &mut Plan) -> (Snapshot, Snapshot) {
        let l = limits();
        p.driver
            .start(
                &encode(p.artifact.artifact().params(), &args()).unwrap(),
                budgets(l),
            )
            .unwrap();
        let mut old = p.driver.snapshot().unwrap();
        loop {
            let new = p.driver.resume(1).unwrap();
            validate(p, l, &old, &new, 1).unwrap();
            let c = new.counters;
            assert_eq!(c[1], 0, "no rich pending state before completion");
            if c[2] > 1 && c[8] >= 3 && c[10] >= 1 && c[8] > old.counters[8] {
                return (old, new);
            }
            old = new;
        }
    }
    #[test]
    fn issue_gnet_11_loaded_malformed_pending_state_refused() {
        let mut p = plan();
        let l = limits();
        let (old, good) = rich_pending(&mut p);
        let frame_bytes = p.artifact.artifact().frame_bytes();
        let frame_len = frame_bytes / 64;
        let slots = p.storage.allocations();
        let allocated = good.counters[8] as usize;
        let count = good.counters[10] as usize;
        let dead = (0..allocated)
            .find(|i| good.heap[i * 4 + 3] == 0)
            .expect("dead owner");
        let live: Vec<_> = (0..allocated)
            .filter(|i| good.heap[i * 4 + 3] == 1)
            .collect();
        assert!(live.len() >= 2);
        let run = p
            .program
            .functions
            .iter()
            .position(|f| f.name == "run")
            .unwrap();
        let xs = p.program.functions[run]
            .declaration_order
            .iter()
            .position(|n| n == "xs")
            .unwrap();
        let place = (p.frame_values + xs) * 8;
        assert_ne!(word(&good.task[place..place + 8]).unwrap(), 0);
        for mutation in 0..44 {
            let mut s = copy(&good);
            match mutation {
                0 => s.counters[0] = 0,
                1 => s.counters[1] = 2,
                2 => s.counters[2] = 65,
                3 => s.counters[3] = 0,
                4 => s.counters[4] = l.fuel + 1,
                5 => s.counters[5] = l.transitions + 1,
                6 => s.counters[6] = l.trace_events as u64 + 1,
                7 => s.counters[7] = s.counters[5] + 2,
                8 => s.counters[8] += 1,
                9 => s.counters[9] += 1,
                10 => s.counters[10] += 1,
                11 => s.counters[11] += 1,
                12 => s.counters[12] += 1,
                13 => s.counters[13] += 1,
                14 => s.counters[14] = 8,
                15 => s.counters[15] = 1,
                16 => s.counters[16] = 1,
                17 => s.counters[17] = 1,
                18 => s.output[9] = 0,
                19 => s.trace[0] = 2,
                20 => s.trace[3] = u64::MAX,
                21 => s.trace[6] = u64::MAX,
                22 => *s.trace.last_mut().unwrap() = 1,
                23 => s.input[0] ^= 1,
                24 => s.native_calls += 1,
                // Issued owner length/kind are immutable; dead never resurrects.
                25 => s.heap[1] += 1,
                26 => s.heap[2] = 2,
                27 => s.heap[3] = 2,
                28 => s.heap[dead * 4 + 3] = 1,
                29 => s.heap[allocated * 4] = 1,
                30 => s.heap[slots * 4] = i64::MAX,
                31 => *s.heap.last_mut().unwrap() = 1,
                32 => s.heap[live[1] * 4] = s.heap[live[0] * 4],
                33 => s.cleanup[0] = 99,
                34 => s.cleanup[1] = 99,
                35 => s.cleanup[2] = 99,
                36 => s.cleanup[3] = u64::MAX,
                37 => s.cleanup[count * 4] = 1,
                // Active array places hold zero or a live owner of their kind.
                38 => s.task[place..place + 8].copy_from_slice(&99u64.to_ne_bytes()),
                39 => s.task[place..place + 8].copy_from_slice(&(dead as u64 + 1).to_ne_bytes()),
                40 => s.counters[5] -= 1,
                41 => s.counters[3] = 0,
                42 => s.heap[slots * 4 + live[0]] = i64::from(u32::MAX) + 1,
                43 => s.trace[1] = 1,
                _ => unreachable!(),
            }
            sync_task(&mut s, frame_bytes);
            assert!(
                matches!(validate(&p, l, &old, &s, 1), Err(Error::Malformed)),
                "mutation {mutation}"
            );
        }
        // Counter bytes inside the task must equal reported counters.
        let mut s = copy(&good);
        s.task[frame_bytes] ^= 1;
        assert!(matches!(
            validate(&p, l, &old, &s, 1),
            Err(Error::Malformed)
        ));
        for part in 0..4 {
            let mut s = copy(&good);
            let begin = frame_len - 32 + part * 8;
            s.task[begin..begin + 8].copy_from_slice(&u64::MAX.to_ne_bytes());
            assert!(matches!(
                validate(&p, l, &old, &s, 1),
                Err(Error::Malformed)
            ));
        }
        // A pending quantum must consume exactly its transitions.
        assert!(matches!(
            validate(&p, l, &old, &good, 2),
            Err(Error::Malformed)
        ));
        for name in ["trace", "heap", "cleanup", "task"] {
            let mut s = copy(&good);
            let shortened = match name {
                "trace" => s.trace.pop().is_some(),
                "heap" => s.heap.pop().is_some(),
                "cleanup" => s.cleanup.pop().is_some(),
                _ => s.task.pop().is_some(),
            };
            assert!(shortened);
            assert!(
                matches!(validate(&p, l, &old, &s, 1), Err(Error::Malformed)),
                "short {name}"
            );
        }
        retain(&mut p, &good, "deliberate malformed pending qualification");
        p.driver.finish().unwrap();
        assert!(matches!(p.start(&args(), l), Err(Error::Closed)));
        assert!(p.evidence_directory().join("failure-heap.bin").is_file());
    }
    #[test]
    fn issue_gnet_11_loaded_malformed_terminal_state_refused() {
        let mut p = plan();
        let l = limits();
        p.driver
            .start(
                &encode(p.artifact.artifact().params(), &args()).unwrap(),
                budgets(l),
            )
            .unwrap();
        let mut old = p.driver.snapshot().unwrap();
        let done = loop {
            let new = p.driver.resume(5).unwrap();
            validate(&p, l, &old, &new, 5).unwrap();
            if new.counters[1] == 1 {
                break new;
            }
            old = new;
        };
        assert_eq!(done.counters[2], 0);
        // Frozen R1: resuming an already-done native handle returns success
        // without a native call or any storage mutation.
        for quantum in [1, 5, 1_000_000] {
            let again = p.driver.resume(quantum).unwrap();
            assert_eq!(again.native_calls, done.native_calls);
            assert_eq!(again.counters, done.counters);
            assert_eq!(again.output, done.output);
            assert_eq!(again.task, done.task);
            assert_eq!(again.trace, done.trace);
            assert_eq!(again.heap, done.heap);
            assert_eq!(again.cleanup, done.cleanup);
            assert_eq!(again.input, done.input);
        }
        let frame_bytes = p.artifact.artifact().frame_bytes();
        let root = usize::try_from(done.output[1] - 1).unwrap();
        for mutation in 0..8 {
            let mut s = copy(&done);
            match mutation {
                0 => s.output[8] += 1,
                1 => s.output[9] += 1,
                2 => s.output[0] = 1,
                3 => s.counters[2] = 1,
                4 => s.heap[root * 4 + 3] = 0,
                5 => s.counters[14] = 14,
                6 => s.output = PENDING,
                7 => s.native_calls += 1,
                _ => unreachable!(),
            }
            sync_task(&mut s, frame_bytes);
            let accepted = validate(&p, l, &old, &s, 5).and_then(|_| {
                let raw = numeric::Invocation {
                    output: s.output,
                    trace: s.trace,
                    heap: s.heap,
                    cleanup: s.cleanup,
                };
                owned_numeric_decode::decode(
                    p.artifact.artifact().result(),
                    p.artifact.source().len(),
                    &p.sites,
                    &p.program,
                    &p.storage,
                    l,
                    &raw,
                )
                .map_err(Error::Decode)
            });
            assert!(accepted.is_err(), "terminal mutation {mutation}");
        }
        p.driver.finish().unwrap();
    }
    fn fault_library(resume: u32, destroy: u32) -> (PathBuf, PathBuf) {
        let dir = private_directory("gnet11-fault-library").unwrap();
        let cpp = dir.join("fault.cpp");
        let lib = dir.join(format!("fault.{}", std::env::consts::DLL_EXTENSION));
        fs::write(&cpp, format!(r#"#include <climits>
#include <cstddef>
static int owner;
extern "C" unsigned int invar_fault_identity(unsigned char* p,std::size_t n){{if(n!=32)return 2;for(int i=0;i<32;i++)p[i]=0;return 0;}}
extern "C" unsigned int invar_fault_create(const long long*,std::size_t,unsigned long long,unsigned long long,unsigned long long,void** h){{*h=&owner;return 0;}}
extern "C" unsigned int invar_fault_resume(void*,unsigned long long){{return {resume};}}
extern "C" unsigned int invar_fault_capture(void*,long long* in,std::size_t n,long long* out,std::size_t,unsigned long long*,std::size_t,unsigned char*,std::size_t,unsigned long long*,std::size_t,long long*,std::size_t,unsigned long long*,std::size_t,unsigned long long* calls){{for(std::size_t i=0;i<n;i++)in[i]=3;for(int i=0;i<10;i++)out[i]=LLONG_MIN;*calls=0;return 0;}}
extern "C" unsigned int invar_fault_destroy(void*){{return {destroy};}}
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
        (dir, lib)
    }
    fn inject(p: &mut Plan, lib: &Path) {
        p.driver = Driver::load(
            lib,
            numeric::file_digest(lib).unwrap(),
            [0; 32],
            "invar_fault",
            vec![numeric::Carrier::Word],
            p.artifact
                .artifact()
                .continuation_layout()
                .unwrap()
                .task_bytes(),
            [p.storage.heap_words(), p.storage.cleanup_words()],
        )
        .unwrap();
    }
    #[test]
    fn issue_gnet_11_loaded_returning_faults_poison_and_retain_evidence() {
        for protocol in [0, 1, 2, 3] {
            let mut p = plan();
            let (_fixture, lib) = fault_library(protocol, 0);
            inject(&mut p, &lib);
            let l = limits();
            let mut run = p.start(&args(), l).unwrap();
            let failure = run.resume(1);
            match protocol {
                0 => assert!(matches!(failure, Err(Error::Malformed))),
                1 => assert!(matches!(
                    failure,
                    Err(Error::Loader(numeric::Error::Allocation))
                )),
                code => assert!(matches!(
                    failure,
                    Err(Error::Loader(numeric::Error::Protocol(c))) if c == code
                )),
            }
            assert!(matches!(run.resume(1), Err(Error::Closed)));
            drop(run);
            assert!(matches!(p.start(&args(), l), Err(Error::Closed)));
            let evidence = p.evidence_directory().to_owned();
            drop(p);
            assert!(evidence.join("failure.json").is_file());
            assert!(evidence.join("failure-heap.bin").is_file());
            assert!(evidence.join("receipt.json").is_file());
        }
    }
    #[test]
    fn issue_gnet_11_loaded_destruction_failure_keeps_mapping_and_evidence() {
        let mut p = plan();
        let (_fixture, lib) = fault_library(0, 3);
        inject(&mut p, &lib);
        let run = p.start(&args(), limits()).unwrap();
        drop(run);
        assert!(matches!(p.start(&args(), limits()), Err(Error::Closed)));
        let evidence = p.evidence_directory().to_owned();
        // Dropping the plan keeps the failed native mapping loaded forever.
        drop(p);
        let reason = fs::read_to_string(evidence.join("failure.json")).unwrap();
        assert!(reason.contains("native destruction"));
    }
    #[cfg(unix)]
    #[test]
    fn issue_gnet_11_loaded_compiler_deadline_and_log_caps() {
        let path = private_directory("gnet11-capture").unwrap();
        let r = bounded(
            Command::new("sh").args(["-c", "printf 'partial ABI11 log' >&2;exec sleep 10"]),
            &path.join("deadline"),
            Duration::from_millis(100),
        );
        assert!(r.unwrap_err().contains("deadline"));
        assert_eq!(
            fs::read(path.join("deadline.stderr")).unwrap(),
            b"partial ABI11 log"
        );
        let r = bounded(
            Command::new("head").args(["-c", "65537", "/dev/zero"]),
            &path.join("cap"),
            Duration::from_secs(1),
        );
        assert!(r.unwrap_err().contains("output limit"));
        assert_eq!(fs::metadata(path.join("cap.stdout")).unwrap().len(), 65536);
    }
}
