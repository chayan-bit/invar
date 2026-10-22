# GNET P5b: emitted reachability and Metal backend archives

Status: D10 implemented on `feat/v01-gnet-p5b`, stacked on
`feat/v01-net-normal-form`. Per-program measurements are complete for every
admitted corpus program; explicit refusals remain in the denominator. See the
evidence checkpoint for current hardware checks and interface gaps.

## Emission

`general_net::compiled::reachable::blocks` derives a conservative CFG and
transitive call closure from the selected entry of `Net::compiled_mir(proof)`.
It retains both branch arms, loops, recursion and every statically called
function. No input, filename, family, benchmark identity or timing enters the
analysis. Whole-candidate type/rule/span admission still runs over every
function, including ones whose code is omitted. The existing frame/value/place
bounds remain conservative; this change does not widen those budgets. Foreign
stubs have no CFG and remain subject to the original typed admission checks.

The emitter omits unreachable continuation switch arms and speculative projections.
It reserves the full original logical cursor table: `Artifact::states()` and
every PC retain their existing ABI meaning. `Artifact::emitted_states()` reports
only emitted arms. This matters to the owned-numeric host, which reconstructs
logical cursors independently. The original state-count admission limit is
unchanged.
Each return state's continuation switch contains only callers of that function.
The exact numeric template retains the transitive helper-call closure of the
emitted program. Scalar-only integer programs omit the unused heap helper
bodies while retaining the same ten-word result and heap/cleanup ABI.

These are code-emission omissions, not changes to the net, scheduling, algebra
or floating-point expressions. No algebraic law is applied and no law
certificate is fabricated. The existing checked net proof is still mandatory.
P1's certificate interfaces and the sibling PNF/lowering implementations are
unchanged. D5 execution geometry and tier selection are unchanged. The frozen follow-up
measurement misses all four flat resident comparisons; binary archives do not
close that runtime gap. See `evidence/gnet-p5b/followup/D5-RESULTS.md`.

Two CUDA compiler-sharing diagnostics are retained as negative results in the
follow-up ledger. Sharing exact arithmetic helper bodies, and then the larger
continuation engine, did not admit the selected robotics preparations within
the existing 300-second compiler deadline. Those annotations are absent from
production. Source admission and physical runtime qualification stay separate.

## Cache interface and identity

The existing `MetalDevice::compile` and bounded pipeline LRU remain the shared
boundary for all Metal routes. `metal_aot::Image::install` supplies the validated
container header through `MetalDevice::load_library_with_identity`. Raw
`load_library` callers without compiler identity keep working and do not use
the persistent cache. Developer runtime-source libraries identify the system
compiler by its OS build; production AOT keys include the recorded offline
compiler, linker, SDK and options, plus the exact library bytes.

A persistent key is SHA-256 over length-framed fields:

1. exact generated MSL SHA-256;
2. validated container/program/compiler/options identity and metallib SHA-256;
3. function name;
4. Metal device registry identity and device name;
5. OS version/build;
6. pinned runtime source and default pipeline option identity.

Thus original source/net/rule/layout/numeric changes recorded in the AOT header
or emitted code, compiler changes, options, GPU changes and OS updates cannot
reuse an old identity. The cache directory is
`$HOME/.cache/invar/metal-archives`, overridable with
`INVAR_METAL_ARCHIVE_DIR`. It must be a private directory, without a final
symlink. Each entry is bounded to 64 MiB. New identities stop at 128 files;
concurrent arrivals can overshoot by one bounded file per writer. There is no
eviction or shared-service maintenance.

## Admission, rebuild and publication

`INVGMBA1` frames the expected identity, payload digest and Metal backend
archive. AOT and archives share the same `cache_file` implementation:
`O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC`, regular-file/permission/size checks and
read through one descriptor. Cache bytes are then copied into a newly created
0700 transaction directory; Metal receives that private snapshot URL, never
the original checked cache path.

A hit requires `MTLPipelineOption::FailOnBinaryArchiveMiss` to succeed, following
[Apple's archive API](https://developer.apple.com/documentation/metal/creating-binary-archives-from-device-built-pipeline-state-objects). Loading
an archive without that option is not counted as a hit. Header, identity,
digest, permission, size and Metal refusals are typed. A refused entry rebuilds
from the already admitted library. Production never falls back to runtime MSL
source compilation. Cache write failure is reported while preserving a
successfully created pipeline.

Publication uses a process/thread-unique 0600 sibling, sync, atomic rename and
parent sync. Concurrent writers can duplicate compilation, but readers observe
one complete entry. A failed exclusive creation never removes another writer's
file. Cache reports separate memory hits, archive hits, misses, rebuilds, writes
and write refusals. Memory hits restore the reused entry's identity and size
without resetting cumulative counters. Successful AOT registration invalidates
only that source's prior runtime-source pipelines, so a memory hit cannot hide
a changed compiler identity. `gnet-device` includes a `metal_archive` receipt; the library
API exposes `CacheReport::archive` to other callers.

`MetalDevice::measure_pipeline` compiles without dispatching, returning source
library time, the isolated pipeline-creation API time, pipeline/cache time,
pipeline limits and serialized archive bytes.
Archive bytes are an on-disk pipeline-size proxy, not resident GPU code size.
Cold processes do not imply an empty OS shader cache; evidence must say so.

## Qualification and interface gaps

See `evidence/gnet-p5b/README.md` and `sources.json` for exact revisions,
per-program bytes, explicit refusals and completed/unavailable checks.

The follow-up adds owned i32/u32/f32 root results to the existing ABI13
completed-heap boundary. Result validation checks the issued allocation count,
kind, live flag, length, range and canonical words before any host publication.
The caller receives a fresh array; source cleanup and traps remain unchanged.
Compiled zero-child profiles omit unusable speculative projections while
preserving every logical cursor. Default limits remain unchanged. Explicit
source budgets now reach a shared bounded4MiB GNET ceiling across emission,
Metal admission and direct/child CUDA compilation; ordinary device-source
limits are unchanged. The robotics corpus profile uses4,096 values,
512 places,16,384 states,2MiB frames and4MiB generated source, with zero
child slots. Admission and physical execution evidence remain separate.

ABI13 production AOT now includes its exact offline D1 preflight library.
`metal_aot::gnet_kernels` is the shared build/load inventory;
`Header::verify_gnet` checks it before installing any library. Existing format-v1
multi-library framing already supports this, so no container version change or
runtime source fallback is needed. ABI5 continues to require one program
library. Missing, stale and wrong-entry probe inventories have regression tests.

The historical source/compile measurements above describe the earlier
qualified snapshot. Current follow-up evidence is kept separately under
`evidence/gnet-p5b/followup`; pending hardware jobs and failed preparations are
not promoted to acceptance. Robotics still depends on explicit compiler,
heap, trace and fuel budgets. Admission does not establish that every scored
robotics size fits or executes. The frozen D5 diagnostic measures the actual
emitted kernel against unchanged handwritten baselines; its retained-buffer
wall scope does not replace production host-to-host acceptance.

Changes to #129 code remain limited to ABI13 integration, the archive hook and
identity, shared file validation/publication, metrics and reporting. No unrelated
compiler-option or AOT container-format change is intended.
