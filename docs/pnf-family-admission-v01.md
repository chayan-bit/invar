# PNF family admission (P3)

Static family admission is implemented in `general_net::regions::family`, beside
P1's region partition. It does not register an A1 adapter, call a recognizer,
execute a region, choose placement or grant publication authority.

`family::analyze(net, proof, form) -> Result<Report, Error>` consumes a verified
net and its PNF. It first checks every law certificate with P1's independent
checker. It then re-derives the node classification and loop inventory; checked
lowering annotations may be added, but deleting a loop or changing its node is
rejected. A dense frontier may match a serial round wrapper only when its frontier and
copy children have checked PNF laws; rounds remain ordered. The predicates inspect the net-derived statement tree, PNF
representations and MIR type/ownership metadata. No source text, program id,
filename, source hash or benchmark label participates in admission.

`Report { fingerprint, admissions, refusals }` exposes static matches as
`Admission { region: RegionId, span: Span, family: Family }`. The fingerprint is
the compiled-MIR contract digest and `span` is the original MIR loop-head span,
matching A0's reached-loop selector. More than one static family may match.
The dispatcher retains A0 precedence: core, grid, tensor, segmented, indexed,
summary; a live refusal never retries another recognizer.

`Refusal { candidate: Admission, reason: MissingFiniteStateLaw }` identifies a
bounded table summary shape that stays native because P1 has no L8 certificate.
It grants no admission or execution authority. This is derived from the checked
node and body, independent of the source recognizers and corpus case names.

| Predicates | Required shape and body |
| --- | --- |
| Map, FusedMap, Dag | independent u32 stores, one wrapping add/multiply over indexed inputs and/or constants; fusion traverses checked L19 edges, with 2 stages for a pair and 3 to 16 for a DAG |
| Scan, Reduce | one wrapping-u32 state; explicit affine multiply/add with optional inclusive/exclusive state publication, or a sealed sum/product reduction |
| Transpose, Stencil | fresh u32/f32 row-major transpose; strict-f32 radius-one five-point stencil with exact north/south/west/east branch and sum order, zero or clamp edges |
| Tensor | fresh strict-f32 rank-2 output, counted i/j/k loops, scalar accumulator or direct output-cell accumulation; the inner f32 order is preserved |
| SegmentedMap, SegmentedReduce, Compaction | nested offset intervals with wrapping maps or sealed reductions; matching comparison/count/copy passes with a certified counter |
| Gather, Scatter, AffineScatter, Histogram, Frontier | indexed gather, guarded injective scatter, bounded literal affine scatter, wrapping weighted histogram, dense-mask ascending frontier form |
| Summary | width 1 to 4 wrapping-u32 matrix summaries, final state or inclusive prefixes; bounded finite-state tables produce an explicit native refusal because L8 is absent |

Every family still needs its existing live guard and source-bound contract.
In particular, offsets must be ordered/covered, scatter indices must be in range
and injective where required, numeric policies must match, and pending device
work must be quiescent before fallback. None of those runtime facts is inferred
from a static match. The publication bridge remains `CompletedRegion`;
P3 constructs no completion, MIR override or device receipt.

## Verification and current gaps

`tests/normal_form/pnf_family_admission.rs` compares admissions by function and exact source
span with the existing recognizers. It reads the 72 GNET-05 programs and the 48
GNET-05 robotics programs in place under `bench/general-net-v01/gnet05`, plus the
140 M5-01 fixtures; no source generator is rerun. Focused
family programs and rename/near-miss variants supplement the corpus.

The differential check requires every recognizer match to have either an
admission or an explicit production-derived native refusal. It rejects invented
refusals and any admission/refusal overlap. Set `INVAR_P3_REPORT` to export every
unresolved difference and every refusal with its complete program. Raw admission
equality is not claimed for refused tables. There is no case-name or hash allowlist.
CPU net-oracle comparisons exercise supported u32 and f32 bodies,
short arrays, wrapping extremes, signed zero, subnormals and NaNs. Mutations
exercise stale fingerprints, missing loops/certificates, invented representations
and the wrong first trap, plus removed frontier-child laws and runtime guards. These are CPU analysis/reference checks, not GPU kernel
qualification.

The normal-form base does not export `segmented_device` or `indexed_device`.
The device branches also contain u32-stencil and explicit Tensor Core source
contracts absent from this base. Their adapter/recognizer files are not imported
or modified by P3. Finite-state tables have no PNF law representation (L8 is
outside P1's v0.1 law set), so they remain refused. Moved-owner fusion consumes
P1's independently checked immutable-length proof; mixed checked/wrapping state
updates remain serial, with checkable forms. Unavailable comparison interfaces
remain explicit. P3 produces no runnable kernels and makes no D5 performance claim.
