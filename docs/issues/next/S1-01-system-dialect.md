# S1-01 - System dialect: components, ports, connections, frames and products

Problem/result: let one Invar source describe a robot system (components with typed ports, their connections, frames and the product they ship in), evaluated at compile time into one checked, versioned IR `invar-system-v1` that every integration target consumes.

Dependencies: L1-01, R1-01, AI-01, S1-11.

Release: v0.1 (milestone S1).

Design: `docs/system-integration-v1.md` sections 1, 2, 5 and "Numbers before M4-01".

## Scope

Grammar (added to `spec/grammar.ebnf` and the front end) for these declarations, all ordinary items inside L1-01 modules:

```text
frame world;
frame base : world;

component Planner {
  port in  goal:  Stamped<Pose<f64, base, tool>>  @ 10 Hz;
  port out traj:  JointTrajectory<f64, 7>         @ 10 Hz latency 20 ms;
  param max_velocity: Quantity<dec, rad/s> = 1.5 rad/s;
  resource cpu_cores = 1, memory = 512 MiB, gpu_memory = 2 GiB;
  license "Apache-2.0";
  isolation separate-process;
  wcet 15 ms; deadline 100 ms;
  impl extern curobo_planner;          // S1-02 manifest, or `impl invar path::to::fn`
}

extern component curobo_planner { ... }  // clauses listed below; S1-02 adds generation and contract tests

connect planner.traj -> controller.traj policy queue(4);
flow reach: camera.image -> planner.goal -> controller.cmd within 120 ms;

product SimCell {
  distribution proprietary-binary; commercial true;
  robot menagerie::franka_emika_panda;  // S1-05 robot model reference
  capacity cpu_cores = 8, memory = 16 GiB, gpu_memory = 24 GiB, bus.ethercat = 100 Mbit/s;
}

system cell for SimCell { planner: Planner; controller: Controller; ... }
```

- `component` clauses: `port in|out name: Type @ rate [latency budget]`, `param name: Type = constant`, `resource key = quantity`, `license "<SPDX expression>"`, `isolation in-process | separate-process | separate-partition`, `wcet`, `deadline`, `impl extern <manifest> | impl invar <function path>`.
- `extern component` clauses (parsed, checked for well-formedness and recorded in the IR here, so S1-03 can check them before S1-02): `abi c "<prefix>"`, `lifecycle`, `port ... layout c "<name>"`, `resource`, `license "<SPDX>" link static | dynamic | process`, `isolation`, `golden`, `property`, `shim`; `impl invar` components are recorded as `link static` Invar code.
- `connect a.out -> b.in policy sample-and-hold | queue(n) | decimate(n)`; `flow name: p1 -> p2 -> ... within budget`.
- `frame`, `product`, `system` as above; the `system` names exactly one product and instantiates components by name.
- `dec` is the exact decimal scalar for compile-time constants; `f32` and `f64` are admitted as wire element types of ports with IEEE layouts but no arithmetic before M4-01.
- Evaluation: the reference evaluator evaluates the `system` at compile time (bounded by the existing fuel and size limits) into `invar-system-v1`, serialized as canonical JSON (sorted keys, no insignificant whitespace, UTF-8, decimals as shortest exact strings, stable ids from declaration paths).
- The JSON Schema `schema/invar-system-v1.schema.json` is committed.
- CLI: `invar system check PATH --format json` emits `invar-diagnostics-v1`; `invar system ir PATH` writes the canonical IR to stdout.
- Checker for well-formedness and agreement: every `connect` joins an `out` to an `in` of the identical type (unit and frame included), each `in` port has at most one producer, names resolve, every component instantiated by the `system` is defined, the product and robot reference resolve, parameter defaults check against their types.

## Semantics

System declarations have no runtime semantics; they denote the IR.
Two sources that differ only in formatting, comments or declaration order within a module produce byte-identical IR.
Rates, budgets and resources are exact rationals with units; S1-03 consumes them.
An unconnected `in` port is an error unless declared `optional`; an unconnected `out` port is a warning.
Diagnostics use codes `S1000` to `S1099` registered in the AI-01 registry.

## Acceptance tests

Add the integration test binary `issue_s1_01_system_dialect`.
Run `cargo test --locked issue_s1_01_system_dialect` in Debug and Release and require a nonzero test count.
Positive cases:
- fixtures under `tests/fixtures/s1_01/` covering every clause produce IR that validates against the committed schema and matches golden IR files;
- permuting declarations and reformatting yields byte-identical IR (double-run hash and permutation test);
- a round trip IR -> Rust model -> canonical JSON is the identity for every fixture;
- ports typed with S1-11 records and R1-01 quantities connect across components with no conversion.

## Negative cases

Each is rejected with a registered `S10xx` code and a primary span: type, unit or frame mismatch across a `connect`; `out -> out` or `in -> in`; two producers for one `in`; an undefined component, port, product, frame or manifest; a `system` naming no product or two; a parameter default of the wrong unit; an unparsable SPDX string (syntax only; license rules are S1-03); arithmetic on `f64` before M4-01; an evaluation that exceeds fuel or size limits (a limit diagnostic, never a truncated IR).

## Done criteria

Grammar, front end, checker, evaluator lowering, schema, CLI commands and golden fixtures are committed; the test binary passes in Debug and Release with a nonzero count; every code is registered and has a fixture.

## Out of scope

Graph-wide checks (S1-03), manifest contents and contract tests (S1-02), generators (S1-04, S1-05), product variants (R2-04), runtime execution of any kind.
