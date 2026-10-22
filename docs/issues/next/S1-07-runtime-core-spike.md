# S1-07 - Runtime-core spike: ros2_control versus Copper-rs as generation target

Release: v0.2 (milestone S1); planning contract, to be expanded into a full contract before implementation.

Dependencies: S1-06.

## Goal

Decide by measurement whether the real-time core is generated as ros2_control or as Copper-rs, and record the outcome as a design decision.

## Motivation

v0.1 generates ros2_control as the interim target (design note section 10); Copper-rs (Apache-2.0, static compile-time task graph, deterministic replay) is the closest design reference, and the choice fixes what S1 generators emit for hard real-time islands.

## Scope

- Implement the same 1 kHz joint impedance controller on a simulated arm (the S1-06 Panda cell) twice: as generated ros2_control configuration plus controller, and as a generated Copper-rs task graph.
- Measure on a PREEMPT_RT kernel: control-loop jitter (p50, p99, p99.9, max over at least 10 minutes), bit-exact replay of a recorded run, size of the generator code for each target, and the ROS 2 interop cost (latency and CPU of bridging the Copper-rs graph to ROS 2 topics via Zenoh).
- Both targets are generated from the same unchanged system source; if either needs a dialect change, the spike records it as a finding against design answer 10.

## Acceptance sketch

- Integration test binary `issue_s1_07_runtime_core_spike` checks that both targets generate deterministically from the same source.
- A committed measurement report with raw logs, kernel version and host, and a one-paragraph recommendation; the maintainers record the decision.

## Out of scope

No production controller, no real hardware, no second arm, no removal of the losing target's generator before the decision is recorded.
