# S1-10 - Timing analysis export to AADL and osate-cli

Release: v0.3 (milestone S1); planning contract, to be expanded into a full contract before implementation.

Dependencies: S1-03.

## Goal

Export the system's timing model to AADL, run osate-cli flow-latency, bus-load and binding analyses headless, and fold the results into Invar diagnostics.

## Motivation

S1-03 proves the carried subset of AADL semantics directly; osate-cli adds independent analyses (bus load, binding) and a second opinion OEMs recognize.

## Scope

- Generate an AADL model (threads with period, deadline and compute execution time from `wcet`, connections with latency, flows with budgets, buses from product capacity) from `invar-system-v1`.
- Run osate-cli (0.2.1 or newer, pinned) and parse its text, XML and CSV output into `invar-diagnostics-v1` with codes in `S1900` to `S1999`.
- Differential check: every S1-03 flow result agrees with the osate flow-latency result on the fixture set.

## Acceptance sketch

- Integration test binary `issue_s1_10_aadl_osate` with generation golden files, and an environment-gated run of osate-cli on the S1-06 cell whose results agree with S1-03.

## Out of scope

AADL import, AADL annex languages, schedulability analysis beyond what osate-cli provides headless.
