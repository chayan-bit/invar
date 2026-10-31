//! Acceptance tests of the `invar-tools` benchmark, qualification and report tooling, in one
//! binary so the shared harnesses (`qualification/`, `bench/hash.rs`, `bench/corpus.rs`,
//! `bench/heldout.rs`) compile once.
#![forbid(unsafe_code)]

#[path = "../../bench/corpus.rs"]
mod corpus;
#[path = "../../bench/hash.rs"]
mod hash;
#[path = "../../bench/heldout.rs"]
mod heldout;
#[path = "../support/invar_bin.rs"]
mod invar_bin;
#[path = "../../qualification/mod.rs"]
mod qualification;

mod benchmark_manifest;
mod build_lineage;
mod cold_process;
mod cuda_performance_acceptance;
mod device_qualification;
mod domain_suites;
mod full_pipeline_qualification;
mod gnet_parity_report;
mod heldout_generators;
mod qualification_faults;
mod qualify_cli;
