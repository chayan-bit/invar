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

mod issue_eng_10_domain_suites;
mod issue_gnet_05_parity_report;
mod issue_m3_03_cold_process;
mod issue_m5_01_benchmark_manifest;
mod issue_m5_01_build_lineage;
mod issue_m5_01_heldout_generators;
mod issue_m5_02_device_qualification;
mod issue_m5_02_full_pipeline_qualification;
mod issue_m5_03_cuda_performance_acceptance;
mod qualification_faults;
mod qualify_cli;
