// Shared fixtures, each loaded once and used by several modules below.
#[allow(dead_code)]
#[path = "../support/fsum_cases.rs"]
mod fsum_cases_support;
mod issue_v02_f64_oracle;
mod issue_v02_f64_source;
mod issue_v02_fk_panda;
mod issue_v02_math_differential;
mod issue_v02_math_engines;
mod issue_v02_math_msl;
#[allow(dead_code)]
#[path = "../support/language_limits.rs"]
mod language_limits_support;
mod sort_fsum;
#[allow(dead_code)]
#[path = "../support/spatial_oracle.rs"]
mod spatial_oracle_support;
mod v02_fsum_cli;
mod v02_fsum_oracle;
mod v02_placement;
mod v02_sort_contract;
mod v02_sort_device;
mod v02_sort_lowering;
mod v02_spatial_checker;
mod v02_spatial_demo;
mod v02_spatial_oracle;
mod v02_spatial_placement;
mod v02_spatial_reference;
