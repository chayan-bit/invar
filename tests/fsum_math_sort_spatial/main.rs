// Shared fixtures, each loaded once and used by several modules below.
mod f64_oracle;
mod f64_source;
mod fk_panda;
#[path = "../support/fsum_cases.rs"]
mod fsum_cases_support;
mod fsum_cli;
mod fsum_oracle;
#[allow(dead_code)]
#[path = "../support/language_limits.rs"]
mod language_limits_support;
mod math_cr_default;
mod math_differential;
mod math_engines;
mod math_msl;
mod placement;
mod sort_contract;
mod sort_device;
mod sort_fsum;
mod sort_lowering;
mod spatial_checker;
mod spatial_demo;
mod spatial_oracle;
#[allow(dead_code)]
#[path = "../support/spatial_oracle.rs"]
mod spatial_oracle_support;
mod spatial_placement;
mod spatial_reference;
#[path = "../support/v02_math_msl_common.rs"]
mod v02_math_msl_common_support;
