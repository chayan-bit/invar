#[path = "../../test-support/term_oracle.rs"]
mod term;

// Shared by the native tests below; one module so it loads once.
#[cfg(feature = "native")]
#[path = "../support/scratch.rs"]
mod scratch;

mod corpus;
mod mir_owner_return_fuel;
mod mir_verifier_and_evaluator;
mod native_cpu_trap_abi;
mod native_parallel_regions;
mod native_simd_and_targets;
mod typed_net_rewrite_extraction;
mod typed_net_verifier;
