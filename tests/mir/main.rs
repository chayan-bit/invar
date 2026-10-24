#[path = "../../test-support/term_oracle.rs"]
mod term;

// Shared by the native tests below; one module so it loads once.
#[cfg(feature = "native")]
#[path = "../support/scratch.rs"]
mod scratch;

mod corpus;
mod issue_eng_18_native_simd;
mod issue_m1_01;
mod issue_m1_01_owned_return_fuel;
mod issue_m1_02;
mod issue_m1_04;
mod issue_m1_05;
