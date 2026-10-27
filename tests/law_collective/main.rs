#[allow(dead_code)]
#[path = "../../test-support/child_deadline.rs"]
mod child_deadline;
// Shared by both modules; one declaration, so clippy::duplicate_mod holds.
#[path = "../support/law_collective.rs"]
mod law_support;

mod net_law_collective;
mod net_law_collective_bench;
