#![forbid(unsafe_code)]
#[allow(dead_code)]
#[path = "../../test-support/child_deadline.rs"]
mod child_deadline;

#[path = "corpus_cpu.rs"]
mod corpus_cpu;

#[test]
fn full_ordinary_source_denominator() {
    corpus_cpu::ordinary_sources();
}

#[test]
fn independent_runtime_input_vectors() {
    corpus_cpu::runtime_inputs();
}

#[test]
fn owner_source_fuel_and_cleanup_prefixes() {
    corpus_cpu::owner_fuel_prefixes();
}

#[path = "corpus_owned.rs"]
mod corpus_owned;

#[test]
fn full_owned_and_resumable_denominator() {
    corpus_owned::full_denominator();
}

#[test]
fn compiled_owner_prefixes_and_returned_roots() {
    corpus_owned::owner_prefixes_and_returned_roots();
}

#[test]
fn fair_resumed_roots_and_distinct_backend_budgets() {
    corpus_owned::fair_worklist_and_backend_budgets();
}

#[path = "corpus_children.rs"]
mod corpus_children;

#[test]
fn child_extraction_full_denominator() {
    corpus_children::full_denominator();
}

#[test]
fn child_failure_prefixes_and_source_order() {
    corpus_children::failures_and_source_order();
}

#[test]
fn child_live_workset_continuation_and_fence_admission() {
    corpus_children::workset_and_fence_admission();
}

#[test]
fn child_source_reference_vectors() {
    corpus_children::source_reference_vectors();
}

#[path = "corpus_numeric.rs"]
mod corpus_numeric;
#[test]
fn numeric_full_scalar_denominator() {
    corpus_numeric::full_denominator();
}
#[test]
fn numeric_rounding_environment() {
    corpus_numeric::rounding_environment();
}
#[test]
fn numeric_source_failure_fuel_and_admission() {
    corpus_numeric::fuel_and_admission();
}

#[test]
fn numeric_environment_capabilities_keep_fenv_without_sse() {
    corpus_numeric::environment_capabilities();
}

#[test]
fn child_extended_record_entry_cpu_models() {
    corpus_children::extended_record_entry_cpu_models();
}
