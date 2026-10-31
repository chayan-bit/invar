// Shared fixtures, each loaded once and used by several modules below.
mod check_fix;
mod debug_command;
mod dwarf_debug_info;
mod gdb_helper;
mod machine_diagnostics;
#[allow(dead_code)]
#[path = "../support/scratch.rs"]
mod scratch_support;
