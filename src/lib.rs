#![forbid(unsafe_code)]

pub mod admission;
mod angle;
pub mod array_limits;
pub mod bounded_summary;
pub mod budget;
pub mod check;
pub mod compaction_source;
pub mod contraction_source;
mod counted;
pub mod device_cache;
pub mod device_ir;
pub mod diagnostics;
pub mod engine;
pub mod eval;
pub mod exports;
pub mod f32_source;
pub mod fixes;
pub mod front;
pub mod frontier_source;
pub mod fsum;
pub mod fsum_reference;
pub mod general_net;
pub mod hex;
pub mod histogram_source;
pub mod indexed_device;
pub mod indexed_source;
pub mod laws;
pub mod laws_synth;
pub mod layout_contract;
pub mod loops;
pub mod math;
pub mod metal_aot;
pub mod mir;
pub mod mir_eval;
pub mod mir_net;
#[cfg(feature = "native")]
pub mod native;
#[cfg(any(feature = "native", test))]
mod native_foreign_source;
pub mod numeric_contract;
pub mod ownership;
pub mod pair_reduction;
pub mod placement;
pub mod plan;
pub mod recurrence;
pub mod robotics;
pub mod scatter_proof;
pub mod schedule;
pub mod segmented_device;
pub mod segmented_source;
mod sort_cpu;
pub mod sorting;
pub mod spatial;
mod spatial_cell;
pub mod stencil_source;
pub mod summary_source;
pub mod system;
pub mod tensor;
pub mod tensor_cache;
pub mod term;
pub mod transpose_source;
pub mod typed_net;

/// Help text for the current command-line interface.
pub fn help() -> &'static str {
    "Usage: invar --help\n       invar calibrate [--out NEW_PATH] [--family placement.wrapping_add1.u32] [--host-class CLASS]\n       invar check PATH [--format json]\n       invar check PATH --fix [--write]\n       invar import mjcf FILE\n       invar import urdf FILE\n       invar component generate MANIFEST --output NEW_DIRECTORY\n       invar component test MANIFEST [--library PATH] [--output DIR] [--model PATH] [--child PATH] [--planner-runtime PATH]\n       invar system check PATH --format json\n       invar system ir PATH\n       invar system report PATH [--run-evidence JSON]\n       invar system gen ros2 PATH --out NEW_DIRECTORY\n       invar system gen mjcf PATH --out NEW_DIRECTORY\n       invar system gen urdf PATH --out NEW_DIRECTORY\n       invar explain PATH [--engine cuda|metal|auto] [--artifact METAL_LIBRARY] [--profile PROFILE] [--host-class CLASS] [--trust-provisional] --format json\n       invar run PATH --engine reference|native-serial|cuda|metal|auto [--fuel N] [--format json] [--c-library ABSOLUTE_PATH] [--artifact METAL_LIBRARY] [--profile PROFILE] [--host-class CLASS] [--trust-provisional]\n       invar run PATH --engine cpu-parallel --workers 1..64 [--fuel N] [--format json]\n       invar run PATH --debug [--engine native-serial] [--fuel N] [--format json] [--c-library ABSOLUTE_PATH]\n       invar debug PATH [--gdb|--lldb] [--fuel N] [--c-library ABSOLUTE_PATH] [-- DEBUGGER_ARGS]\n       invar source-device PATH tensor|transpose ENTRY cpu-model|cuda|metal FUEL [--artifact METAL_LIBRARY]\n       invar gnet-device PATH ENTRY cpu-model-cuda|cpu-model-metal|cuda|metal ARGS_JSON CONFIG_JSON [--artifact METAL_LIBRARY]\n       invar build PATH --target native --output PATH [--c-library ABSOLUTE_PATH] [--debug]\n       invar build PATH --target metal [--entry ENTRY] --output PATH\n       invar build PATH --target metal --entry ENTRY --family tensor --fuel FUEL --output PATH\n       invar build PATH --target cuda --entry ENTRY --sm SM --output FILE\n"
}

#[cfg(test)]
#[allow(dead_code)]
#[path = "../test-support/child_deadline.rs"]
mod child_deadline;

#[cfg(test)]
mod tests {
    #[test]
    fn help_names_the_command() {
        assert!(super::help().starts_with("Usage: invar --help\n"));
        assert!(
            super::help()
                .contains("invar build PATH --target cuda --entry ENTRY --sm SM --output FILE")
        );
    }
}
