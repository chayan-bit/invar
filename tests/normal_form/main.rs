#[path = "../support/gnet_agree.rs"]
#[allow(dead_code)]
mod agree;
#[allow(dead_code)]
#[path = "../../src/contraction_source.rs"]
mod contraction_source;
#[allow(dead_code)]
#[path = "../../src/frontier_source.rs"]
mod frontier_source;
#[path = "../../test-support/split_mix.rs"]
mod split_mix;
#[path = "../support/task_model.rs"]
mod task_model;

#[path = "../../bench/general-net-v01/cpu_tests.rs"]
mod cpu_tests;

// The `src/*_source.rs` oracles above resolve these through `crate::`.
#[allow(unused_imports)]
use invar::{
    array_limits, budget, check, front, general_net, hex, layout_contract, mir, numeric_contract,
    tensor_cache,
};
use task_model::normal_form_support as support;

mod auto_placement;
mod gnet05_run;
mod law_programs;
mod net_normal_form_annotations;
mod net_normal_form_corpus;
mod net_normal_form_followup;
mod net_normal_form_laws;
mod net_regions;
mod net_task_region;
mod normal_form_residue_seed;
mod normal_form_run;
mod path_sum;
mod pnf_family_admission;
