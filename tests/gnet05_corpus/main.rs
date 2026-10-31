#[path = "../support/gnet_agree.rs"]
mod agree;
#[cfg(feature = "cuda")]
#[path = "../support/gnet_cuda_aot.rs"]
mod aot;
#[path = "../support/gnet_device.rs"]
mod device;
#[path = "../../bench/general-net-v01/gnet05/generate.rs"]
mod generate;
#[path = "../support/gnet_corpus_guard.rs"]
mod guard;
#[path = "../support/gnet_par.rs"]
mod par;
#[path = "../support/gnet_parallel.rs"]
mod parallel;
#[path = "../../bench/general-net-v01/gnet05/robotics/mod.rs"]
mod robotics;
#[path = "../support/gnet_robotics_checks.rs"]
mod robotics_checks;

mod corpus;
mod robotics;
mod robotics_array;
