#[allow(dead_code)]
#[path = "../../test-support/child_deadline.rs"]
mod child_deadline;
#[path = "../support/gnet_device.rs"]
mod device;
#[path = "../support/executable.rs"]
mod executable;
#[path = "../support/gnet_parallel.rs"]
mod parallel;

mod issue_gnet_03_device;
mod issue_gnet_03_injection;
mod issue_gnet_04_device;
mod issue_gnet_12_cuda_owned_numeric_model;
mod issue_gnet_resume_device_model;
