#[path = "../support/gnet_entry.rs"]
mod entry;
#[path = "../support/normal_form.rs"]
mod nf;
use entry::device as support;

mod device_entry_abi;
mod exactness_checks;
mod p2_compiled_slices;
mod robotics_device_checks;
