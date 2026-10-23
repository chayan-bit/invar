//! General-net entry corpus on a real Metal device against the CPU oracle.
#[path = "../support/gnet_entry.rs"]
mod entry;

#[test]
#[ignore = "requires Apple Metal hardware"]
fn metal_entry_matches_cpu_oracle_bit_for_bit() {
    entry::corpus(
        &mut invar::general_net::device_host::metal::Metal::open_runtime_source().unwrap(),
        invar::general_net::compiled::Target::Metal,
    );
}

#[test]
#[ignore = "requires Apple Metal hardware and a bounded 2^24-element allocation"]
fn metal_entry_accepts_two_to_24_host_elements() {
    entry::large(
        &mut invar::general_net::device_host::metal::Metal::open_runtime_source().unwrap(),
        invar::general_net::compiled::Target::Metal,
    );
}
