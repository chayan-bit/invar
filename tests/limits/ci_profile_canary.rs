//! Canary for the profile CI builds each suite with.
//!
//! The Debug suites run at `[profile.ci-debug]` (opt-level 1) so they stay fast, and they are only worth
//! running if the checks that make Debug different from Release are still compiled in. The workflow sets
//! `INVAR_CI_PROFILE` to `debug` or `release`; a local run leaves it unset and only checks consistency.
use std::hint::black_box;
use std::panic;

#[test]
fn ci_profile_matches_its_debug_assertions() {
    // `black_box` keeps Clippy from folding the cfg into a constant assertion.
    let debug_assertions = black_box(cfg!(debug_assertions));
    match std::env::var("INVAR_CI_PROFILE").as_deref() {
        Ok("debug") => assert!(
            debug_assertions,
            "the Debug suites must build with debug assertions"
        ),
        Ok("release") => assert!(
            !debug_assertions,
            "the Release suite must build without debug assertions"
        ),
        _ => {}
    }
}

#[test]
fn ci_profile_overflow_checks_follow_debug_assertions() {
    // `black_box` keeps the addition from being folded away or rejected at compile time.
    let overflowed = panic::catch_unwind(|| black_box(u32::MAX) + black_box(1u32)).is_err();
    if cfg!(debug_assertions) {
        assert!(
            overflowed,
            "u32 overflow must panic when debug assertions are on"
        );
    }
}
