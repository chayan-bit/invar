//! The `invar-foreign` source embedded in the standalone native host.
//!
//! The host embeds only the L1-04 C-call boundary; each separately loaded
//! module is compiled through its owning crate path, so its `pub mod` line is
//! stripped. This module is also compiled for default-features unit tests, so
//! a new `invar-foreign` module that is missing here (the `79e1eba` class of
//! drift, otherwise visible only under the `native` feature) fails default CI.

const LIBRARY: &str = include_str!("../crates/invar-foreign/src/lib.rs");
const SEPARATE_MODULES: [&str; 8] = [
    "component",
    "device",
    "exports",
    "host_array",
    "numeric",
    "numeric_owned",
    "numeric_resume",
    "numeric_owned_resume",
];

/// `invar-foreign/src/lib.rs` without its separately loaded module declarations.
pub(crate) fn standalone() -> String {
    SEPARATE_MODULES
        .iter()
        .fold(LIBRARY.to_owned(), |source, module| {
            source.replacen(&format!("pub mod {module};"), "", 1)
        })
}

#[cfg(test)]
mod tests {
    use super::{LIBRARY, SEPARATE_MODULES, standalone};

    /// Lines that declare an out-of-line module (`mod name;`), which the
    /// single-file host compilation cannot resolve (E0583).
    fn module_declarations(source: &str) -> Vec<&str> {
        source
            .lines()
            .map(str::trim)
            .filter(|line| !line.starts_with("//"))
            .filter(|line| {
                let tokens = line.split_whitespace().collect::<Vec<_>>();
                tokens
                    .windows(2)
                    .any(|pair| pair[0] == "mod" && pair[1].ends_with(';'))
            })
            .collect()
    }

    #[test]
    fn issue_l1_04_embedded_foreign_source_keeps_no_module_declarations() {
        let remaining = standalone();
        assert_eq!(
            module_declarations(&remaining),
            Vec::<&str>::new(),
            "add each new invar-foreign module to SEPARATE_MODULES"
        );
        // Every stripped entry still names a real declaration, so the list
        // cannot silently go stale, and the detector itself sees all of them.
        assert_eq!(module_declarations(LIBRARY).len(), SEPARATE_MODULES.len());
        for module in SEPARATE_MODULES {
            assert!(
                LIBRARY.contains(&format!("pub mod {module};")),
                "{module} is no longer declared by invar-foreign"
            );
        }
        // The C-call boundary itself is retained.
        assert!(remaining.contains("pub const ABI_VERSION: u32"));
    }
}
