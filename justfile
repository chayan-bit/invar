set positional-arguments

# CPU test defaults match CI; explicit environment values remain authoritative.

# Format, lint, the full Debug/Release suites and documentation tests.
check: fmt lint test

# Fast inner loop; build only the binaries selected by the quick profile.
test-quick:
    NEXTEST_TEST_THREADS="${NEXTEST_TEST_THREADS-8}" INVAR_TEST_TIMEOUT_SCALE="${INVAR_TEST_TIMEOUT_SCALE-8}" cargo nextest run --locked --profile quick --lib --test language --test gnet_net

# Full coverage, including tests intentionally excluded from the quick loop and the
# whole corpus where pull-request CI runs a deterministic 1-in-8 sample.
test:
    NEXTEST_TEST_THREADS="${NEXTEST_TEST_THREADS-8}" INVAR_TEST_TIMEOUT_SCALE="${INVAR_TEST_TIMEOUT_SCALE-8}" INVAR_FULL_CORPUS="${INVAR_FULL_CORPUS-1}" cargo nextest run --locked --profile ci -p invar -p invar-tools
    NEXTEST_TEST_THREADS="${NEXTEST_TEST_THREADS-8}" INVAR_TEST_TIMEOUT_SCALE="${INVAR_TEST_TIMEOUT_SCALE-8}" INVAR_FULL_CORPUS="${INVAR_FULL_CORPUS-1}" cargo nextest run --locked --profile ci -p invar -p invar-tools --release
    cargo test --locked --doc

# CUDA hardware suites; needs an NVIDIA GPU with no other heavy work running.
test-gpu-cuda:
    cargo test --locked -p invar -p invar-tools --features cuda -- --include-ignored

# Apple GPU suites; needs an Apple Silicon Mac with no other heavy work running.
test-metal:
    cargo test --locked -p invar -p invar-tools --features metal -- --include-ignored

# Check formatting without rewriting files.
fmt:
    cargo fmt --all -- --check

# Default-feature lint, matching CI.
lint:
    cargo clippy --locked -p invar -p invar-tools --all-targets -- -D warnings

# Ratchet on .rs files over 800 lines (scripts/file-size-allowlist.txt); `just size --update` lowers entries.
size *args:
    scripts/check-file-size {{args}}
