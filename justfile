set positional-arguments

# CPU test defaults match CI; explicit environment values remain authoritative.

# Format, lint, the full Debug/Release suites and documentation tests.
check: fmt lint test

# Fast inner loop; build only the binaries selected by the quick profile.
test-quick:
    NEXTEST_TEST_THREADS="${NEXTEST_TEST_THREADS-8}" INVAR_TEST_TIMEOUT_SCALE="${INVAR_TEST_TIMEOUT_SCALE-8}" cargo nextest run --locked --profile quick --lib --test 'issue_m0_0[2-9]' --test 'issue_l1_0[1-3]*' --test issue_gnet_01

# Full coverage, including tests intentionally excluded from the quick loop.
test:
    NEXTEST_TEST_THREADS="${NEXTEST_TEST_THREADS-8}" INVAR_TEST_TIMEOUT_SCALE="${INVAR_TEST_TIMEOUT_SCALE-8}" cargo nextest run --locked --profile ci
    NEXTEST_TEST_THREADS="${NEXTEST_TEST_THREADS-8}" INVAR_TEST_TIMEOUT_SCALE="${INVAR_TEST_TIMEOUT_SCALE-8}" cargo nextest run --locked --profile ci --release
    cargo test --locked --doc

# CUDA hardware suites; needs an NVIDIA GPU with no other heavy work running.
test-gpu-cuda:
    cargo test --locked --features cuda -- --include-ignored

# Apple GPU suites; needs an Apple Silicon Mac with no other heavy work running.
test-metal:
    cargo test --locked --features metal -- --include-ignored

# Check formatting without rewriting files.
fmt:
    cargo fmt --all -- --check

# Default-feature lint, matching CI.
lint:
    cargo clippy --locked --all-targets -- -D warnings

# Ratchet on .rs files over 800 lines (scripts/file-size-allowlist.txt); `just size --update` lowers entries.
size *args:
    scripts/check-file-size {{args}}
