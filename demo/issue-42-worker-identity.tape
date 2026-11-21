# #42: law-admitted regions, sort, fsum and spatial are bit-identical at 1, 2, 4, 16 and 64 CPU workers.
# Run from the repository root with the Rust toolchain on PATH: vhs demo/issue-42-worker-identity.tape
Output demo/issue-42-worker-identity.gif
Set Shell bash
Set FontSize 14
Set Width 1400
Set Height 760
Set TypingSpeed 10ms

Hide
Type "clear" Enter
Show

Type "# The sweeps now cover 1, 2, 4, 16 and 64 workers:" Enter
Type `git grep -n -E 'for workers in \[1, 2, (3, )?4' tests/family_lowerings.rs tests/fsum_math_sort_spatial` Enter
Sleep 3s
Type `cargo nextest run -p invar --test family_lowerings --test fsum_math_sort_spatial -E 'test(/equal_to_reference|sort_and_spatial|radix_differential|sort_contract|fsum_adversarial|fsum_random|spatial_reference_oracle/)' 2>&1 | grep -E 'PASS|FAIL|Summary'` Enter
Sleep 55s
