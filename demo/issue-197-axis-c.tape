# #197: the contract axis C chapter and its checked contract-to-level table (vhs demo/issue-197-axis-c.tape from the repo root, invar on PATH).
Output demo/issue-197-axis-c.gif
Set Shell bash
Set FontSize 14
Set Width 1400
Set Height 820
Set TypingSpeed 10ms

Hide
Type `export PS1='$ ' && C=docs/spec/invar-v0.2-axes.md && T=$(mktemp -d) && clear` Enter
Show

Type "# Every numeric contract on main maps to exactly one C level (spec table, checked against src/axes.rs):" Enter
Type `grep -E '^\| .(strict|invar|f32|tc)-' $C` Enter
Sleep 3s
Type "# One fixture per implemented level runs on the reference evaluator (C0 strict 5, C1 tree/exact 1, C2 relaxed 2):" Enter
Type `awk -v t=$T '/^.{3}invar outcome=value/{n++; f=1; next} /^.{3}$/{f=0} f{print > (t "/l" n ".inv")}' $C; for f in $T/l2.inv $T/l3.inv $T/l4.inv; do invar run $f --engine reference --format json | jq -c '{value: .result.value}'; done` Enter
Sleep 4s
Type "# The spec test runs every block and citation; the unit test proves the mapping total and equal to the table:" Enter
Type `cargo nextest run --locked --lib --test spec_examples -E 'test(/axes::|spec_examples_(every|normative|token)/)' 2>&1 | grep -E 'PASS|Summary'` Enter
Sleep 12s
