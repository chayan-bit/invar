# #345: a region's C level gates which rules its plan may use, and explain records each rule's level (vhs demo/issue-345-c-level-gate.tape, invar and the test toolchain on PATH).
Output demo/issue-345-c-level-gate.gif
Set Shell bash
Set FontSize 14
Set Width 1400
Set Height 820
Set TypingSpeed 10ms

Hide
Type `export PS1='$ ' && cd demo && P=issue-345-c-level.inv` Enter
Type `rules() { invar explain $1 --format json | jq -c '.regions[] | {chosen_plan, c_level, rule_identities, rule_levels}'; }` Enter
Type `gate_tests() { cargo nextest run --lib -E 'test(/axes::rules/)' 2>&1 | grep -E 'PASS|Summary'; }` Enter
Type "clear" Enter
Show

Type "head -1 $P && rules $P" Enter
Sleep 4s
Type "# Every rule on main is c0; the spec table and the code are checked row for row" Enter
Type "grep -A9 '^| Rule | Level | Plan |' ../docs/spec/invar-v0.2-regions.md" Enter
Sleep 4s
Type "clear" Enter
Type "# A rule tagged c2 in the tests: refused at c0 and c1 with CONTRACT_LEVEL, serial bits, trap and fuel; admitted at c2" Enter
Type "gate_tests" Enter
Sleep 20s
