# #344: a for-each fold admits the plan of its counted form (vhs demo/issue-344-for-each.tape, invar and the test toolchain on PATH).
Output demo/issue-344-for-each.gif
Set Shell bash
Set FontSize 14
Set Width 1400
Set Height 820
Set TypingSpeed 10ms

Hide
Type `export PS1='$ ' && cd demo && P=issue-344-for-each.inv` Enter
Type `plans() { invar explain $1 --format json | jq -c '.regions[] | {function, chosen_plan, rule_identities, first_rejected_obligation}'; }` Enter
Type `for_each_tests() { cargo nextest run --test numeric_tiers -E 'test(/for_each_plans/)' 2>&1 | grep -E 'PASS|Summary'; }` Enter
Type "clear" Enter
Show

Type "head -10 $P" Enter
Sleep 3s
Type "# The for-each fold and its counted form admit the same sealed wrapping reduction" Enter
Type "plans $P" Enter
Sleep 4s
Type "# A for-each f32 fold under f32-tree-v2 now compiles and folds as a tree (main returns 1)" Enter
Type "invar run $P --engine reference --format json | jq -c .result; invar run $P --engine cpu-parallel --workers 16 --format json | jq -c .result" Enter
Sleep 4s
Type "clear" Enter
Type "for_each_tests" Enter
Sleep 20s
