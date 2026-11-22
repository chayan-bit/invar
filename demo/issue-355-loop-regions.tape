# #355: #[region(...)] on a loop declares that loop's levels (vhs demo/issue-355-loop-regions.tape, invar and the test toolchain on PATH).
Output demo/issue-355-loop-regions.gif
Set Shell bash
Set FontSize 14
Set Width 1400
Set Height 820
Set TypingSpeed 10ms

Hide
Type `export PS1='$ ' && cd demo && P=issue-355-loop-regions.inv` Enter
Type `levels() { invar explain $1 --format json | jq -c '.regions[] | {span: .span.start, chosen_plan, c_level, n_level, contract_name}'; }` Enter
Type `loop_tests() { cargo nextest run --test numeric_tiers -E 'test(/loop_regions/)' 2>&1 | grep -E 'PASS|Summary'; }` Enter
Type "clear" Enter
Show

Type "cat $P" Enter
Sleep 3s
Type "# Each loop records its own levels: the declared loop is c1 f32-tree-v2 n1, the other keeps the function's c0" Enter
Type "levels $P && invar check $P --region-report" Enter
Sleep 4s
Type "# The declared loop folds as a tree, the other strict, on every engine (main returns 1)" Enter
Type "invar run $P --engine reference; invar run $P --engine cpu-parallel --workers 16" Enter
Sleep 3s
Type "invar check $P --max-c c0" Enter
Sleep 3s
Type "clear" Enter
Type "loop_tests" Enter
Sleep 25s
