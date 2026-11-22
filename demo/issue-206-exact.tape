# #206: sum_exact and dot_exact split over any partition (vhs demo/issue-206-exact.tape, invar on PATH,
# BEFORE and AFTER naming `fsum-cpu-bench curve` outputs from main and from this branch).
Output demo/issue-206-exact.gif
Set Shell bash
Set FontSize 15
Set Width 1400
Set Height 820
Set TypingSpeed 12ms

Hide
Type "cd demo && P=issue-206-exact.inv && clear" Enter
Show

Type "cat $P" Enter
Sleep 2s
Type "# Each split reduction names the law that admitted it:" Enter
Type `invar run $P --engine auto --trust-provisional --fuel 100000000 --format json | jq -c '[.placement.reductions[] | {kind, backend, law}]'` Enter
Sleep 4s
Type "# Same result on every worker count (0: every check passed):" Enter
Type `for w in 1 2 7 64; do printf 'workers %-3s' $w; invar run $P --engine cpu-parallel --workers $w --fuel 100000000 --format json | jq -c .result; done` Enter
Sleep 12s
Type "clear; # 2^26 terms, ms by forced worker count: main (left) and this PR (right):" Enter
Type `paste <(grep 'sum_exact.f32 n=2^26' $BEFORE | cut -d' ' -f2-5) <(grep 'sum_exact.f32 n=2^26' $AFTER | cut -d' ' -f5,6)` Enter
Sleep 3s
Type `paste <(grep 'dot_exact.f32 n=2^26' $BEFORE | cut -d' ' -f2-5) <(grep 'dot_exact.f32 n=2^26' $AFTER | cut -d' ' -f5,6)` Enter
Sleep 6s
