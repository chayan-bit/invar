# #204: checked integer sums in parallel without per-lane replay (vhs demo/issue-204-checked-sum.tape, invar on PATH).
Output demo/issue-204-checked-sum.gif
Set Shell bash
Set FontSize 15
Set Width 1400
Set Height 800
Set TypingSpeed 15ms

Hide
Type "cd demo && P=issue-204-checked-sum.inv && clear" Enter
Show

Type "cat $P" Enter
Sleep 2s
Type "# explain names the plan and the L3 carry rule:" Enter
Type `invar explain $P --format json | jq -c '.regions[0] | {status, chosen_plan, law: .execution.law, rule_identities}'` Enter
Sleep 3s
Type "# 2^23 elements: cpu-parallel on 16 workers vs the reference evaluator, same value:" Enter
Type "time invar run $P --engine cpu-parallel --workers 16 --fuel 200000000" Enter
Sleep 3s
Type "time invar run $P --engine reference --fuel 200000000" Enter
Sleep 8s
Type "clear; cat issue-204-overflow.inv" Enter
Sleep 2s
Type "# The first overflow wins at every worker count, at the reference trap span:" Enter
Type `for e in reference 'cpu-parallel --workers 1' 'cpu-parallel --workers 7' 'cpu-parallel --workers 64'; do printf '%-26s' "$e"; invar run issue-204-overflow.inv --engine $e --format json | jq -c '.trap'; done` Enter
Sleep 5s
Type "# Refusal: a[i] * 2 is not the L3 sum shape, so the loop stays serial:" Enter
Type `invar explain issue-204-refused.inv --format json | jq -c '.regions[0] | {status, chosen_plan, first_rejected_obligation}'` Enter
Sleep 4s
