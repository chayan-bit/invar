# #207: L7 select loops on CPU workers (vhs demo/issue-207-select.tape, this branch's invar on PATH, main's as invar-main).
Output demo/issue-207-select.gif
Set Shell bash
Set FontSize 15
Set Width 1400
Set Height 860
Set TypingSpeed 15ms

Hide
Type "cd demo && P=issue-207-argmin.inv && F='--fuel 4000000000' && clear" Enter
Show

Type "cat $P" Enter
Sleep 2s
Type "# The receipt: admitted as a reduction by law L7, with its evidence:" Enter
Type `invar explain $P --format json | jq -c '.regions[] | select(.chosen_plan) | {status, chosen_plan, law: .execution.law, rule_identities, evidence: [.obligations[] | .name + ":" + .evidence]}'` Enter
Sleep 4s
Type "# Same index as the reference evaluator (NaN skipped, first of the tied minima):" Enter
Type "invar run $P --engine reference $F" Enter
Sleep 3s
Type "invar run $P --engine cpu-parallel --workers 16 $F" Enter
Sleep 2s
Type "# Wall time before (main) and after (this branch), 2^22 elements:" Enter
Type "time invar-main run $P --engine cpu-parallel --workers 16 $F" Enter
Wait+Screen@20s /real/
Sleep 1s
Type "time invar run $P --engine cpu-parallel --workers 16 $F" Enter
Sleep 5s
