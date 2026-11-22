# #371: admitted CPU reductions past one million elements run on workers (vhs demo/issue-371-reduction.tape, this branch's invar on PATH, main's as invar-main).
Output demo/issue-371-reduction.gif
Set Shell bash
Set FontSize 15
Set Width 1400
Set Height 860
Set TypingSpeed 15ms

Hide
Type "cd demo && P=issue-371-reduction.inv && F='--fuel 4000000000' && clear" Enter
Show

Type "cat $P   # the #368 program, 4,194,304 elements" Enter
Sleep 2s
Type `invar explain $P --format json | jq -c '.regions[] | {status, chosen_plan, law: .execution.law}'` Enter
Sleep 3s
Type "time invar run $P --engine reference $F" Enter
Wait+Screen@20s /real/
Sleep 1s
Type "# main: the admitted region is refused past 1,000,000 elements and MIR interprets the loop:" Enter
Type "time invar-main run $P --engine cpu-parallel --workers 8 $F --format json | jq -c .diagnostics[0].code" Enter
Sleep 7s
Sleep 1s
Type "# this branch: published from 8 workers, same result:" Enter
Type "time invar run $P --engine cpu-parallel --workers 8 $F --format json | jq -c .result" Enter
Wait+Screen@20s /real/
Sleep 5s
