# #207: min_of, max_of, argmin and argmax (vhs demo/issue-207-extrema.tape, this branch's invar on PATH).
Output demo/issue-207-extrema.gif
Set Shell bash
Set FontSize 15
Set Width 1400
Set Height 860
Set TypingSpeed 15ms

Hide
Type "cd demo && P=issue-207-extrema.inv && E=issue-207-extrema-empty.inv && F='--fuel 4000000000' && clear" Enter
Show

Type "cat $P" Enter
Sleep 2s
Type "# 2^22 elements: reference evaluator, then 16 CPU workers (same index, NaN ignored, -0 < +0, lowest index):" Enter
Type "time invar run $P --engine reference $F" Enter
Wait+Screen@30s /real/
Sleep 1s
Type "time invar run $P --engine cpu-parallel --workers 16 $F" Enter
Sleep 3s
Sleep 2s
Type "clear && cat $E" Enter
Sleep 1s
Type "# An empty input traps at the call on every engine:" Enter
Type "invar run $E --engine reference" Enter
Sleep 2s
Type "invar run $E --engine cpu-parallel --workers 16" Enter
Sleep 4s
