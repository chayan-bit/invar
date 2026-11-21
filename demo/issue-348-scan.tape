# #348: the normal-form executor's prefix scan in linear work (vhs demo/issue-348-scan.tape, this branch's invar on PATH, main's as invar-main).
Output demo/issue-348-scan.gif
Set Shell bash
Set FontSize 15
Set Width 1400
Set Height 800
Set TypingSpeed 15ms

Hide
Type "cd demo && P=issue-348-argmax.inv && F='--fuel 4000000000' && clear" Enter
Show

Type "cat $P" Enter
Sleep 2s
Type "# 2^20 elements through the normal-form route: law L7 applied, same value as the reference:" Enter
Type `invar run $P --engine cpu-parallel --workers 16 $F --format json | jq -c '{result: .result.value, normal_form: .normal_form | {status, laws}}'` Enter
Sleep 4s
Type "invar run $P --engine reference $F" Enter
Sleep 2s
Type "# Wall time before (main) and after (this branch):" Enter
Type "time invar-main run $P --engine cpu-parallel --workers 16 $F" Enter
Sleep 5s
Type "time invar run $P --engine cpu-parallel --workers 16 $F" Enter
Sleep 5s
