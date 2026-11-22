# #362: a pair outside a loop keeps the loop admitted (vhs demo/issue-362-pair-admission.tape, invar on PATH).
Output demo/issue-362-pair-admission.gif
Set Shell bash
Set FontSize 15
Set Width 1300
Set Height 600
Set TypingSpeed 20ms

Hide
Type "export PS1='$ ' && clear" Enter
Show

Type "cat demo/issue-362-pair-admission.inv" Enter
Sleep 2s
Type `invar explain demo/issue-362-pair-admission.inv --format json | grep -o -E '"(status|chosen_plan)":"[^"]*"'` Enter
Sleep 2s
Type "invar run demo/issue-362-pair-admission.inv --engine reference" Enter
Sleep 1s
Type "invar run demo/issue-362-pair-admission.inv --engine cpu-parallel --workers 4" Enter
Sleep 4s
