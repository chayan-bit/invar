# #261: break and continue with loop labels (vhs demo/issue-261-loops.tape, invar on PATH).
Output demo/issue-261-loops.gif
Set Shell bash
Set FontSize 15
Set Width 1200
Set Height 800
Set TypingSpeed 20ms

Hide
Type "cd demo && clear" Enter
Show

Type "cat issue-261-loops.inv" Enter
Sleep 2s
Type "invar run issue-261-loops.inv --engine reference && invar run issue-261-loops.inv --engine cpu-parallel --workers 4" Enter
Sleep 3s
Type "# A label that names no enclosing loop is refused:" Enter
Type "sed 's/break outer/break rows/' issue-261-loops.inv > /tmp/bad.inv && invar check /tmp/bad.inv" Enter
Sleep 4s
