# #262: pairs of scalars and pair binding (vhs demo/issue-262-pairs.tape, invar on PATH).
Output demo/issue-262-pairs.gif
Set Shell bash
Set FontSize 15
Set Width 1200
Set Height 800
Set TypingSpeed 20ms

Hide
Type "cd demo && clear" Enter
Show

Type "cat issue-262-pairs.inv" Enter
Sleep 2s
Type "invar run issue-262-pairs.inv --engine reference" Enter
Sleep 2s
Type "invar run issue-262-pairs.inv --engine cpu-parallel --workers 4" Enter
Sleep 2s
Type "# Three elements are refused:" Enter
Type "sed 's/a % b)/a % b, 0)/' issue-262-pairs.inv > /tmp/bad.inv" Enter
Type "invar check /tmp/bad.inv" Enter
Sleep 4s
