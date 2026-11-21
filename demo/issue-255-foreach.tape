# #255: for-each loops (vhs demo/issue-255-foreach.tape, invar on PATH).
Output demo/issue-255-foreach.gif
Set Shell bash
Set FontSize 15
Set Width 1200
Set Height 860
Set TypingSpeed 20ms

Hide
Type "cd demo && clear" Enter
Show

Type "cat issue-255-foreach.inv" Enter
Sleep 2s
Type "invar run issue-255-foreach.inv --engine reference && invar run issue-255-foreach.inv --engine cpu-parallel --workers 4" Enter
Sleep 3s
Type "# A mutable iteration cannot read its owner, exactly as for tail[i] = tail[i] * xs[0u32]:" Enter
Type "sed 's/t [*]= 10u32/t *= xs[0u32]/' issue-255-foreach.inv > /tmp/bad.inv && invar check /tmp/bad.inv" Enter
Sleep 4s
