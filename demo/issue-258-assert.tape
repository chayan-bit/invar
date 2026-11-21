# #258: assert with a message (vhs demo/issue-258-assert.tape, invar on PATH).
Output demo/issue-258-assert.gif
Set Shell bash
Set FontSize 15
Set Width 1200
Set Height 800
Set TypingSpeed 20ms

Hide
Type "cd demo && clear" Enter
Show

Type "cat issue-258-assert.inv" Enter
Sleep 2s
Type "invar run issue-258-assert.inv --engine reference; echo exit=$?" Enter
Sleep 3s
Type "invar run issue-258-assert.inv --engine cpu-parallel --workers 4 --format json | jq -c '.trap, .diagnostics[0].message'" Enter
Sleep 3s
Type "# The message must be a string literal:" Enter
Type `sed 's/"mean rounds down"/xs/' issue-258-assert.inv > /tmp/bad.inv && invar check /tmp/bad.inv` Enter
Sleep 4s
