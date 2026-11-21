# #254: compound assignment (vhs demo/issue-254-compound.tape, invar on PATH).
Output demo/issue-254-compound.gif
Set Shell bash
Set FontSize 15
Set Width 1200
Set Height 800
Set TypingSpeed 20ms

Hide
Type "cd demo && clear" Enter
Show

Type "cat issue-254-compound.inv" Enter
Sleep 2s
Type "invar run issue-254-compound.inv --engine reference && invar run issue-254-compound.inv --engine cpu-parallel --workers 4" Enter
Sleep 3s
Type "# A computed index is evaluated once, before the right side; here it is out of bounds:" Enter
Type "sed 's/k + 1u32/k + 9u32/' issue-254-compound.inv > /tmp/oob.inv && invar run /tmp/oob.inv --engine reference --format json | jq -c .trap" Enter
Sleep 3s
Type "# Compound assignment needs a place, like assignment:" Enter
Type "sed 's/acc %= 7u32;/len(ys) += 1u32;/' issue-254-compound.inv > /tmp/bad.inv && invar check /tmp/bad.inv" Enter
Sleep 4s
