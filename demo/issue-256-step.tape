# #256: ranges with a step (vhs demo/issue-256-step.tape, invar on PATH).
Output demo/issue-256-step.gif
Set Shell bash
Set FontSize 15
Set Width 1200
Set Height 800
Set TypingSpeed 20ms

Hide
Type "cd demo && clear" Enter
Show

Type "cat issue-256-step.inv" Enter
Sleep 2s
Type "invar run issue-256-step.inv --engine reference && invar run issue-256-step.inv --engine cpu-parallel --workers 4" Enter
Sleep 3s
Type "# A literal zero step is refused before checking:" Enter
Type "sed 's/step 2u32/step 0u32/' issue-256-step.inv > /tmp/bad.inv && invar check /tmp/bad.inv" Enter
Sleep 4s
