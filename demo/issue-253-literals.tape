# #253: integer literal separators and radix prefixes (vhs demo/issue-253-literals.tape, invar on PATH).
Output demo/issue-253-literals.gif
Set Shell bash
Set FontSize 15
Set Width 1200
Set Height 640
Set TypingSpeed 20ms

Hide
Type "cd demo && clear" Enter
Show

Type "cat issue-253-literals.inv" Enter
Sleep 2s
Type "invar run issue-253-literals.inv --engine reference && invar run issue-253-literals.inv --engine cpu-parallel --workers 4" Enter
Sleep 3s
Type "# A doubled separator is a lexical error with its exact span:" Enter
Type "sed 's/1_000_000u32/1__000u32/' issue-253-literals.inv > /tmp/bad.inv && invar check /tmp/bad.inv --format json | jq -c '.diagnostics[0] | {code, message, line: .span.start_line, col: .span.start_column}'" Enter
Sleep 4s
