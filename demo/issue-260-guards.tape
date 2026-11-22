# #260: match guards and let-else (vhs demo/issue-260-guards.tape, invar on PATH).
Output demo/issue-260-guards.gif
Set Shell bash
Set FontSize 15
Set Width 1300
Set Height 900
Set TypingSpeed 20ms

Hide
Type "export PS1='$ ' && clear" Enter
Show

Type "cat examples/language/match-guards.inv" Enter
Sleep 3s
Type "invar run examples/language/match-guards.inv --engine reference" Enter
Sleep 1s
Type "invar run examples/language/match-guards.inv --engine cpu-parallel --workers 4" Enter
Sleep 2s
Type "# A let-else block that can fall through is refused:" Enter
Type "invar check examples/language/let-else-fallthrough.inv" Enter
Sleep 2s
Type "# A guarded arm never covers its case:" Enter
Type "sed '/Celsius(v) => 3,/d' examples/language/match-guards.inv > /tmp/partial.inv && invar check /tmp/partial.inv" Enter
Sleep 4s
