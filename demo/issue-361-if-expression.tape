# #361: if c { a } else { b } as a scalar expression (vhs demo/issue-361-if-expression.tape from the repository root, invar on PATH).
Output demo/issue-361-if-expression.gif
Set Shell bash
Set FontSize 14
Set Width 1600
Set Height 720
Set LetterSpacing 0
Set TypingSpeed 20ms

Hide
Type "export PS1='$ ' && cd examples/language && clear" Enter
Show

Type "cat if-expression.inv" Enter
Sleep 3s
Type "invar run if-expression.inv --engine reference" Enter
Sleep 1s
Type "invar run if-expression.inv --engine cpu-parallel --workers 4" Enter
Sleep 2s
Type "# The unselected arm never runs; the selected one traps at its own operator:" Enter
Type "sed 's/if z == 0 {/if z != 0 {/' if-expression.inv >/tmp/t.inv && invar run /tmp/t.inv --engine reference" Enter
Sleep 3s
Type "# An if expression needs an else arm:" Enter
Type "invar check if-without-else.inv" Enter
Sleep 2s
Type "# Both arms have one type:" Enter
Type "sed 's/{ 1 } else { 100 }/{ 1 } else { 100u32 }/' if-expression.inv >/tmp/m.inv && invar check /tmp/m.inv" Enter
Sleep 4s
