# #361: if c { a } else { b } as a scalar expression (vhs demo/issue-361-if-expression.tape, invar on PATH).
Output demo/issue-361-if-expression.gif
Set Shell bash
Set FontSize 15
Set Width 1300
Set Height 900
Set TypingSpeed 20ms

Hide
Type "export PS1='$ ' && clear" Enter
Show

Type "cat examples/language/if-expression.inv" Enter
Sleep 3s
Type "invar run examples/language/if-expression.inv --engine reference" Enter
Sleep 1s
Type "invar run examples/language/if-expression.inv --engine cpu-parallel --workers 4" Enter
Sleep 2s
Type "# The unselected arm never runs; the selected one traps at its own operator:" Enter
Type "sed 's/if z == 0 {/if z != 0 {/' examples/language/if-expression.inv > /tmp/trap.inv && invar run /tmp/trap.inv --engine reference" Enter
Sleep 3s
Type "# An if expression needs an else arm:" Enter
Type "invar check examples/language/if-without-else.inv" Enter
Sleep 2s
Type "# Both arms have one type:" Enter
Type "sed 's/{ 1 } else { 100 }/{ 1 } else { 100u32 }/' examples/language/if-expression.inv > /tmp/mixed.inv && invar check /tmp/mixed.inv" Enter
Sleep 4s
