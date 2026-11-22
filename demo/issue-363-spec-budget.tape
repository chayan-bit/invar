# #363: chapters.json records the current token counts, so spec_budget passes (vhs demo/issue-363-spec-budget.tape, the test toolchain on PATH).
Output demo/issue-363-spec-budget.gif
Set Shell bash
Set FontSize 14
Set Width 1400
Set Height 700
Set TypingSpeed 10ms

Hide
Type `export PS1='$ ' && spec_tests() { cargo nextest run --locked --test spec_budget --test spec_examples 2>&1 | grep -E 'PASS|FAIL|Summary'; }` Enter
Type "clear" Enter
Show

Type "# The only change: the regenerated token records of the axes and regions chapters" Enter
Type "git diff origin/main --stat && git diff origin/main | grep -E '^[-+] '" Enter
Sleep 4s
Type "clear" Enter
Type "spec_tests" Enter
Sleep 15s
