# #198: #[region(...)] declares C, S, N and T per function; misuse is refused with a code (vhs demo/issue-198-region.tape, invar on PATH).
Output demo/issue-198-region.gif
Set Shell bash
Set FontSize 14
Set Width 1400
Set Height 820
Set TypingSpeed 10ms

Hide
Type `export PS1='$ ' && cd demo && P=issue-198-region.inv && T=$(mktemp -d)` Enter
Type `try() { sed "1s/.*/$1/" $P > $T/a.inv; invar check $T/a.inv --format json | jq -c '.diagnostics[0] | {code, message}'; }` Enter
Type "clear" Enter
Show

Type "cat $P" Enter
Sleep 2s
Type "invar run $P --engine reference --format json | jq -c '{value: .result.value}'   # 3: the f32-exact-v2 sum ran" Enter
Sleep 2s
Type "invar check $P --region-report" Enter
Sleep 3s
Type "# Unknown level, unimplemented level, level without a contract, contract at the wrong level:" Enter
Type `try '#[region(c = "c3")]'; try '#[region(n = "n4")]'; try '#[region(c = "c1")]'; try '#[region(c = "c0", contract = "f32-tree-v2")]'` Enter
Sleep 5s
Type "invar run $P --engine reference --max-c c0 --format json | jq -c '.diagnostics[0] | {code, message}'   # caps only narrow" Enter
Sleep 4s
