# #210: every contract refusal has its own code, a note and a validated fix where one exists (vhs demo/issue-210-contract-codes.tape, invar on PATH).
Output demo/issue-210-contract-codes.gif
Set Shell bash
Set FontSize 14
Set Width 1400
Set Height 820
Set TypingSpeed 10ms

Hide
Type `export PS1='$ ' && cd demo && P=issue-198-region.inv && T=$(mktemp -d)` Enter
Type `try() { sed "1s/.*/$1/" $P > $T/a.inv; invar check $T/a.inv --format json | jq -c '.diagnostics[0] | {code, note: .notes[0], fix: [.suggested_fixes[0].edits[]?.replacement]}'; }` Enter
Type "clear" Enter
Show

Type `jq -r '.codes[] | .code + ": " + .meaning' ../schema/diagnostic-codes-contract-v1.json` Enter
Sleep 3s
Type `try '#[region(s = "s2", n = "n4")]'; try '#[region(c = "c0", contract = "f32-exact-v2")]'; try '#[region(c = "C1", contract = "f32-tree-v2")]'` Enter
Sleep 5s
Type "invar run $P --engine reference --max-n n0 --format json | jq -c '.diagnostics[0] | {code, note: .notes[0]}'" Enter
Sleep 3s
Type `sed '1s/.*/#[region(contract = "f32-exact-v2", n = "n4")]/' $P > $T/b.inv && invar check $T/b.inv --fix --write > /dev/null; head -1 $T/b.inv && invar run $T/b.inv --engine reference --format json | jq -c '{value: .result.value}'` Enter
Sleep 4s
