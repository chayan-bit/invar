# #259: one closed attribute registry; #[numeric] keeps its accept and refuse behaviour (vhs demo/issue-259-attributes.tape, invar on PATH).
Output demo/issue-259-attributes.gif
Set Shell bash
Set FontSize 15
Set Width 1400
Set Height 700
Set TypingSpeed 15ms

Hide
Type `export PS1='$ ' && cd demo && P=eng-03-tier-folds.inv && T=$(mktemp -d)` Enter
Type `try() { { printf '%s\n' "$1"; sed 1d $P; } > $T/a.inv; invar check $T/a.inv --format json | jq -c '.diagnostics[0] | {code, message}'; }` Enter
Type "clear" Enter
Show

Type "head -2 $P && invar run $P --engine reference --format json | jq -c '{status, value: .result.value}'" Enter
Sleep 3s
Type "# The registry refuses an unknown name, an unknown key, an unknown tier, a non-string value and a wrong position:" Enter
Type `try '#[inline]'; try '#[numeric(order = "tree")]'; try '#[numeric(fold = "fast")]'; try '#[numeric(fold = tree)]'` Enter
Sleep 4s
Type `printf '#[numeric(fold = "tree")]\nstruct S { x: f32 }\nfn main() -> i32 { return 0; }\n' > $T/b.inv && invar check $T/b.inv --format json | jq -c '.diagnostics[0] | {code, message}'` Enter
Sleep 4s
