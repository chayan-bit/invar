# #267: the N axis chapter; every level has an oracle and a fixture the spec test runs (vhs demo/issue-267-axis-n.tape, invar on PATH).
Output demo/issue-267-axis-n.gif
Set Shell bash
Set FontSize 14
Set Width 1400
Set Height 820
Set TypingSpeed 10ms

Hide
Type `export PS1='$ ' && S=docs/spec/invar-v0.2-axis-n.md && T=$(mktemp -d)` Enter
Type `awk -v d=$T '/^.{3}invar/{n++; f=d"/b"n".inv"; next} /^.{3}$/{f=""; next} f{print > f}' $S` Enter
Type `fix() { invar run $T/b$1.inv --engine reference --format json | jq -c '{level: "'$2'", value: .result.value, code: .diagnostics[0].code}'; }` Enter
Type "clear" Enter
Show

Type "grep -A6 '^| Level' $S" Enter
Sleep 4s
Type "fix 1 n0; fix 2 n1; fix 3 n2; fix 4 n3; fix 5 n4   # the five fixtures of the chapter" Enter
Sleep 4s
Type "invar explain $T/b3.inv --format json | jq -c '.regions[0] | {n_level, deterministic, n_justification}'" Enter
Sleep 4s
