# #199: explain-v6 records every axis level, contract, bound and determinism per region (vhs demo/issue-199-explain-v6.tape, invar on PATH).
Output demo/issue-199-explain-v6.gif
Set Shell bash
Set FontSize 14
Set Width 1400
Set Height 820
Set TypingSpeed 10ms

Hide
Type `export PS1='$ ' && cd demo && P=issue-198-region.inv && T=$(mktemp -d)` Enter
Type `axes() { invar explain $1 --format json | jq -c '{schema, levels: .numeric_contracts.levels, region: (.regions[0] | {c_level, n_level, contract_name, bound, deterministic, n_justification})}'; }` Enter
Type "clear" Enter
Show

Type "head -1 $P" Enter
Sleep 1s
Type "axes $P" Enter
Sleep 4s
Type `sed '1s/.*/#[region(contract = "f32-relaxed-v2", n = "n2")]/' $P > $T/c2.inv && axes $T/c2.inv   # C2 names its bound` Enter
Sleep 5s
Type "# An undeclared program: v6 minus the axis fields is the v5 document main printed before" Enter
Type `invar explain ../examples/sum.inv --format json | jq -S '.schema = "invar-explain-v5" | .schema_version = 5 | .regions |= map(del(.c_level, .s_level, .n_level, .t_level, .contract_name, .bound, .deterministic, .n_justification, .s_artifact))' > $T/v5.json` Enter
Type "jq -S . ../tests/numeric_tiers/explain_v5/sum.json | diff - $T/v5.json && echo identical to the v5 golden" Enter
Sleep 4s
