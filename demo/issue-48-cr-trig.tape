# #48: f32 sin, cos and sqrt run invar-math-cr-v1 on every engine (vhs demo/issue-48-cr-trig.tape, invar on PATH).
Output demo/issue-48-cr-trig.gif
Set Shell bash
Set FontSize 15
Set Width 1400
Set Height 760
Set TypingSpeed 20ms

Hide
Type "cd demo && P=issue-48-cr-trig.inv && export INVAR_METAL_RUNTIME_SOURCE=1 && clear" Enter
Show

Type "cat $P" Enter
Sleep 2s
Type "# Every engine returns 0: each sin, cos and sqrt equals the correctly rounded word." Enter
Type `for e in reference auto metal; do printf '%-14s' $e; invar run $P --engine $e --format json | jq -c '{value: .result.value, contracts: .numeric_contracts}'; done` Enter
Sleep 6s
Type `printf '%-14s' cpu-parallel; invar run $P --engine cpu-parallel --workers 4 --format json | jq -c '{value: .result.value, contracts: .numeric_contracts}'` Enter
Sleep 6s
Type "# Metal runs bump; it has no cr-v1 sin/cos yet, so trig stays on the CPU and explain says why:" Enter
Type `invar explain $P --engine metal --format json | jq -c '[.device_execution.choices[] | {function, backend, reason}]'` Enter
Sleep 5s
