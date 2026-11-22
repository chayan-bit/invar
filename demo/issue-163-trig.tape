# #163: cr-v1 f32 sin and cos on the devices (vhs demo/issue-163-trig.tape).
# BEFORE and AFTER name invar binaries built from main and from this branch;
# EVIDENCE names a text file with the device test summaries quoted in the PR.
Output demo/issue-163-trig.gif
Set Shell bash
Set FontSize 15
Set Width 1400
Set Height 820
Set TypingSpeed 12ms

Hide
Type `cd demo && P=issue-163-trig.inv && C='{"fuel":100000,"transitions":1000000,"trace_events":4000,"quantum":16,"rounds":64,"capacity":2}' && clear` Enter
Show

Type "cat $P" Enter
Sleep 2s
Type "# main: a device request keeps the function calling sin/cos on the CPU" Enter
Type `$BEFORE explain $P --engine auto --host-class linux-aarch64 --format json | jq -c '.device_execution.choices[] | {function, backend, reason}'` Enter
Sleep 3s
Type "# this branch: nothing is kept on the CPU for sin/cos" Enter
Type `$AFTER explain $P --engine auto --host-class linux-aarch64 --format json | jq -c '.device_execution.choices[] | {function, backend, reason}'` Enter
Sleep 3s
Type "clear; # general net: CUDA and Metal device text on the CPU model, x = 0x5922aa80 (a CORE-MATH hard case)" Enter
Type `for t in cpu-model-cuda cpu-model-metal; do $BEFORE gnet-device $P run $t '[{"bits":1495444096}]' "$C" | jq -c '{backend, refusal: .placement.refusal, result}'; done` Enter
Sleep 3s
Type `for t in cpu-model-cuda cpu-model-metal; do $AFTER gnet-device $P run $t '[{"bits":1495444096}]' "$C" | jq -c '{backend, refusal: .placement.refusal, result}'; done` Enter
Sleep 8s
Type "# real devices: every word equals the CPU cr-v1 word" Enter
Type "cat $EVIDENCE" Enter
Sleep 8s
