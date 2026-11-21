# #334: an NVRTC helper killed by a signal is a host fault, never a typed compile refusal (vhs demo/issue-334-killed-compiler.tape, cargo on PATH).
Output demo/issue-334-killed-compiler.gif
Set Shell bash
Set FontSize 13
Set Width 1400
Set Height 700
Set TypingSpeed 15ms

Hide
Type `export PS1='$ ' && clear` Enter
Show

Type "# A helper that dies by a signal its parent never sent, as under the kernel OOM killer:" Enter
Type "cat crates/invar-cuda/tests/fixtures/compiler-child/killed.sh" Enter
Sleep 2s
Type `cargo nextest run -p invar-cuda --features runtime compiler_child 2>&1 | grep -E 'PASS|FAIL|Summary'` Enter
Wait+Line@120s /^\$$/
Sleep 2s
Type "# Before the fix: Compile, compiler exit signal 9, which the census recorded as refused compile_error." Enter
Sleep 4s
