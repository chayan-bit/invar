# verify-invar skill sync: doctor, the parser reject drive and the --fix dry run match the skill text
# (vhs demo/verify-invar-skill.tape from the repository root, with a built target/debug/invar).
Output demo/verify-invar-skill.gif
Set Shell bash
Set FontSize 15
Set Width 1400
Set Height 760
Set TypingSpeed 20ms

Hide
Type "export PS1='$ ' S=.cursor/skills/verify-invar/scripts/invar-verify INVAR_VERIFY_ROOT=$(mktemp -d) && clear" Enter
Show

Type "# doctor names the C++20 compiler probe that the skill now documents:" Enter
Type "$S doctor --format json | jq -c '{ready, toolchain_ok, cxx: .host.cxx.compiler}'" Enter
Sleep 4s
Type "# the reject drive: a '->' where a parameter name belongs is PARSE_EXPECTED_TOKEN, 'expected identifier':" Enter
Type "$S drive parser --format json | jq -c '.checks[] | {id, status, summary}'" Enter
Sleep 6s
Type "# --fix without --write is the dry run: it prints a diff and writes nothing:" Enter
Type "invar check examples/views/overlap.inv --fix; echo exit=$?; git status --short examples | wc -l" Enter
Sleep 4s
