# nix develop gives the pinned toolchain, nextest 0.9.146 and just, then just test-quick runs
# (vhs demo/nix-dev-shell.tape from the repository root, with Nix and flakes enabled).
Output demo/nix-dev-shell.gif
Set Shell bash
Set FontSize 15
Set Width 1400
Set Height 560
Set TypingSpeed 20ms

Hide
Type `export PROMPT_COMMAND="PS1='$ '" && clear` Enter
Show

Type "nix develop" Enter
Sleep 6s
Type "rustc --version; cargo nextest --version | head -1; just --version; c++ --version | head -1" Enter
Sleep 3s
Type "just test-quick 2>&1 | tail -3" Enter
Sleep 22s
Type "exit" Enter
Sleep 1s
