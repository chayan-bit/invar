# #263: the TextMate grammar highlights keywords, literals and comments, shows reserved words as
# illegal, and the drift test passes (vhs demo/editor-grammar.tape from the repository root;
# needs `npm install shiki` in a scratch directory exported as SHIKI_DIR/node_modules).
Output demo/editor-grammar.gif
Set Shell bash
Set FontSize 14
Set Width 1400
Set Height 820
Set TypingSpeed 20ms

Hide
Type "export PS1='$ ' NODE_PATH=$SHIKI_DIR/node_modules && clear" Enter
Show

Type "node demo/render-grammar.mjs demo/editor-grammar.inv" Enter
Sleep 7s
Type "cargo nextest run --test editor_grammar --cargo-profile ci-debug --profile ci 2>&1 | tail -7" Enter
Sleep 12s
