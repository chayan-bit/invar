# Invar for VS Code

Syntax highlighting for Invar (`.inv`) as a TextMate grammar packaged in a minimal VS Code extension.
It is not published; use it from this directory.

## Try it

```sh
code --extensionDevelopmentPath=editors/vscode examples/sum.inv
```

To install it for every window, link the folder into your extensions directory (`~/.vscode/extensions/invar-language`) or build a package with `npx @vscode/vsce package` from this directory.

## What it highlights

- Keywords, split by role: control flow, declarations, ownership modifiers (`read`, `mut`, `take`, `copy`, `move`), word operators, `true` and `false`, and the primitive types the lexer treats as keywords.
- Reserved words (`async`, `await`, `scope`, `break`, `continue`, `total`, `effects`, `requires`, `ensures`) with the `invalid.illegal` scope, so a theme shows them as unavailable rather than as keywords.
- Integer and decimal literals with their suffixes, text and `b"..."` byte strings with escapes, `//` and nested `/* */` comments, `#[...]` attributes, function declarations and calls, the correctly rounded math and conversion intrinsics, and the built-in type names.

## Keeping it in step with the language

`syntaxes/invar.tmLanguage.json` is written by hand.
`tests/editor_grammar.rs` (part of the normal test suite) fails when its keyword or reserved-word lists differ from the lexer's tables in `src/front/lexer.rs`, when a lexer keyword is not a terminal of `spec/grammar.ebnf`, or when an intrinsic or type name it lists is not a word of that spec grammar.
When the lexer gains or loses a word, edit the matching `\b(a|b|c)\b` rule in the grammar; the failure message names the words that differ.
New literal forms or attribute syntax need a matching change to the `numbers`, `strings` or `attributes` rules, which the test does not check.
