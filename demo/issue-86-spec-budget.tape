# #86: spec chapters, budgets and topic assembly (vhs demo/issue-86-spec-budget.tape, invar on PATH).
Output demo/issue-86-spec-budget.gif
Set Shell bash
Set FontSize 15
Set Width 1300
Set Height 800
Set TypingSpeed 20ms

Hide
Type "export PS1='$ ' && clear" Enter
Show

Type "ls docs/spec/chapters" Enter
Sleep 2s
Type "invar spec --topic ownership --format json | jq -c '{budget, tokens, chapters: [.chapters[].id]}'" Enter
Sleep 2s
Type "invar spec --topic views --budget 6000 | grep '^## \|^### 5'" Enter
Sleep 3s
Type "invar spec --topic engines --budget 6000; echo exit=$?" Enter
Sleep 3s
Type "invar spec --topic nowhere; echo exit=$?" Enter
Sleep 2s
Type "cargo nextest run --locked --test spec_budget --no-capture 2>&1 | grep -E 'bytes/3|whole spec|Summary'" Enter
Sleep 14s
