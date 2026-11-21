# #257: doc comments (vhs demo/issue-257-docs.tape, invar on PATH).
Output demo/issue-257-docs.gif
Set Shell bash
Set FontSize 15
Set Width 1200
Set Height 820
Set TypingSpeed 20ms

Hide
Type "cd demo && clear" Enter
Show

Type "cat issue-257-docs.inv" Enter
Sleep 2s
Type "invar check issue-257-docs.inv && invar run issue-257-docs.inv --engine reference" Enter
Sleep 3s
Type "# A doc comment that documents no item is refused:" Enter
Type "sed 's|    var sum|    /// not an item\n    var sum|' issue-257-docs.inv > /tmp/bad.inv && invar check /tmp/bad.inv" Enter
Sleep 4s
