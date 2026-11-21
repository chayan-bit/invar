// Render a file with editors/vscode's TextMate grammar as 24-bit ANSI, for the VHS demo.
// Usage: NODE_PATH=<dir with shiki> node demo/render-grammar.mjs FILE   (npm install shiki in a scratch dir first)
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
const require = createRequire(process.env.NODE_PATH + "/");
const { createHighlighter } = require("shiki");
const grammar = JSON.parse(readFileSync("editors/vscode/syntaxes/invar.tmLanguage.json", "utf8"));
const highlighter = await createHighlighter({ themes: ["dark-plus"], langs: [{ ...grammar, name: "invar" }] });
const { tokens } = highlighter.codeToTokens(readFileSync(process.argv[2], "utf8"), { lang: "invar", theme: "dark-plus" });
const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)).join(";");
for (const line of tokens) {
  console.log(line.map((t) => (t.color ? `\x1b[38;2;${rgb(t.color)}m${t.content}\x1b[0m` : t.content)).join(""));
}
