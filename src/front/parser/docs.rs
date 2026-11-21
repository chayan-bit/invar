// Doc comments (#257): trivia the lexer keeps beside the tokens, attached
// here to the item each documents. A doc comment that documents nothing is
// refused, so none is silently dropped.

impl Parser {
    /// The span from `start` to the end of the last token taken.
    fn since(&self, start: usize) -> Span {
        let end = self.tokens[self.pos.saturating_sub(1)].span.end;
        Span {
            start,
            end: end.max(start),
        }
    }

    /// Attach each `///` to the item whose first token follows it, and each
    /// `//!` to the file when no token precedes it, or to the module whose
    /// opening `{` does.
    fn attach_docs(
        &mut self,
        mut items: Vec<Span>,
        bodies: &[(usize, Span)],
    ) -> Result<Vec<DocComment>, FrontError> {
        items.sort_by_key(|item| item.start);
        // The lexer always appends EOF at the end of the source.
        let file = Span {
            start: 0,
            end: self.tokens.last().map_or(0, |eof| eof.span.end),
        };
        let mut docs = std::mem::take(&mut self.docs);
        for doc in &mut docs {
            let next = self
                .tokens
                .partition_point(|token| token.span.start < doc.span.end);
            let item = if doc.inner {
                match next.checked_sub(1) {
                    None => Some(file),
                    Some(previous) => bodies
                        .iter()
                        .find(|(open, _)| *open == self.tokens[previous].span.end)
                        .map(|(_, module)| *module),
                }
            } else {
                let start = self.tokens[next].span.start;
                items
                    .binary_search_by_key(&start, |item| item.start)
                    .ok()
                    .map(|at| items[at])
            };
            doc.item = item.ok_or_else(|| {
                FrontError::source(
                    "PARSE_DOC_COMMENT_MISPLACED",
                    if doc.inner {
                        "`//!` must come before the first item of its file or module"
                    } else {
                        "`///` must come directly before an item"
                    },
                    doc.span,
                )
            })?;
        }
        Ok(docs)
    }
}
