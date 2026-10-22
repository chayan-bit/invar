//! Item prefixes the parser looks ahead at: a foreign function, and the
//! `#[numeric(fold = "tier")]` attribute with an optional `pub`
//! (`docs/numeric-v2.md`).

use super::*;
use crate::numeric_contract::FoldTier;

impl Parser {
    pub(super) fn is_foreign_fn(&self) -> bool {
        matches!(self.current().kind, TokenKind::Keyword("extern"))
            && matches!(self.tokens.get(self.pos + 1).map(|token| &token.kind), Some(TokenKind::Name(name)) if name == "c")
    }

    /// An optional `#[numeric(fold = "tier")]`, which must precede `fn` or
    /// `pub fn`, then an optional `pub`. Without the attribute a function is
    /// Strict. An unknown tier is a compile error, never a silent strict fold.
    pub(super) fn item_prefix(&mut self) -> Result<(FoldTier, bool), FrontError> {
        let mut fold = FoldTier::Strict;
        if self.take_symbol("#").is_some() {
            fold = self.numeric_attribute()?;
            let public_fn = self.is_keyword("pub")
                && matches!(&self.tokens[self.pos + 1].kind, TokenKind::Keyword("fn"));
            if !(self.is_keyword("fn") || public_fn) {
                return Err(FrontError::source(
                    "UNSUPPORTED_FEATURE",
                    "#[numeric] applies only to a function",
                    self.current().span,
                ));
            }
        }
        Ok((fold, self.take_keyword("pub").is_some()))
    }

    /// The rest of `#[numeric(fold = "tier")]` after the `#`.
    fn numeric_attribute(&mut self) -> Result<FoldTier, FrontError> {
        self.expect_symbol("[")?;
        let (attribute, span) = self.expect_name()?;
        if attribute != "numeric" {
            return Err(FrontError::source(
                "UNSUPPORTED_FEATURE",
                "the only attribute is #[numeric(fold = \"...\")]",
                span,
            ));
        }
        self.expect_symbol("(")?;
        let (key, span) = self.expect_name()?;
        if key != "fold" {
            return Err(FrontError::source(
                "UNSUPPORTED_FEATURE",
                "#[numeric] takes one key, fold",
                span,
            ));
        }
        self.expect_symbol("=")?;
        let token = self.bump();
        let TokenKind::StringLit(tier) = token.kind else {
            return Err(FrontError::source(
                "PARSE_EXPECTED_TOKEN",
                "expected a fold tier string",
                token.span,
            ));
        };
        let fold = FoldTier::from_name(&tier).ok_or_else(|| {
            FrontError::source(
                "NUMERIC_POLICY",
                format!(
                    "unknown numeric fold tier \"{tier}\" (expected strict, tree, exact or relaxed)"
                ),
                token.span,
            )
        })?;
        self.expect_symbol(")")?;
        self.expect_symbol("]")?;
        Ok(fold)
    }
}
