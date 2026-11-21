//! Item prefixes the parser looks ahead at: a foreign function, and the closed
//! attribute registry with an optional `pub` (`docs/numeric.md`).
//!
//! `REGISTRY` is the one place that knows every attribute name, the positions
//! it may take, its keys and each key's value grammar. Unknown names, keys,
//! positions and values all refuse here, with one diagnostic family; adding an
//! attribute is one table entry.

use super::*;
use crate::axes::{self, Axes};
use crate::numeric_contract::FoldTier;

/// Where an attribute may appear.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Position {
    Function,
}

impl Position {
    const fn noun(self) -> &'static str {
        match self {
            Self::Function => "a function",
        }
    }
}

/// One key: its value is a string literal naming a member of a closed set.
pub(super) struct Key {
    pub(super) name: &'static str,
    /// What one value names, as in "expected a fold tier string".
    what: &'static str,
    /// The longer name used when a value is unknown.
    unknown: &'static str,
    /// The code of an unknown value.
    code: &'static str,
    pub(super) values: &'static [&'static str],
}

/// One registered attribute.
pub(super) struct Attribute {
    pub(super) name: &'static str,
    positions: &'static [Position],
    /// The code of an unknown or repeated key.
    unknown_key: &'static str,
    pub(super) keys: &'static [Key],
}

/// Every attribute Invar accepts.
pub(super) const REGISTRY: &[Attribute] = &[
    Attribute {
        name: "numeric",
        positions: &[Position::Function],
        unknown_key: "UNSUPPORTED_FEATURE",
        keys: &[Key {
            name: "fold",
            what: "fold tier",
            unknown: "numeric fold tier",
            code: "NUMERIC_POLICY",
            values: &["strict", "tree", "exact", "relaxed"],
        }],
    },
    Attribute {
        name: "region",
        positions: &[Position::Function],
        unknown_key: "CONTRACT_UNKNOWN",
        keys: &[
            level_key("c", "C level", axes::C_AXIS.levels),
            Key {
                name: "contract",
                what: "contract name",
                unknown: "contract",
                code: "CONTRACT_UNKNOWN",
                values: &axes::CONTRACT_NAMES,
            },
            level_key("s", "S level", axes::S_AXIS.levels),
            level_key("n", "N level", axes::N_AXIS.levels),
            level_key("t", "T level", axes::T_AXIS.levels),
        ],
    },
];

/// A `#[region]` axis key: its values are the axis's levels.
const fn level_key(name: &'static str, what: &'static str, values: &'static [&'static str]) -> Key {
    Key {
        name,
        what,
        unknown: what,
        code: "CONTRACT_UNKNOWN",
        values,
    }
}

/// One parsed attribute whose keys and values the registry accepted, in source order.
pub(super) struct Parsed {
    pub(super) entry: &'static Attribute,
    /// From the `#` to the closing `]`.
    pub(super) span: Span,
    pub(super) values: Vec<(&'static str, String)>,
}

impl Parsed {
    pub(super) fn value(&self, key: &str) -> Option<&str> {
        self.values
            .iter()
            .find(|(name, _)| *name == key)
            .map(|(_, value)| value.as_str())
    }
}

/// `a`, `a or b`, `a, b or c`.
fn join(items: &[String], last: &str) -> String {
    match items {
        [] => String::new(),
        [only] => only.clone(),
        [init @ .., tail] => format!("{} {last} {tail}", init.join(", ")),
    }
}

fn unknown_attribute() -> String {
    if let [only] = REGISTRY {
        let key = only.keys.first().map_or("", |key| key.name);
        return format!("the only attribute is #[{}({key} = \"...\")]", only.name);
    }
    let names: Vec<_> = REGISTRY
        .iter()
        .map(|entry| format!("#[{}(...)]", entry.name))
        .collect();
    format!("the attributes are {}", join(&names, "and"))
}

fn unknown_key(entry: &Attribute) -> String {
    let keys: Vec<_> = entry.keys.iter().map(|key| key.name.to_owned()).collect();
    match keys.as_slice() {
        [only] => format!("#[{}] takes one key, {only}", entry.name),
        _ => format!("#[{}] takes the keys {}", entry.name, join(&keys, "and")),
    }
}

impl Parser {
    pub(super) fn is_foreign_fn(&self) -> bool {
        matches!(self.current().kind, TokenKind::Keyword("extern"))
            && matches!(self.tokens.get(self.pos + 1).map(|token| &token.kind), Some(TokenKind::Name(name)) if name == "c")
    }

    /// Optional attributes, each at most once, which must precede `fn` or
    /// `pub fn`, then an optional `pub`. Without `#[numeric]` or a fold
    /// contract a function is Strict. An unknown tier is a compile error, never
    /// a silent strict fold, and a declared level the engine cannot honour is
    /// refused, never run at another level.
    pub(super) fn item_prefix(&mut self) -> Result<(FoldTier, Axes, bool), FrontError> {
        let (mut fold, mut axes, mut seen) = (FoldTier::Strict, Axes::default(), Vec::new());
        while let Some(start) = self.take_symbol("#") {
            let parsed = self.attribute(start)?;
            if seen
                .iter()
                .any(|(entry, _): &(&Attribute, Span)| entry.name == parsed.entry.name)
            {
                return Err(FrontError::source(
                    "UNSUPPORTED_FEATURE",
                    format!("#[{}] may appear only once on an item", parsed.entry.name),
                    parsed.span,
                ));
            }
            if let Some(tier) = parsed.value("fold") {
                // The registry admitted the spelling, so this always finds the tier.
                fold = FoldTier::from_name(tier).unwrap_or(FoldTier::Strict);
            }
            if parsed.entry.name == "region" {
                axes = Axes::resolve(&parsed.values)
                    .map_err(|(code, message)| FrontError::source(code, message, parsed.span))?;
            }
            seen.push((parsed.entry, parsed.span));
        }
        let public_fn = self.is_keyword("pub")
            && matches!(&self.tokens[self.pos + 1].kind, TokenKind::Keyword("fn"));
        let position = (self.is_keyword("fn") || public_fn).then_some(Position::Function);
        for (entry, _) in &seen {
            self.check_position(entry, position)?;
        }
        if let [(_, first), (_, second)] = seen.as_slice()
            && axes.declares_c()
        {
            return Err(FrontError::source(
                "CONTRACT_MISMATCH",
                "a function names its numeric contract once: #[numeric(fold = ...)] or #[region(c = ..., contract = ...)]",
                Span {
                    start: first.start,
                    end: second.end,
                },
            ));
        }
        fold = axes.fold().unwrap_or(fold);
        Ok((fold, axes, self.take_keyword("pub").is_some()))
    }

    /// Refuse an attribute whose following item is not one of its positions.
    fn check_position(
        &self,
        entry: &Attribute,
        position: Option<Position>,
    ) -> Result<(), FrontError> {
        if position.is_some_and(|position| entry.positions.contains(&position)) {
            return Ok(());
        }
        let nouns: Vec<_> = entry
            .positions
            .iter()
            .map(|position| position.noun().to_owned())
            .collect();
        Err(FrontError::source(
            "UNSUPPORTED_FEATURE",
            format!("#[{}] applies only to {}", entry.name, join(&nouns, "or")),
            self.current().span,
        ))
    }

    /// The rest of `#[name(key = "value", ...)]` after the `#`, checked against `REGISTRY`.
    fn attribute(&mut self, start: Span) -> Result<Parsed, FrontError> {
        self.expect_symbol("[")?;
        let (name, span) = self.expect_name()?;
        let entry = REGISTRY
            .iter()
            .find(|entry| entry.name == name)
            .ok_or_else(|| FrontError::source("UNSUPPORTED_FEATURE", unknown_attribute(), span))?;
        self.expect_symbol("(")?;
        let mut values: Vec<(&'static str, String)> = Vec::new();
        loop {
            let (name, span) = self.expect_name()?;
            let key = entry
                .keys
                .iter()
                .find(|key| key.name == name)
                .ok_or_else(|| FrontError::source(entry.unknown_key, unknown_key(entry), span))?;
            if values.iter().any(|(seen, _)| *seen == key.name) {
                return Err(FrontError::source(
                    entry.unknown_key,
                    format!("#[{}] takes the key {} at most once", entry.name, key.name),
                    span,
                ));
            }
            self.expect_symbol("=")?;
            values.push((key.name, self.attribute_value(key)?));
            if self.take_symbol(",").is_none() {
                break;
            }
        }
        self.expect_symbol(")")?;
        let end = self.expect_symbol("]")?.end;
        let span = Span {
            start: start.start,
            end,
        };
        Ok(Parsed {
            entry,
            span,
            values,
        })
    }

    fn attribute_value(&mut self, key: &Key) -> Result<String, FrontError> {
        let token = self.bump();
        let TokenKind::StringLit(value) = token.kind else {
            return Err(FrontError::source(
                "PARSE_EXPECTED_TOKEN",
                format!("expected a {} string", key.what),
                token.span,
            ));
        };
        if !key.values.contains(&value.as_str()) {
            let names: Vec<_> = key.values.iter().map(|value| (*value).to_owned()).collect();
            return Err(FrontError::source(
                key.code,
                format!(
                    "unknown {} \"{value}\" (expected {})",
                    key.unknown,
                    join(&names, "or")
                ),
                token.span,
            ));
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The registry's fold spellings are exactly the tiers, in `FoldTier::ALL` order.
    #[test]
    fn numeric_fold_values_are_the_fold_tiers() {
        let names: Vec<_> = FoldTier::ALL.iter().map(|tier| tier.name()).collect();
        assert_eq!(REGISTRY[0].keys[0].values, names.as_slice());
    }

    /// Each refusal keeps the code, message and span it had before the registry.
    #[test]
    fn numeric_attribute_refusals_keep_their_messages() {
        for (attribute, code, message, at) in [
            (
                "#[numeric(fold = \"fast\")]",
                "NUMERIC_POLICY",
                "unknown numeric fold tier \"fast\" (expected strict, tree, exact or relaxed)",
                "\"fast\"",
            ),
            (
                "#[numeric(fold = tree)]",
                "PARSE_EXPECTED_TOKEN",
                "expected a fold tier string",
                "tree",
            ),
            (
                "#[numeric(order = \"tree\")]",
                "UNSUPPORTED_FEATURE",
                "#[numeric] takes one key, fold",
                "order",
            ),
            (
                "#[inline]",
                "UNSUPPORTED_FEATURE",
                "the attributes are #[numeric(...)] and #[region(...)]",
                "inline",
            ),
            (
                "#[numeric(fold = \"tree\")] struct",
                "UNSUPPORTED_FEATURE",
                "#[numeric] applies only to a function",
                "struct",
            ),
        ] {
            let source = format!("{attribute} S {{ x: f32 }}");
            let error = parse(source.as_bytes()).err().unwrap();
            let span = error.span.unwrap();
            assert_eq!(
                (error.code, error.message.as_str()),
                (code, message),
                "{attribute}"
            );
            assert_eq!(&source[span.start..span.end], at, "{attribute}");
        }
    }
}
