//! Fixes and notes for the contract codes of `#[region(...)]`
//! (`schema/diagnostic-codes-contract-v1.json`). Each edit rewrites one
//! quoted value of the attribute the diagnostic spans; the engine rechecks it.

use super::{Candidate, Edit, Generated, Recheck};
use crate::axes::{AXES, CONTRACT_NAMES, contract_level};
use crate::front::Span;

/// The `key = "value"` pairs of the attribute at `span`, with each value's
/// span (its quotes excluded), or nothing when the span holds no attribute.
fn pairs(source: &[u8], span: Span) -> Vec<(&str, &str, Span)> {
    let Some(text) = source
        .get(span.start..span.end)
        .and_then(|bytes| std::str::from_utf8(bytes).ok())
    else {
        return Vec::new();
    };
    let (Some(open), Some(close)) = (text.find('('), text.rfind(')')) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut at = open + 1;
    for part in text[open + 1..close].split(',') {
        if let Some((key, value)) = part.split_once('=') {
            let quote = at + key.len() + 1 + value.find('"').unwrap_or(value.len());
            let inner = value.trim().trim_matches('"');
            if value.trim().starts_with('"') {
                let start = span.start + quote + 1;
                out.push((
                    key.trim(),
                    inner,
                    Span {
                        start,
                        end: start + inner.len(),
                    },
                ));
            }
        }
        at += part.len() + 1;
    }
    out
}

fn replace(description: String, span: Span, value: &str) -> Candidate {
    Candidate {
        description,
        edits: vec![Edit {
            span,
            replacement: value.to_owned(),
        }],
    }
}

/// `CONTRACT_UNAVAILABLE`: declare, on every refused axis at once, the highest
/// level this engine honours.
pub(super) fn unavailable(source: &[u8], span: Span) -> Generated {
    let mut names = Vec::new();
    let edits: Vec<Edit> = pairs(source, span)
        .into_iter()
        .filter_map(|(key, value, at)| {
            let axis = AXES.iter().find(|axis| axis.key == key)?;
            let best = axis.honoured.last()?;
            (!axis.honoured.contains(&value)).then(|| {
                names.push(*best);
                Edit {
                    span: at,
                    replacement: (*best).to_owned(),
                }
            })
        })
        .collect();
    let candidates = (!edits.is_empty()).then(|| Candidate {
        description: format!(
            "Declare {}, the highest level this engine honours on each refused axis",
            names.join(" and ")
        ),
        edits,
    });
    Generated::of(candidates.into_iter().collect(), Recheck::Clean)
}

/// `CONTRACT_MISMATCH`: make the C level the one the named contract implies.
pub(super) fn mismatch(source: &[u8], span: Span) -> Generated {
    let pairs = pairs(source, span);
    let needed = pairs
        .iter()
        .find(|(key, ..)| *key == "contract")
        .and_then(|(_, value, _)| contract_level(value));
    let candidates = pairs
        .iter()
        .filter(|(key, ..)| *key == "c")
        .filter_map(|(_, _, at)| {
            let level = needed?.name();
            Some(replace(
                format!("Declare {level}, the level of the named contract"),
                *at,
                level,
            ))
        })
        .collect();
    Generated::of(candidates, Recheck::Clean)
}

/// `CONTRACT_UNKNOWN` on a value: the same spelling in lower case, when that is a level or contract.
pub(super) fn unknown(source: &[u8], span: Span) -> Generated {
    let text = source
        .get(span.start..span.end)
        .and_then(|bytes| std::str::from_utf8(bytes).ok());
    let Some(inner) = text.and_then(|text| text.strip_prefix('"')?.strip_suffix('"')) else {
        return Generated::none();
    };
    let lower = inner.to_ascii_lowercase();
    let known = AXES
        .iter()
        .any(|axis| axis.levels.contains(&lower.as_str()))
        || CONTRACT_NAMES.contains(&lower.as_str());
    let at = Span {
        start: span.start + 1,
        end: span.end - 1,
    };
    let candidates = (known && lower != inner)
        .then(|| replace(format!("Write {lower} in lower case"), at, &lower));
    Generated::of(candidates.into_iter().collect(), Recheck::Clean)
}

/// The note each contract code carries.
pub(super) fn note(code: &str) -> Option<&'static str> {
    Some(match code {
        "CONTRACT_UNKNOWN" => {
            "#[region] takes the keys c, contract, s, n and t, each with a level or contract listed in docs/spec/invar-v0.2-axes.md section 4"
        }
        "CONTRACT_UNAVAILABLE" => {
            "a level this engine does not honour is refused, never run at another level"
        }
        "CONTRACT_MISMATCH" => {
            "a contract implies its C level, C1 and C2 each name one of their contracts, and a function names its contract once"
        }
        "CONTRACT_CAP" => "a cap only narrows: raise --max-c or --max-n, or declare a lower level",
        _ => return None,
    })
}
