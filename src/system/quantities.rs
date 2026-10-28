use super::*;

pub(super) fn qualified(module: Option<&str>, name: &str) -> String {
    if name.contains("::") {
        name.to_owned()
    } else {
        module.map_or_else(|| name.to_owned(), |module| format!("{module}::{name}"))
    }
}

pub(super) fn decimal(value: &str, span: Span) -> Result<String, FrontError> {
    let negative = value.starts_with('-');
    let raw = value.strip_prefix('-').unwrap_or(value);
    let (whole, fraction) = raw.split_once('.').map_or((raw, ""), |(a, b)| (a, b));
    if value.len() > 64
        || whole.is_empty()
        || !whole.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(error("S1000", "invalid exact decimal", span));
    }
    let whole = whole.trim_start_matches('0');
    let whole = if whole.is_empty() { "0" } else { whole };
    let fraction = fraction.trim_end_matches('0');
    let mut result = String::new();
    if negative && !(whole == "0" && fraction.is_empty()) {
        result.push('-');
    }
    result.push_str(whole);
    if !fraction.is_empty() {
        result.push('.');
        result.push_str(fraction);
    }
    Ok(result)
}

pub(super) fn quantity(value: &SystemQuantity) -> Result<Value, FrontError> {
    Ok(json!({"value": decimal(&value.value, value.span)?, "unit": value.unit}))
}

pub(super) fn positive(value: &SystemQuantity, unit: Option<&str>) -> Result<Value, FrontError> {
    if let Some(unit) = unit
        && value.unit != unit
    {
        return Err(error(
            "S1006",
            format!("expected {unit} quantity"),
            value.span,
        ));
    }
    let result = quantity(value)?;
    if result["value"] == "0"
        || result["value"]
            .as_str()
            .is_some_and(|number| number.starts_with('-'))
    {
        return Err(error("S1000", "quantity must be positive", value.span));
    }
    Ok(result)
}

pub(super) fn duration(value: &SystemQuantity) -> Result<Value, FrontError> {
    if !matches!(value.unit.as_str(), "s" | "ms" | "us" | "ns") {
        return Err(error("S1006", "duration must have a time unit", value.span));
    }
    positive(value, None)
}

pub(super) fn resources(values: &[(String, SystemQuantity)]) -> Result<Value, FrontError> {
    let mut result = BTreeMap::new();
    for (key, value) in values {
        if result.contains_key(key) {
            return Err(error(
                "S1000",
                format!("duplicate resource '{key}'"),
                value.span,
            ));
        }
        if !matches!(value.unit.as_str(), "one" | "MiB" | "GiB" | "Mbit/s") {
            return Err(error("S1006", "unsupported resource unit", value.span));
        }
        result.insert(key.clone(), positive(value, None)?);
    }
    Ok(json!(result))
}
