//! Physical types: scalar, unit and frame tags with their source parsing.

use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum FrameTag {
    Vector(String),
    Point(String),
    Transform(String, String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PhysicalType {
    pub(super) scalar: Type,
    pub(super) unit: Option<Unit>,
    pub(super) frame: Option<FrameTag>,
}

impl PhysicalType {
    pub(super) fn plain(scalar: Type) -> Self {
        Self {
            scalar,
            unit: None,
            frame: None,
        }
    }

    pub(super) fn from_source(
        ty: &Type,
        span: Span,
        frames: &HashSet<String>,
    ) -> Result<Self, FrontError> {
        let Type::Named(name) = ty else {
            return Ok(Self::plain(ty.clone()));
        };
        let frame_type = if let Some(parts) = name
            .strip_prefix("Vector3<")
            .and_then(|name| name.strip_suffix('>'))
            .and_then(split_type_args)
        {
            if parts.len() != 3 {
                return Err(mismatch("Vector3 needs scalar, unit and frame", span));
            }
            let scalar = coordinate_scalar(parts[0], span)?;
            let unit = Unit::parse(parts[1])
                .ok_or_else(|| FrontError::source("UNIT_UNKNOWN", "unknown vector unit", span))?;
            Some(Self {
                scalar: array_type(&scalar),
                unit: Some(unit),
                frame: Some(FrameTag::Vector(parts[2].to_owned())),
            })
        } else if let Some(parts) = name
            .strip_prefix("Point<")
            .and_then(|name| name.strip_suffix('>'))
            .and_then(split_type_args)
        {
            if parts.len() != 2 {
                return Err(mismatch("Point needs scalar and frame", span));
            }
            let scalar = coordinate_scalar(parts[0], span)?;
            Some(Self {
                scalar: array_type(&scalar),
                unit: Unit::builtin("um"),
                frame: Some(FrameTag::Point(parts[1].to_owned())),
            })
        } else if let Some(parts) = name
            .strip_prefix("Transform<")
            .and_then(|name| name.strip_suffix('>'))
            .and_then(split_type_args)
        {
            if parts.len() != 2 {
                return Err(mismatch("Transform needs source and target frames", span));
            }
            Some(Self {
                scalar: Type::ArrayI32,
                unit: None,
                frame: Some(FrameTag::Transform(
                    parts[0].to_owned(),
                    parts[1].to_owned(),
                )),
            })
        } else {
            None
        };
        if let Some(frame_type) = frame_type {
            let names: Vec<&str> = match &frame_type.frame {
                Some(FrameTag::Vector(name) | FrameTag::Point(name)) => vec![name],
                Some(FrameTag::Transform(from, to)) => vec![from, to],
                None => Vec::new(),
            };
            if names.iter().any(|name| !frames.contains(*name)) {
                return Err(FrontError::source(
                    "FRAME_UNKNOWN",
                    "coordinate frame is not declared",
                    span,
                ));
            }
            return Ok(frame_type);
        }
        let Some(inside) = name
            .strip_prefix("Quantity<")
            .and_then(|name| name.strip_suffix('>'))
        else {
            return Ok(Self::plain(ty.clone()));
        };
        let (scalar, unit) = inside
            .split_once(',')
            .ok_or_else(|| mismatch("invalid Quantity type", span))?;
        let scalar = match scalar {
            "i32" => Type::I32,
            "u32" => Type::U32,
            "f32" => Type::F32,
            _ => return Err(mismatch("Quantity scalar must be i32, u32 or f32", span)),
        };
        let unit = Unit::parse(unit).ok_or_else(|| {
            FrontError::source(
                "UNIT_UNKNOWN",
                format!("unknown or overflowing unit '{unit}'"),
                span,
            )
        })?;
        Ok(Self {
            scalar,
            unit: Some(unit),
            frame: None,
        })
    }

    pub(super) fn require(self, expected: &Self, span: Span) -> Result<(), FrontError> {
        if self == *expected {
            return Ok(());
        }
        if self.frame != expected.frame {
            return Err(FrontError::source(
                "FRAME_MISMATCH",
                "coordinate frames or geometric value kinds do not match",
                span,
            ));
        }
        if self.unit.is_some() || expected.unit.is_some() {
            return Err(FrontError::source(
                "UNIT_MISMATCH",
                "quantity unit or scalar does not match; use an explicit conversion",
                span,
            ));
        }
        Ok(()) // The ordinary checker gives its existing scalar/type diagnostic.
    }
}

fn split_type_args(value: &str) -> Option<Vec<&str>> {
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut parts = Vec::new();
    for (index, byte) in value.bytes().enumerate() {
        match byte {
            b'<' => depth = depth.checked_add(1)?,
            b'>' => depth = depth.checked_sub(1)?,
            b',' if depth == 0 => {
                parts.push(&value[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return None;
    }
    parts.push(&value[start..]);
    Some(parts)
}

fn coordinate_scalar(name: &str, span: Span) -> Result<Type, FrontError> {
    match name {
        "i32" => Ok(Type::I32),
        "u32" => Ok(Type::U32),
        _ => Err(mismatch("coordinate scalar must be i32 or u32", span)),
    }
}

pub(super) fn array_type(scalar: &Type) -> Type {
    if *scalar == Type::U32 {
        Type::ArrayU32
    } else {
        Type::ArrayI32
    }
}
