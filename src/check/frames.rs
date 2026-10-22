//! Nominal frame-tree validation; frame tags have no runtime representation.

use super::*;
use crate::front::FrameDecl;

const MAX_FRAMES: usize = 128;

pub(super) fn validate(frames: &[FrameDecl]) -> Result<(), FrontError> {
    if frames.len() > MAX_FRAMES {
        return Err(FrontError::source(
            "RESOURCE_LIMIT",
            "frame declaration limit exceeds 128",
            frames[MAX_FRAMES].span,
        ));
    }
    if frames.is_empty() {
        return Ok(());
    }

    let mut declared = HashMap::new();
    for frame in frames {
        if declared.insert(frame.name.as_str(), frame).is_some() {
            return Err(FrontError::source(
                "FRAME_DUPLICATE",
                format!("frame '{}' is declared twice", frame.name),
                frame.span,
            ));
        }
    }
    for frame in frames {
        if let Some(parent) = &frame.parent
            && !declared.contains_key(parent.as_str())
        {
            return Err(FrontError::source(
                "FRAME_PARENT_UNKNOWN",
                format!("frame '{}' has undeclared parent '{parent}'", frame.name),
                frame.span,
            ));
        }
    }

    // ponytail: at most 128 frames, so following each chain is simpler than a graph framework.
    for frame in frames {
        let mut path = HashSet::new();
        let mut current = frame;
        loop {
            if !path.insert(current.name.as_str()) {
                return Err(FrontError::source(
                    "FRAME_CYCLE",
                    format!("frame '{}' has a parent cycle", frame.name),
                    current.span,
                ));
            }
            let Some(parent) = &current.parent else { break };
            current = declared[parent.as_str()];
        }
    }

    let roots = frames
        .iter()
        .filter(|frame| frame.parent.is_none())
        .collect::<Vec<_>>();
    if roots.len() != 1 {
        return Err(FrontError::source(
            "FRAME_ROOTS",
            "frame declarations require exactly one root",
            roots.get(1).copied().unwrap_or(&frames[0]).span,
        ));
    }
    Ok(())
}
