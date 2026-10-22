use super::*;

#[derive(Clone)]
pub(super) struct Port {
    pub(super) direction: String,
    pub(super) ty: Type,
    pub(super) optional: bool,
    pub(super) span: Span,
}

pub(super) struct Component<'a> {
    pub(super) item: &'a SystemItem,
    pub(super) external: bool,
    pub(super) ports: BTreeMap<String, Port>,
    pub(super) json: Value,
}

pub(super) fn foreign_config_layout(name: &str) -> Option<Value> {
    match name {
        "inv_planner_config_v1" => Some(crate::robotics::planner::config_layout()),
        "inv_fk_config_v1" => Some(json!({
            "layout": name,
            "size": 16,
            "align": 8,
            "fields": [
                {"name":"model_xml_path","type":"borrowed_utf8","offset":0,"borrow":"configure-call"},
                {"name":"model_xml_path_len","type":"u32","offset":8},
                {"name":"model_kind","type":"u32","offset":12}
            ]
        })),
        "inv_planar2_config_v1" => Some(json!({
            "layout": name,
            "size": 16,
            "align": 8,
            "fields": [
                {"name":"proximal_m","type":"f64","offset":0},
                {"name":"distal_m","type":"f64","offset":8}
            ]
        })),
        _ => None,
    }
}

pub(super) fn foreign_port_layout(ty: &Type, layout: &str) -> bool {
    let name = type_name(ty);
    if crate::robotics::planner::port_layout(&name).as_deref() == Some(layout) {
        return true;
    }
    matches!(
        (name.as_str(), layout),
        ("std::robotics::JointPositions<f64,2>", "inv_joint2_f64")
            | ("std::robotics::JointPositions<f64,6>", "inv_joint6_f64")
            | ("std::robotics::JointPositions<f64,7>", "inv_joint7_f64")
    ) || (name.starts_with("std::robotics::Pose<f64,") && layout == "inv_pose_f64")
}

pub(super) fn manifest_relative_file(path: &str) -> bool {
    let path = std::path::Path::new(path);
    !path.as_os_str().is_empty()
        && path.as_os_str().len() <= 4096
        && path
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_)))
}

pub(super) fn manifest_dependency_path(path: &str) -> bool {
    let path = std::path::Path::new(path);
    !path.as_os_str().is_empty()
        && path.as_os_str().len() <= 4096
        && path.components().all(|part| {
            matches!(
                part,
                std::path::Component::Normal(_) | std::path::Component::ParentDir
            )
        })
}

pub(super) fn c_identifier(name: &str) -> bool {
    name.len() <= 64
        && name
            .bytes()
            .next()
            .is_some_and(|first| first.is_ascii_alphabetic() || first == b'_')
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

pub(super) fn checked_foreign_property(
    value: &str,
    ports: &BTreeMap<String, Port>,
    span: Span,
) -> Result<(), FrontError> {
    if let Some((port, raw_tolerance)) = value.split_once(".rotation is unit norm within ") {
        let Some(output) = ports.get(port) else {
            return Err(error(
                "S1400",
                "property names an unknown output port",
                span,
            ));
        };
        let tolerance = decimal(raw_tolerance, span)
            .map_err(|_| error("S1400", "invalid property tolerance", span))?;
        if output.direction == "out"
            && type_name(&output.ty).starts_with("std::robotics::Pose<f64,")
            && !tolerance.starts_with('-')
        {
            return Ok(());
        }
    }
    if let Some(port) = value.strip_suffix(" is finite")
        && ports
            .get(port)
            .is_some_and(|output| output.direction == "out")
    {
        return Ok(());
    }
    Err(error(
        "S1400",
        "unsupported or invalid foreign property",
        span,
    ))
}
