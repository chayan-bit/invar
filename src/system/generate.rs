use super::*;

pub fn generate_mjcf(ir: &SystemIr) -> Result<BTreeMap<String, Vec<u8>>, FrontError> {
    generate_mjcf_target(ir, false)
}

/// ROS-only naming projection; canonical MJCF and checked vendor assets are unchanged.
pub fn generate_ros2_mjcf(ir: &SystemIr) -> Result<BTreeMap<String, Vec<u8>>, FrontError> {
    generate_mjcf_target(ir, true)
}

pub(super) fn generate_mjcf_target(
    ir: &SystemIr,
    ros_names: bool,
) -> Result<BTreeMap<String, Vec<u8>>, FrontError> {
    let span = Span { start: 0, end: 0 };
    let document = ir
        .checked_document()
        .map_err(|message| error("S1600", message, span))?;
    substep_admission::ensure_export_supported(document)
        .map_err(|message| error("S1504", message, span))?;
    if !matches!(
        document["schema"].as_str(),
        Some(
            "invar-system-v5"
                | "invar-system-v6"
                | "invar-system-v7"
                | "invar-system-v8"
                | "invar-system-v9"
                | "invar-system-v10"
                | "invar-system-v11"
                | "invar-system-v12"
        )
    ) {
        return Err(error(
            "S1600",
            "MJCF generation needs a checked v5 robot model",
            span,
        ));
    }
    let product_id = document["system"]["product"]
        .as_str()
        .ok_or_else(|| error("S1601", "system product is missing", span))?;
    let product = document["products"]
        .as_array()
        .and_then(|products| products.iter().find(|product| product["id"] == product_id))
        .ok_or_else(|| error("S1601", "selected product is missing", span))?;
    let robot_id = product["robot"]
        .as_str()
        .ok_or_else(|| error("S1601", "product robot is missing", span))?;
    let robot = document["robots"]
        .as_array()
        .and_then(|robots| robots.iter().find(|robot| robot["id"] == robot_id))
        .ok_or_else(|| error("S1601", "selected robot is missing", span))?;
    let mut tree = if robot["source_tree"].is_null() {
        xml::synthesize_robot(robot).map_err(|issue| error(issue.code, issue.message, span))?
    } else {
        let mut tree = xml::Element::from_value(&robot["source_tree"])
            .ok_or_else(|| error("S1607", "checked MJCF tree is malformed", span))?;
        xml::overlay_robot(&mut tree, robot)
            .map_err(|issue| error(issue.code, issue.message, span))?;
        tree
    };
    xml::apply_tool(&mut tree, robot).map_err(|issue| error(issue.code, issue.message, span))?;
    let cells = document["cells"]
        .as_array()
        .ok_or_else(|| error("S1607", "checked cell list is missing", span))?;
    xml::apply_cells(
        &mut tree,
        cells,
        robot["links"]
            .as_array()
            .and_then(|links| links.iter().find(|link| link["parent"] == "world"))
            .and_then(|link| link["name"].as_str())
            .ok_or_else(|| error("S1601", "robot root link is missing", span))?,
        robot_id,
    )
    .map_err(|issue| error(issue.code, issue.message, span))?;
    if ros_names {
        ros2_tendon::project(&mut tree).map_err(|message| error("S1504", message, span))?;
    }
    let mut files = ir.assets.get(robot_id).cloned().unwrap_or_default();
    files.insert("scene.xml".to_owned(), tree.render().into_bytes());
    Ok(files)
}

pub fn generate_urdf(ir: &SystemIr) -> Result<BTreeMap<String, Vec<u8>>, FrontError> {
    let span = Span { start: 0, end: 0 };
    let document = ir
        .checked_document()
        .map_err(|message| error("S1600", message, span))?;
    if !matches!(
        document["schema"].as_str(),
        Some(
            "invar-system-v5"
                | "invar-system-v6"
                | "invar-system-v7"
                | "invar-system-v8"
                | "invar-system-v9"
                | "invar-system-v10"
                | "invar-system-v11"
                | "invar-system-v12"
        )
    ) {
        return Err(error(
            "S1600",
            "URDF generation needs a checked v5 robot model",
            span,
        ));
    }
    let product_id = document["system"]["product"]
        .as_str()
        .ok_or_else(|| error("S1601", "system product is missing", span))?;
    let product = document["products"]
        .as_array()
        .and_then(|products| products.iter().find(|product| product["id"] == product_id))
        .ok_or_else(|| error("S1601", "selected product is missing", span))?;
    let robot_id = product["robot"]
        .as_str()
        .ok_or_else(|| error("S1601", "product robot is missing", span))?;
    let robot = document["robots"]
        .as_array()
        .and_then(|robots| robots.iter().find(|robot| robot["id"] == robot_id))
        .ok_or_else(|| error("S1601", "selected robot is missing", span))?;
    if document["cells"].as_array().is_some_and(|cells| {
        cells.iter().any(|cell| {
            cell["objects"]
                .as_array()
                .is_some_and(|objects| objects.iter().any(|object| !object["dynamic"].is_null()))
        })
    }) {
        return Err(error(
            "S1607",
            "URDF cannot preserve dynamic cell attachment",
            span,
        ));
    }
    let (source, warnings) =
        urdf::from_robot(robot).map_err(|issue| error(issue.code, issue.message, span))?;
    let mut files = generate_mjcf(ir)?;
    files.insert("robot.urdf".to_owned(), source.into_bytes());
    if !warnings.is_empty() {
        files.insert(
            "urdf-warnings.txt".to_owned(),
            format!("{}\n", warnings.join("\n")).into_bytes(),
        );
    }
    Ok(files)
}
