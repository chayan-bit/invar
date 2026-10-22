use super::overlay::{plain_values, quantity_values};
use super::*;

pub(super) fn element(tag: &str, attrs: &[(&str, String)], children: Vec<Element>) -> Element {
    Element {
        tag: tag.to_owned(),
        attrs: attrs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), value.clone()))
            .collect(),
        children,
    }
}

pub(super) fn pose_attrs(value: &Value) -> Result<Vec<(&'static str, String)>, XmlError> {
    Ok(vec![
        (
            "pos",
            quantity_values(&value["xyz"]).ok_or_else(|| fail("S1607", "invalid checked pose"))?,
        ),
        (
            "quat",
            plain_values(&value["quat"])
                .ok_or_else(|| fail("S1607", "invalid checked quaternion"))?,
        ),
    ])
}

pub(super) fn geometry_element(geometry: &Value) -> Result<Element, XmlError> {
    let kind = geometry["kind"]
        .as_str()
        .ok_or_else(|| fail("S1607", "invalid geometry kind"))?;
    if kind == "mesh" {
        return Err(fail(
            "S1607",
            "source-free mesh generation needs a checked asset file set",
        ));
    }
    let size = plain_values(&geometry["dimensions"])
        .ok_or_else(|| fail("S1607", "invalid geometry dimensions"))?;
    let role = geometry["role"]
        .as_str()
        .ok_or_else(|| fail("S1607", "invalid geometry role"))?;
    let mut attrs = vec![("type", kind.to_owned()), ("size", size)];
    if role == "visual" {
        attrs.push(("contype", "0".to_owned()));
        attrs.push(("conaffinity", "0".to_owned()));
    }
    Ok(element("geom", &attrs, Vec::new()))
}

pub(super) fn build_body(
    link: &Value,
    links: &[Value],
    joints: &[Value],
    sensors: &[Value],
) -> Result<Element, XmlError> {
    let name = link["name"]
        .as_str()
        .ok_or_else(|| fail("S1607", "checked link name is missing"))?;
    let mut attrs = vec![("name", name.to_owned())];
    attrs.extend(pose_attrs(&link["origin"])?);
    let mut inertial_attrs = vec![(
        "mass",
        link["mass"]["value"]
            .as_str()
            .ok_or_else(|| fail("S1607", "checked mass is missing"))?
            .to_owned(),
    )];
    if link["inertial_origin"].is_null() {
        inertial_attrs.extend([("pos", "0 0 0".to_owned()), ("quat", "1 0 0 0".to_owned())]);
    } else {
        inertial_attrs.extend(pose_attrs(&link["inertial_origin"])?);
    }
    let inertia = link["inertia"]
        .as_array()
        .ok_or_else(|| fail("S1607", "checked inertia is missing"))?;
    if inertia[3..].iter().all(|value| value == "0") {
        inertial_attrs.push(("diaginertia", plain_values(&json!(&inertia[..3])).unwrap()));
    } else {
        inertial_attrs.push(("fullinertia", plain_values(&link["inertia"]).unwrap()));
    }
    let mut children = vec![element("inertial", &inertial_attrs, Vec::new())];
    for joint in joints
        .iter()
        .filter(|joint| joint["child"] == name && joint["kind"] != "fixed")
    {
        let kind = if joint["kind"] == "prismatic" {
            "slide"
        } else {
            "hinge"
        };
        let mut joint_attrs = vec![
            ("name", joint["name"].as_str().unwrap().to_owned()),
            ("type", kind.to_owned()),
            ("axis", plain_values(&joint["axis"]).unwrap()),
        ];
        if !joint["position"].is_null() {
            joint_attrs.push((
                "range",
                format!(
                    "{} {}",
                    joint["position"]["lower"]["value"].as_str().unwrap(),
                    joint["position"]["upper"]["value"].as_str().unwrap()
                ),
            ));
        }
        children.push(element("joint", &joint_attrs, Vec::new()));
    }
    for geometry in link["geometries"]
        .as_array()
        .ok_or_else(|| fail("S1607", "checked geometry list is missing"))?
    {
        children.push(geometry_element(geometry)?);
    }
    for sensor in sensors
        .iter()
        .filter(|sensor| sensor["source"] == name && sensor["kind"] == "camera")
    {
        let camera_name = sensor["name"].as_str().unwrap();
        let intrinsics = &sensor["intrinsics"];
        let field = |key| {
            intrinsics[key]
                .as_str()
                .ok_or_else(|| fail("S1607", "camera intrinsics field is missing"))
        };
        let width = field("width_px")?;
        let height = field("height_px")?;
        children.push(element(
            "camera",
            &[
                ("name", camera_name.to_owned()),
                ("resolution", format!("{width} {height}")),
                ("sensorsize", format!("{width} {height}")),
                (
                    "focalpixel",
                    format!("{} {}", field("fx_px")?, field("fy_px")?),
                ),
                (
                    "principalpixel",
                    format!(
                        "{} {}",
                        principal_offset(field("cx_px")?, width)?,
                        principal_offset(field("cy_px")?, height)?
                    ),
                ),
            ],
            Vec::new(),
        ));
    }
    for sensor in sensors
        .iter()
        .filter(|sensor| sensor["kind"] == "force-torque")
    {
        let joint = joints
            .iter()
            .find(|joint| joint["name"] == sensor["source"])
            .ok_or_else(|| fail("S1601", "force-torque sensor joint is missing"))?;
        if joint["child"] == name {
            children.push(element(
                "site",
                &[
                    ("name", format!("{}_site", sensor["name"].as_str().unwrap())),
                    ("pos", "0 0 0".to_owned()),
                    ("size", "0.001".to_owned()),
                ],
                Vec::new(),
            ));
        }
    }
    for child in links.iter().filter(|child| child["parent"] == name) {
        children.push(build_body(child, links, joints, sensors)?);
    }
    Ok(element("body", &attrs, children))
}

pub(super) fn principal_offset(value: &str, extent: &str) -> Result<String, XmlError> {
    let fraction = value.split_once('.').map_or("", |(_, fraction)| fraction);
    let scale = fraction.len().max(1);
    let digits = value.replace('.', "");
    let value = BigInt::parse_bytes(digits.as_bytes(), 10)
        .ok_or_else(|| fail("S1607", "invalid camera principal point"))?
        * BigInt::from(10u8).pow((scale - fraction.len()) as u32);
    let extent = extent
        .parse::<u32>()
        .map_err(|_| fail("S1607", "invalid camera resolution"))?;
    let offset: BigInt =
        value - BigInt::from(extent) * 5 * BigInt::from(10u8).pow((scale - 1) as u32);
    let negative = offset < BigInt::from(0);
    let mut digits = offset.to_string();
    if negative {
        digits.remove(0);
    }
    if digits.len() <= scale {
        digits = format!("{}{}", "0".repeat(scale + 1 - digits.len()), digits);
    }
    digits.insert(digits.len() - scale, '.');
    exact_decimal(&format!("{}{digits}", if negative { "-" } else { "" }))
        .ok_or_else(|| fail("S1607", "invalid camera principal point"))
}

pub(crate) fn synthesize_robot(robot: &Value) -> Result<Element, XmlError> {
    let links = robot["links"]
        .as_array()
        .ok_or_else(|| fail("S1607", "checked links are missing"))?;
    let joints = robot["joints"]
        .as_array()
        .ok_or_else(|| fail("S1607", "checked joints are missing"))?;
    let sensors = robot["sensors"]
        .as_array()
        .ok_or_else(|| fail("S1607", "checked sensors are missing"))?;
    let root = links
        .iter()
        .find(|link| link["parent"] == "world")
        .ok_or_else(|| fail("S1601", "robot root link is missing"))?;
    let world = element(
        "worldbody",
        &[],
        vec![build_body(root, links, joints, sensors)?],
    );
    let mut children = vec![
        element("compiler", &[("angle", "radian".to_owned())], Vec::new()),
        world,
    ];
    let mut actuator_items = Vec::new();
    for actuator in robot["actuators"]
        .as_array()
        .ok_or_else(|| fail("S1607", "checked actuators are missing"))?
    {
        let tag = match actuator["mode"].as_str() {
            Some("position") => "position",
            Some("velocity") => "velocity",
            Some("motor-torque") => "motor",
            _ => return Err(fail("S1607", "unknown checked actuator mode")),
        };
        let target_kind = actuator["target_kind"]
            .as_str()
            .ok_or_else(|| fail("S1607", "checked actuator target kind is missing"))?;
        let attrs = [
            ("name", actuator["name"].as_str().unwrap().to_owned()),
            (target_kind, actuator["target"].as_str().unwrap().to_owned()),
            ("gear", actuator["gear"].as_str().unwrap().to_owned()),
            (
                "ctrlrange",
                format!(
                    "{} {}",
                    actuator["control"]["lower"]["value"].as_str().unwrap(),
                    actuator["control"]["upper"]["value"].as_str().unwrap()
                ),
            ),
        ];
        actuator_items.push(element(tag, &attrs, Vec::new()));
    }
    if !actuator_items.is_empty() {
        children.push(element("actuator", &[], actuator_items));
    }
    let mut sensor_items = Vec::new();
    for sensor in sensors {
        let name = sensor["name"]
            .as_str()
            .ok_or_else(|| fail("S1607", "checked sensor name is missing"))?;
        match sensor["kind"].as_str() {
            Some("joint-position" | "joint-velocity") => {
                let kind = if sensor["kind"] == "joint-position" {
                    "jointpos"
                } else {
                    "jointvel"
                };
                sensor_items.push(element(
                    kind,
                    &[
                        ("name", name.to_owned()),
                        ("joint", sensor["source"].as_str().unwrap().to_owned()),
                    ],
                    Vec::new(),
                ));
            }
            Some("force-torque") => {
                let site = format!("{name}_site");
                for (tag, suffix) in [("force", "force"), ("torque", "torque")] {
                    sensor_items.push(element(
                        tag,
                        &[("name", format!("{name}_{suffix}")), ("site", site.clone())],
                        Vec::new(),
                    ));
                }
            }
            Some("camera") => {}
            _ => return Err(fail("S1607", "unknown checked sensor kind")),
        }
    }
    if !sensor_items.is_empty() {
        children.push(element("sensor", &[], sensor_items));
    }
    Ok(element(
        "mujoco",
        &[(
            "model",
            robot["name"].as_str().unwrap_or("robot").to_owned(),
        )],
        children,
    ))
}

pub(crate) fn apply_tool(tree: &mut Element, robot: &Value) -> Result<(), XmlError> {
    if robot["tool"].is_null() {
        return Ok(());
    }
    let link = robot["tool"]["link"]
        .as_str()
        .ok_or_else(|| fail("S1607", "checked tool link is missing"))?;
    fn has_site(node: &Element) -> bool {
        (node.tag == "site" && attr(node, "name") == Some("invar_tool"))
            || node.children.iter().any(has_site)
    }
    if has_site(tree) {
        return Err(fail("S1600", "tool site name collides with source MJCF"));
    }
    fn visit(node: &mut Element, link: &str, site: &Element) -> bool {
        if node.tag == "body" && attr(node, "name") == Some(link) {
            node.children.push(site.clone());
            return true;
        }
        node.children
            .iter_mut()
            .any(|child| visit(child, link, site))
    }
    let mut attrs = vec![
        ("name", "invar_tool".to_owned()),
        ("size", "0.001".to_owned()),
    ];
    attrs.extend(pose_attrs(&robot["tool"]["pose"])?);
    if !visit(tree, link, &element("site", &attrs, Vec::new())) {
        return Err(fail("S1601", "tool link is missing from MJCF"));
    }
    Ok(())
}

/// `robot_id` selects which robot-qualified mount exclusions become MJCF excludes.
pub(super) fn find_body<'a>(node: &'a mut Element, name: &str) -> Option<&'a mut Element> {
    if node.tag == "body" && attr(node, "name") == Some(name) {
        return Some(node);
    }
    node.children
        .iter_mut()
        .find_map(|child| find_body(child, name))
}

pub(super) fn gravity_compensate(body: &mut Element) -> Result<(), XmlError> {
    if body.tag == "body" {
        match attr(body, "gravcomp") {
            None => {
                body.attrs.insert("gravcomp".to_owned(), "1".to_owned());
            }
            Some("1") => {}
            Some(_) => {
                return Err(fail(
                    "S1607",
                    "robot body declares partial gravity compensation",
                ));
            }
        }
    }
    body.children.iter_mut().try_for_each(gravity_compensate)
}
