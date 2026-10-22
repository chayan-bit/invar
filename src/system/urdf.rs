//! Exact-decimal URDF subset and deterministic ros2_control target.

use std::{collections::BTreeMap, path::Path};

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::xml::{Element, XmlError};

fn fail(message: impl Into<String>) -> XmlError {
    XmlError {
        code: "S1607",
        message: message.into(),
    }
}

fn element(tag: &str, attrs: &[(&str, String)], children: Vec<Element>) -> Element {
    Element {
        tag: tag.to_owned(),
        attrs: attrs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), value.clone()))
            .collect(),
        children,
    }
}

fn required(node: roxmltree::Node<'_, '_>, name: &str) -> Result<String, XmlError> {
    node.attribute(name)
        .map(str::to_owned)
        .ok_or_else(|| fail(format!("URDF <{}> needs {name}", node.tag_name().name())))
}

fn identifier(raw: &str) -> Result<&str, XmlError> {
    if raw.len() > 128
        || !raw
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        || !raw
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(fail(format!(
            "URDF name cannot be an Invar identifier: {raw}"
        )));
    }
    Ok(raw)
}

fn decimal(raw: &str) -> Result<String, XmlError> {
    super::xml::exact_decimal(raw).ok_or_else(|| fail("invalid exact URDF decimal"))
}

fn numbers(raw: &str, count: usize) -> Result<Vec<String>, XmlError> {
    let parts = raw
        .split_whitespace()
        .map(decimal)
        .collect::<Result<Vec<_>, _>>()?;
    if parts.len() != count {
        return Err(fail(format!("URDF vector needs {count} values")));
    }
    Ok(parts)
}

fn half_decimal(raw: &str) -> Result<String, XmlError> {
    let raw = decimal(raw)?;
    let (whole, fraction) = raw.split_once('.').unwrap_or((&raw, ""));
    let digits: num_bigint::BigInt = format!("{whole}{fraction}")
        .parse()
        .map_err(|_| fail("invalid URDF geometry dimension"))?;
    let mut text = (digits * num_bigint::BigInt::from(5)).to_string();
    let scale = fraction.len() + 1;
    if text.len() <= scale {
        text = format!("{}{}", "0".repeat(scale + 1 - text.len()), text);
    }
    text.insert(text.len() - scale, '.');
    decimal(&text)
}

fn one_child<'a, 'input>(
    node: roxmltree::Node<'a, 'input>,
    tag: &str,
) -> Result<roxmltree::Node<'a, 'input>, XmlError> {
    let mut children = node.children().filter(|child| child.has_tag_name(tag));
    let child = children
        .next()
        .ok_or_else(|| fail(format!("URDF <{}> needs <{tag}>", node.tag_name().name())))?;
    if children.next().is_some() {
        return Err(fail(format!(
            "URDF <{}> repeats <{tag}>",
            node.tag_name().name()
        )));
    }
    Ok(child)
}

fn only_children(node: roxmltree::Node<'_, '_>, names: &[&str]) -> Result<(), XmlError> {
    for child in node.children() {
        if child.is_element() && !names.contains(&child.tag_name().name()) {
            return Err(fail(format!(
                "unsupported URDF <{}> in <{}>",
                child.tag_name().name(),
                node.tag_name().name()
            )));
        }
        if child.is_text()
            && !matches!(node.tag_name().name(), "plugin" | "param")
            && !child.text().unwrap_or("").trim().is_empty()
        {
            return Err(fail("URDF element text is unsupported"));
        }
    }
    Ok(())
}

fn only_attributes(node: roxmltree::Node<'_, '_>, names: &[&str]) -> Result<(), XmlError> {
    for attribute in node.attributes() {
        if !names.contains(&attribute.name()) {
            return Err(fail(format!(
                "unsupported URDF {} attribute on <{}>",
                attribute.name(),
                node.tag_name().name()
            )));
        }
    }
    Ok(())
}

fn canonical_node(node: roxmltree::Node<'_, '_>, count: &mut usize) -> Result<Element, XmlError> {
    *count += 1;
    if *count > 4096 {
        return Err(fail("URDF element count limit exceeded"));
    }
    let mut attrs = BTreeMap::new();
    for attribute in node.attributes() {
        let value = if matches!(
            attribute.name(),
            "xyz"
                | "rpy"
                | "size"
                | "radius"
                | "length"
                | "value"
                | "ixx"
                | "iyy"
                | "izz"
                | "ixy"
                | "ixz"
                | "iyz"
                | "lower"
                | "upper"
                | "velocity"
                | "effort"
        ) {
            attribute
                .value()
                .split_whitespace()
                .map(decimal)
                .collect::<Result<Vec<_>, _>>()?
                .join(" ")
        } else {
            attribute.value().to_owned()
        };
        attrs.insert(attribute.name().to_owned(), value);
    }
    let mut children = Vec::new();
    for child in node.children() {
        if child.is_element() {
            children.push(canonical_node(child, count)?);
        } else if child.is_text() && !child.text().unwrap_or("").trim().is_empty() {
            if !matches!(node.tag_name().name(), "plugin" | "param") {
                return Err(fail("URDF element text is unsupported"));
            }
            children.push(text(child.text().unwrap_or("").trim()));
        }
    }
    if node.has_tag_name("robot") {
        children.sort_by_key(|child| {
            let group = match child.tag.as_str() {
                "link" => 0,
                "joint" => 1,
                "ros2_control" => 2,
                _ => 3,
            };
            (group, child.attrs.get("name").cloned().unwrap_or_default())
        });
    }
    Ok(Element {
        tag: node.tag_name().name().to_owned(),
        attrs,
        children,
    })
}

pub(super) fn canonical_urdf(path: &Path) -> Result<String, XmlError> {
    let bytes = std::fs::read(path).map_err(|_| fail("cannot read URDF source"))?;
    if bytes.len() > 1_048_576 {
        return Err(fail("URDF XML byte limit exceeded"));
    }
    let source = std::str::from_utf8(&bytes).map_err(|_| fail("URDF must be UTF-8"))?;
    let document = roxmltree::Document::parse(source).map_err(|_| fail("invalid URDF XML"))?;
    if !document.root_element().has_tag_name("robot") {
        return Err(fail("URDF root must be robot"));
    }
    canonical_node(document.root_element(), &mut 0).map(|tree| tree.render())
}

fn origin_source(node: Option<roxmltree::Node<'_, '_>>) -> Result<String, XmlError> {
    if let Some(node) = node {
        only_children(node, &[])?;
        only_attributes(node, &["xyz", "rpy"])?;
    }
    let xyz = node
        .and_then(|node| node.attribute("xyz"))
        .unwrap_or("0 0 0");
    let rpy = node
        .and_then(|node| node.attribute("rpy"))
        .unwrap_or("0 0 0");
    if numbers(rpy, 3)? != ["0", "0", "0"] {
        return Err(fail(
            "URDF RPY cannot be represented exactly as a decimal quaternion",
        ));
    }
    let xyz = numbers(xyz, 3)?;
    Ok(format!(
        "xyz ({} m, {} m, {} m) quat (1, 0, 0, 0)",
        xyz[0], xyz[1], xyz[2]
    ))
}

fn geometry_source(node: roxmltree::Node<'_, '_>, root: &Path) -> Result<String, XmlError> {
    only_children(node, &["origin", "geometry"])?;
    only_attributes(node, &[])?;
    if node.children().any(|child| child.has_tag_name("origin")) {
        let pose = origin_source(Some(one_child(node, "origin")?))?;
        if pose != "xyz (0 m, 0 m, 0 m) quat (1, 0, 0, 0)" {
            return Err(fail(
                "URDF geometry origin needs an explicit Invar pose mapping",
            ));
        }
    }
    let wrapper = one_child(node, "geometry")?;
    only_attributes(wrapper, &[])?;
    let children = wrapper
        .children()
        .filter(|child| child.is_element())
        .collect::<Vec<_>>();
    if children.len() != 1 {
        return Err(fail("URDF geometry needs exactly one shape"));
    }
    let shape = children[0];
    only_children(shape, &[])?;
    let role = node.tag_name().name();
    match shape.tag_name().name() {
        "box" => {
            only_attributes(shape, &["size"])?;
            let size = numbers(&required(shape, "size")?, 3)?;
            Ok(format!(
                "        {role} box ({}, {}, {}) m;\n",
                half_decimal(&size[0])?,
                half_decimal(&size[1])?,
                half_decimal(&size[2])?
            ))
        }
        "sphere" => {
            only_attributes(shape, &["radius"])?;
            Ok(format!(
                "        {role} sphere {} m;\n",
                decimal(&required(shape, "radius")?)?
            ))
        }
        "cylinder" => {
            only_attributes(shape, &["radius", "length"])?;
            Ok(format!(
                "        {role} cylinder ({}, {}) m;\n",
                decimal(&required(shape, "radius")?)?,
                half_decimal(&required(shape, "length")?)?
            ))
        }
        "mesh" => {
            only_attributes(shape, &["filename", "scale"])?;
            let path = required(shape, "filename")?;
            if shape.attribute("scale").is_some() {
                return Err(fail("URDF mesh scaling is unsupported"));
            }
            let bytes = super::xml::model_file(root, &path)?;
            let digest = Sha256::digest(bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            Ok(format!(
                "        {role} mesh \"{path}\" sha256 \"{digest}\";\n"
            ))
        }
        other => Err(fail(format!("unsupported URDF geometry: {other}"))),
    }
}

pub(super) fn import_urdf(path: &Path) -> Result<(String, Vec<String>), XmlError> {
    let bytes = std::fs::read(path).map_err(|_| fail("cannot read URDF source"))?;
    if bytes.len() > 1_048_576 {
        return Err(fail("URDF XML byte limit exceeded"));
    }
    let source = std::str::from_utf8(&bytes).map_err(|_| fail("URDF must be UTF-8"))?;
    let document = roxmltree::Document::parse(source).map_err(|_| fail("invalid URDF XML"))?;
    let root = document.root_element();
    if !root.has_tag_name("robot") {
        return Err(fail("URDF root must be robot"));
    }
    only_children(root, &["link", "joint", "ros2_control"])?;
    only_attributes(root, &["name"])?;
    if root.descendants().filter(|node| node.is_element()).count() > 4096 {
        return Err(fail("URDF element count limit exceeded"));
    }
    let name = required(root, "name")?;
    identifier(&name)?;
    let model_dir = path
        .parent()
        .ok_or_else(|| fail("URDF model directory is missing"))?;
    let license_text =
        std::fs::read_to_string(model_dir.join("LICENSE")).map_err(|_| XmlError {
            code: "S1606",
            message: "URDF model LICENSE is missing".into(),
        })?;
    let license = if license_text.contains("Apache License") && license_text.contains("Version 2.0")
    {
        "Apache-2.0"
    } else if license_text.contains("Redistribution and use in source and binary forms")
        && license_text.contains("Neither the name")
    {
        "BSD-3-Clause"
    } else {
        return Err(XmlError {
            code: "S1606",
            message: "unsupported or ambiguous URDF asset license".into(),
        });
    };
    let mut links = BTreeMap::new();
    for link in root.children().filter(|child| child.has_tag_name("link")) {
        only_children(link, &["inertial", "visual", "collision"])?;
        only_attributes(link, &["name"])?;
        let link_name = required(link, "name")?;
        identifier(&link_name)?;
        if links.insert(link_name, link).is_some() {
            return Err(fail("duplicate URDF link"));
        }
    }
    if links.is_empty() || links.len() > 128 {
        return Err(fail("URDF link count is unsupported"));
    }
    let mut joints = BTreeMap::new();
    let mut by_child = BTreeMap::new();
    for joint in root.children().filter(|child| child.has_tag_name("joint")) {
        only_children(joint, &["parent", "child", "origin", "axis", "limit"])?;
        only_attributes(joint, &["name", "type"])?;
        for item in joint.children().filter(|child| child.is_element()) {
            match item.tag_name().name() {
                "parent" | "child" => {
                    only_children(item, &[])?;
                    only_attributes(item, &["link"])?;
                }
                "axis" => {
                    only_children(item, &[])?;
                    only_attributes(item, &["xyz"])?;
                }
                "limit" => {
                    only_children(item, &[])?;
                    only_attributes(item, &["lower", "upper", "velocity", "effort"])?;
                }
                _ => {}
            }
        }
        let joint_name = required(joint, "name")?;
        identifier(&joint_name)?;
        let parent = required(one_child(joint, "parent")?, "link")?;
        let child = required(one_child(joint, "child")?, "link")?;
        if !links.contains_key(&parent) || !links.contains_key(&child) {
            return Err(XmlError {
                code: "S1601",
                message: "URDF joint link is missing".into(),
            });
        }
        if by_child.insert(child, joint).is_some() || joints.insert(joint_name, joint).is_some() {
            return Err(XmlError {
                code: "S1601",
                message: "URDF joint child or name repeats".into(),
            });
        }
    }
    let roots = links
        .keys()
        .filter(|name| !by_child.contains_key(*name))
        .collect::<Vec<_>>();
    if roots.len() != 1 {
        return Err(XmlError {
            code: "S1601",
            message: "URDF needs one kinematic root".into(),
        });
    }
    let controls = root
        .children()
        .filter(|child| child.has_tag_name("ros2_control"))
        .collect::<Vec<_>>();
    if controls.len() > 1 {
        return Err(fail("URDF repeats ros2_control"));
    }
    let has_control = !controls.is_empty();
    for control in controls {
        only_children(control, &["hardware", "joint"])?;
        only_attributes(control, &["name", "type"])?;
        if control.attribute("type") != Some("system")
            || control.attribute("name") != Some("InvarHardware")
        {
            return Err(fail("unsupported URDF ros2_control type"));
        }
        let hardware = one_child(control, "hardware")?;
        only_children(hardware, &["plugin", "param"])?;
        only_attributes(hardware, &[])?;
        let plugin = one_child(hardware, "plugin")?;
        only_attributes(plugin, &[])?;
        only_children(plugin, &[])?;
        if plugin.text().unwrap_or("").trim() != "mujoco_ros2_control/MujocoSystemInterface" {
            return Err(fail("unsupported URDF ros2_control hardware plugin"));
        }
        let model = one_child(hardware, "param")?;
        only_attributes(model, &["name"])?;
        only_children(model, &[])?;
        if model.attribute("name") != Some("mujoco_model")
            || model.text().unwrap_or("").trim() != "scene.xml"
        {
            return Err(fail("URDF ros2_control model must be generated scene.xml"));
        }
        let mut controlled = BTreeMap::new();
        for item in control
            .children()
            .filter(|child| child.has_tag_name("joint"))
        {
            only_attributes(item, &["name"])?;
            let name = required(item, "name")?;
            if !joints.contains_key(&name) {
                return Err(XmlError {
                    code: "S1601",
                    message: "ros2_control joint is missing".into(),
                });
            }
            if controlled.insert(name, item).is_some() {
                return Err(fail("ros2_control joint repeats"));
            }
            only_children(item, &["command_interface", "state_interface"])?;
            let mut interfaces = Vec::new();
            for interface in item.children().filter(|child| child.is_element()) {
                only_children(interface, &[])?;
                only_attributes(interface, &["name"])?;
                if !matches!(interface.attribute("name"), Some("position" | "velocity")) {
                    return Err(fail("unsupported ros2_control interface"));
                }
                interfaces.push((interface.tag_name().name(), required(interface, "name")?));
            }
            interfaces.sort();
            if interfaces
                != [
                    ("command_interface", "position".to_owned()),
                    ("state_interface", "position".to_owned()),
                    ("state_interface", "velocity".to_owned()),
                ]
            {
                return Err(fail(
                    "ros2_control interfaces differ from the generated contract",
                ));
            }
        }
        if joints
            .iter()
            .filter(|(_, joint)| joint.attribute("type") != Some("fixed"))
            .any(|(name, _)| !controlled.contains_key(name))
            || controlled.len()
                != joints
                    .iter()
                    .filter(|(_, joint)| joint.attribute("type") != Some("fixed"))
                    .count()
        {
            return Err(fail("ros2_control joints differ from the robot"));
        }
    }
    let mut out = format!("robot {name} {{\n    license \"{license}\";\n");
    for (link_name, link) in &links {
        let (parent, pose) = if let Some(joint) = by_child.get(link_name) {
            (
                required(one_child(*joint, "parent")?, "link")?,
                origin_source(joint.children().find(|child| child.has_tag_name("origin")))?,
            )
        } else {
            ("world".to_owned(), origin_source(None)?)
        };
        let inertial = one_child(*link, "inertial")?;
        only_children(inertial, &["origin", "mass", "inertia"])?;
        only_attributes(inertial, &[])?;
        let mass_node = one_child(inertial, "mass")?;
        only_children(mass_node, &[])?;
        only_attributes(mass_node, &["value"])?;
        let mass = decimal(&required(mass_node, "value")?)?;
        let inertia = one_child(inertial, "inertia")?;
        only_children(inertia, &[])?;
        only_attributes(inertia, &["ixx", "iyy", "izz", "ixy", "ixz", "iyz"])?;
        let tensor = ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"]
            .iter()
            .map(|key| decimal(&required(inertia, key)?))
            .collect::<Result<Vec<_>, _>>()?;
        out.push_str(&format!("    link {link_name} {{\n        parent {parent};\n        origin {pose};\n        mass {mass} kg;\n        inertial_origin {};\n        inertia ({}) kg*m*m;\n",origin_source(inertial.children().find(|child|child.has_tag_name("origin")))?,tensor.join(", ")));
        for geom in link
            .children()
            .filter(|child| child.has_tag_name("visual") || child.has_tag_name("collision"))
        {
            out.push_str(&geometry_source(geom, model_dir)?);
        }
        out.push_str("    }\n");
    }
    for (joint_name, joint) in joints {
        let kind = required(joint, "type")?;
        let kind = match kind.as_str() {
            "revolute" | "continuous" => "revolute",
            "prismatic" => "prismatic",
            "fixed" => "fixed",
            _ => return Err(fail("unsupported URDF joint kind")),
        };
        let parent = required(one_child(joint, "parent")?, "link")?;
        let child = required(one_child(joint, "child")?, "link")?;
        let axis = if kind == "fixed" {
            vec!["0".into(), "0".into(), "1".into()]
        } else {
            numbers(&required(one_child(joint, "axis")?, "xyz")?, 3)?
        };
        let unit = if kind == "prismatic" { "m" } else { "rad" };
        let speed = if kind == "prismatic" { "m/s" } else { "rad/s" };
        let force = if kind == "prismatic" { "N" } else { "N*m" };
        let limit = joint.children().find(|child| child.has_tag_name("limit"));
        let position = if kind == "fixed" || joint.attribute("type") == Some("continuous") {
            "unbounded".to_owned()
        } else {
            let limit = limit.ok_or_else(|| fail("URDF joint limit is missing"))?;
            format!(
                "{} {unit} .. {} {unit}",
                decimal(&required(limit, "lower")?)?,
                decimal(&required(limit, "upper")?)?
            )
        };
        let velocity = limit
            .and_then(|limit| limit.attribute("velocity"))
            .map(decimal)
            .transpose()?
            .map_or("unbounded".to_owned(), |value| format!("{value} {speed}"));
        let effort = limit
            .and_then(|limit| limit.attribute("effort"))
            .map(decimal)
            .transpose()?
            .map_or("unbounded".to_owned(), |value| format!("{value} {force}"));
        out.push_str(&format!("    joint {joint_name} {{\n        kind {kind};\n        parent {parent};\n        child {child};\n        axis ({}, {}, {});\n        position {position};\n        velocity {velocity};\n        effort {effort};\n    }}\n",axis[0],axis[1],axis[2]));
        if has_control && kind != "fixed" {
            let limit =
                limit.ok_or_else(|| fail("ros2_control position command needs a bounded joint"))?;
            let lower = decimal(&required(limit, "lower")?)?;
            let upper = decimal(&required(limit, "upper")?)?;
            out.push_str(&format!("    actuator {joint_name}_drive {{\n        mode position;\n        joint {joint_name};\n        gear 1;\n        control {lower} {unit} .. {upper} {unit};\n    }}\n"));
        }
    }
    out.push_str("}\n\n");
    out.push_str(&format!("product Imported {{\n    distribution internal;\n    commercial false;\n    robot {name};\n    capacity cpu_cores = 1, memory = 1 GiB;\n}}\nsystem imported for Imported {{}}\n"));
    Ok((out, Vec::new()))
}

fn text(value: &str) -> Element {
    element("#text", &[("value", value.to_owned())], Vec::new())
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a str, XmlError> {
    value[name]
        .as_str()
        .ok_or_else(|| fail(format!("checked {name} is missing")))
}

fn values(value: &Value, quantity: bool) -> Result<String, XmlError> {
    value
        .as_array()
        .ok_or_else(|| fail("checked vector is missing"))?
        .iter()
        .map(|part| {
            if quantity {
                part["value"].as_str()
            } else {
                part.as_str()
            }
            .map(str::to_owned)
            .ok_or_else(|| fail("checked vector contains a non-decimal"))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|parts| parts.join(" "))
}

fn double_decimal(value: &str) -> Result<String, XmlError> {
    let (whole, fraction) = value.split_once('.').unwrap_or((value, ""));
    let digits: num_bigint::BigInt = format!("{whole}{fraction}")
        .parse::<num_bigint::BigInt>()
        .map_err(|_| fail("geometry dimension is not a decimal"))?
        * 2;
    let mut text = digits.to_string();
    if !fraction.is_empty() {
        if text.len() <= fraction.len() {
            text = format!("{}{}", "0".repeat(fraction.len() + 1 - text.len()), text);
        }
        text.insert(text.len() - fraction.len(), '.');
    }
    super::xml::exact_decimal(&text).ok_or_else(|| fail("doubled geometry dimension is invalid"))
}

fn identity_rotation(pose: &Value) -> Result<(), XmlError> {
    if values(&pose["quat"], false)? != "1 0 0 0" {
        return Err(fail(
            "standard URDF origin cannot preserve this exact quaternion",
        ));
    }
    Ok(())
}

fn origin(pose: &Value) -> Result<Element, XmlError> {
    identity_rotation(pose)?;
    Ok(element(
        "origin",
        &[
            ("xyz", values(&pose["xyz"], true)?),
            ("rpy", "0 0 0".to_owned()),
        ],
        Vec::new(),
    ))
}

fn geometry(geometry: &Value) -> Result<Element, XmlError> {
    let kind = field(geometry, "kind")?;
    let inner = match kind {
        "box" => element(
            "box",
            &[(
                "size",
                geometry["dimensions"]
                    .as_array()
                    .ok_or_else(|| fail("box dimensions are missing"))?
                    .iter()
                    .map(|part| {
                        double_decimal(
                            part.as_str()
                                .ok_or_else(|| fail("box dimension is invalid"))?,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .join(" "),
            )],
            Vec::new(),
        ),
        "sphere" => element(
            "sphere",
            &[(
                "radius",
                geometry["dimensions"][0]
                    .as_str()
                    .ok_or_else(|| fail("sphere radius is missing"))?
                    .to_owned(),
            )],
            Vec::new(),
        ),
        "cylinder" => element(
            "cylinder",
            &[
                (
                    "radius",
                    geometry["dimensions"][0]
                        .as_str()
                        .ok_or_else(|| fail("cylinder radius is missing"))?
                        .to_owned(),
                ),
                (
                    "length",
                    double_decimal(
                        geometry["dimensions"][1]
                            .as_str()
                            .ok_or_else(|| fail("cylinder half-length is missing"))?,
                    )?,
                ),
            ],
            Vec::new(),
        ),
        "mesh" => element(
            "mesh",
            &[("filename", field(geometry, "path")?.to_owned())],
            Vec::new(),
        ),
        "capsule" => {
            return Err(fail(
                "standard URDF cannot represent capsule collision geometry exactly",
            ));
        }
        _ => return Err(fail("unsupported checked URDF geometry")),
    };
    Ok(element(
        field(geometry, "role")?,
        &[],
        vec![element("geometry", &[], vec![inner])],
    ))
}

pub(super) fn from_robot(robot: &Value) -> Result<(String, Vec<String>), XmlError> {
    let name = field(robot, "name")?;
    if robot["source_tree"]["children"]
        .as_array()
        .is_some_and(|children| {
            children.iter().any(|child| {
                matches!(
                    child["tag"].as_str(),
                    Some("tendon" | "equality" | "keyframe")
                )
            })
        })
    {
        return Err(fail(
            "standard URDF cannot preserve MJCF tendon, equality, or keyframe semantics",
        ));
    }
    let mut warnings = Vec::new();
    if !robot["sensors"].as_array().is_some_and(Vec::is_empty) {
        warnings.push("S1604 warning: robot sensors are omitted from standard URDF; the generated scene.xml retains them".to_owned());
    }
    let mut links = robot["links"]
        .as_array()
        .ok_or_else(|| fail("checked links are missing"))?
        .iter()
        .collect::<Vec<_>>();
    links.sort_by_key(|link| link["name"].as_str().unwrap_or(""));
    let mut joints = robot["joints"]
        .as_array()
        .ok_or_else(|| fail("checked joints are missing"))?
        .iter()
        .collect::<Vec<_>>();
    joints.sort_by_key(|joint| joint["name"].as_str().unwrap_or(""));
    let actuators = robot["actuators"]
        .as_array()
        .ok_or_else(|| fail("checked actuators are missing"))?;
    if actuators.len()
        != joints
            .iter()
            .filter(|joint| joint["kind"] != "fixed")
            .count()
    {
        return Err(fail(
            "standard URDF ros2_control needs one position actuator per moving joint",
        ));
    }
    for joint in &joints {
        if joint["kind"] == "fixed" {
            continue;
        }
        let matching = actuators
            .iter()
            .filter(|actuator| {
                actuator["target_kind"] == "joint" && actuator["target"] == joint["name"]
            })
            .collect::<Vec<_>>();
        if matching.len() != 1
            || matching[0]["mode"] != "position"
            || matching[0]["gear"] != "1"
            || matching[0]["control"] != joint["position"]
        {
            return Err(fail(
                "standard URDF ros2_control cannot preserve this actuator contract",
            ));
        }
    }
    let mut children = Vec::new();
    for link in &links {
        let inertial_origin = if link["inertial_origin"].is_null() {
            element(
                "origin",
                &[("xyz", "0 0 0".to_owned()), ("rpy", "0 0 0".to_owned())],
                Vec::new(),
            )
        } else {
            origin(&link["inertial_origin"])?
        };
        let mut inertial = vec![inertial_origin];
        inertial.push(element(
            "mass",
            &[("value", field(&link["mass"], "value")?.to_owned())],
            Vec::new(),
        ));
        let tensor = link["inertia"]
            .as_array()
            .ok_or_else(|| fail("checked inertia is missing"))?;
        if tensor.len() != 6 {
            return Err(fail("checked inertia needs six values"));
        }
        inertial.push(element(
            "inertia",
            &[
                ("ixx", tensor[0].as_str().unwrap().to_owned()),
                ("iyy", tensor[1].as_str().unwrap().to_owned()),
                ("izz", tensor[2].as_str().unwrap().to_owned()),
                ("ixy", tensor[3].as_str().unwrap().to_owned()),
                ("ixz", tensor[4].as_str().unwrap().to_owned()),
                ("iyz", tensor[5].as_str().unwrap().to_owned()),
            ],
            Vec::new(),
        ));
        let mut link_children = vec![element("inertial", &[], inertial)];
        for item in link["geometries"]
            .as_array()
            .ok_or_else(|| fail("checked link geometry is missing"))?
        {
            link_children.push(geometry(item)?);
        }
        children.push(element(
            "link",
            &[("name", field(link, "name")?.to_owned())],
            link_children,
        ));
    }
    let mut ros_joints = Vec::new();
    for joint in &joints {
        let joint_name = field(joint, "name")?;
        let kind = field(joint, "kind")?;
        let source_kind = if kind == "revolute" && joint["position"].is_null() {
            "continuous"
        } else {
            kind
        };
        let child = field(joint, "child")?;
        let child_link = links
            .iter()
            .find(|link| link["name"] == child)
            .ok_or_else(|| fail("joint child link is missing"))?;
        let mut content = vec![
            element(
                "parent",
                &[("link", field(joint, "parent")?.to_owned())],
                Vec::new(),
            ),
            element("child", &[("link", child.to_owned())], Vec::new()),
            origin(&child_link["origin"])?,
        ];
        if kind != "fixed" {
            content.push(element(
                "axis",
                &[("xyz", values(&joint["axis"], false)?)],
                Vec::new(),
            ));
            let mut limits = vec![
                (
                    "effort",
                    joint["effort"]["value"].as_str().unwrap_or("0").to_owned(),
                ),
                (
                    "velocity",
                    joint["velocity"]["value"]
                        .as_str()
                        .unwrap_or("0")
                        .to_owned(),
                ),
            ];
            if !joint["position"].is_null() {
                limits.push((
                    "lower",
                    field(&joint["position"]["lower"], "value")?.to_owned(),
                ));
                limits.push((
                    "upper",
                    field(&joint["position"]["upper"], "value")?.to_owned(),
                ));
            }
            content.push(element("limit", &limits, Vec::new()));
            ros_joints.push(element(
                "joint",
                &[("name", joint_name.to_owned())],
                vec![
                    element(
                        "command_interface",
                        &[("name", "position".to_owned())],
                        Vec::new(),
                    ),
                    element(
                        "state_interface",
                        &[("name", "position".to_owned())],
                        Vec::new(),
                    ),
                    element(
                        "state_interface",
                        &[("name", "velocity".to_owned())],
                        Vec::new(),
                    ),
                ],
            ));
        }
        children.push(element(
            "joint",
            &[
                ("name", joint_name.to_owned()),
                ("type", source_kind.to_owned()),
            ],
            content,
        ));
    }
    let control = element(
        "ros2_control",
        &[
            ("name", "InvarHardware".to_owned()),
            ("type", "system".to_owned()),
        ],
        {
            let mut content = vec![element(
                "hardware",
                &[],
                vec![
                    element(
                        "plugin",
                        &[],
                        vec![text("mujoco_ros2_control/MujocoSystemInterface")],
                    ),
                    element(
                        "param",
                        &[("name", "mujoco_model".to_owned())],
                        vec![text("scene.xml")],
                    ),
                ],
            )];
            content.extend(ros_joints);
            content
        },
    );
    children.push(control);
    Ok((
        element("robot", &[("name", name.to_owned())], children).render(),
        warnings,
    ))
}
