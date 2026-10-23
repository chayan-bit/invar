//! Checked robot and cell declarations for the system IR.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Component, Path},
};

use num_rational::BigRational;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::front::{
    CellModelDecl, FrontError, RobotActuatorDecl, RobotGeometryDecl, RobotJointDecl, RobotLinkDecl,
    RobotModelDecl, RobotPoseDecl, RobotSensorDecl, Span, SystemQuantity,
};
use crate::robotics::{CameraIntrinsics, Distortion};

use super::{decimal, error, graph, quantity, xml};

fn failure(code: &'static str, message: impl Into<String>, span: Span) -> FrontError {
    error(code, message, span)
}

pub(super) fn hash_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn number(raw: &str, span: Span) -> Result<(String, BigRational), FrontError> {
    let text = decimal(raw, span).map_err(|_| failure("S1600", "invalid robot decimal", span))?;
    let value =
        graph::decimal(&text).ok_or_else(|| failure("S1600", "invalid robot decimal", span))?;
    Ok((text, value))
}

fn measured(value: &SystemQuantity, unit: &str) -> Result<Value, FrontError> {
    if value.unit != unit {
        return Err(failure(
            "S1600",
            format!("expected {unit} unit"),
            value.span,
        ));
    }
    quantity(value).map_err(|_| failure("S1600", "invalid robot quantity", value.span))
}

fn positive(value: &SystemQuantity, unit: &str) -> Result<Value, FrontError> {
    let result = measured(value, unit)?;
    if graph::decimal(result["value"].as_str().unwrap_or("0"))
        .is_none_or(|number| number <= BigRational::from_integer(0.into()))
    {
        return Err(failure("S1602", "quantity must be positive", value.span));
    }
    Ok(result)
}

fn pose(value: &RobotPoseDecl, span: Span) -> Result<Value, FrontError> {
    if value.xyz.len() != 3 || value.quat.len() != 4 {
        return Err(failure(
            "S1600",
            "pose must have three translations and four quaternion values",
            span,
        ));
    }
    let xyz = value
        .xyz
        .iter()
        .map(|part| measured(part, "m"))
        .collect::<Result<Vec<_>, _>>()?;
    let quat = value
        .quat
        .iter()
        .map(|part| number(part, span))
        .collect::<Result<Vec<_>, _>>()?;
    if quat
        .iter()
        .all(|(_, value)| *value == BigRational::from_integer(0.into()))
    {
        return Err(failure("S1600", "zero quaternion is not a rotation", span));
    }
    Ok(json!({"xyz":xyz,"quat":quat.into_iter().map(|(text, _)| text).collect::<Vec<_>>()}))
}

fn digest(value: &str, span: Span) -> Result<(), FrontError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(failure(
            "S1605",
            "asset SHA-256 must be 64 hex digits",
            span,
        ));
    }
    Ok(())
}

fn relative_path(value: &str, span: Span) -> Result<(), FrontError> {
    let path = Path::new(value);
    if value.is_empty()
        || value.len() > 512
        || value
            .chars()
            .any(|ch| ch == '"' || ch == '\\' || ch.is_control())
        || !path
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
    {
        return Err(failure(
            "S1605",
            "asset path must be bounded and relative without traversal",
            span,
        ));
    }
    Ok(())
}

fn geometry(value: &RobotGeometryDecl, span: Span) -> Result<Value, FrontError> {
    match value.kind.as_str() {
        "mesh" => {
            let path = value
                .path
                .as_deref()
                .ok_or_else(|| failure("S1605", "mesh path missing", span))?;
            let sha256 = value
                .sha256
                .as_deref()
                .ok_or_else(|| failure("S1605", "mesh hash missing", span))?;
            relative_path(path, span)?;
            digest(sha256, span)?;
            Ok(json!({"role":value.role,"kind":"mesh","dimensions":[],"path":path,"sha256":sha256}))
        }
        "box" | "sphere" | "cylinder" | "capsule" => {
            let count = match value.kind.as_str() {
                "box" => 3,
                "sphere" => 1,
                _ => 2,
            };
            if value.dimensions.len() != count {
                return Err(failure(
                    "S1600",
                    "geometry dimension count differs from kind",
                    span,
                ));
            }
            let dimensions = value
                .dimensions
                .iter()
                .map(|raw| {
                    let (text, part) = number(raw, span)?;
                    if part <= BigRational::from_integer(0.into()) {
                        return Err(failure(
                            "S1602",
                            "geometry dimensions must be positive",
                            span,
                        ));
                    }
                    Ok(text)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(
                json!({"role":value.role,"kind":value.kind,"dimensions":dimensions,"path":null,"sha256":null}),
            )
        }
        _ => Err(failure("S1600", "unknown geometry kind", span)),
    }
}

fn link(value: &RobotLinkDecl, span: Span) -> Result<Value, FrontError> {
    let mass = positive(&value.mass, "kg")?;
    if value.inertia.len() != 6 {
        return Err(failure("S1602", "inertia needs six tensor entries", span));
    }
    let entries = value
        .inertia
        .iter()
        .map(|raw| number(raw, span))
        .collect::<Result<Vec<_>, _>>()?;
    let v: Vec<_> = entries.iter().map(|(_, value)| value.clone()).collect();
    let zero = BigRational::from_integer(0.into());
    let minor2 = &v[0] * &v[1] - &v[3] * &v[3];
    let determinant = &v[0] * (&v[1] * &v[2] - &v[5] * &v[5])
        - &v[3] * (&v[3] * &v[2] - &v[4] * &v[5])
        + &v[4] * (&v[3] * &v[5] - &v[4] * &v[1]);
    if v[0] <= zero || minor2 <= zero || determinant <= zero {
        return Err(failure(
            "S1602",
            "inertia tensor is not positive definite",
            span,
        ));
    }
    let geometries = value
        .geometries
        .iter()
        .map(|item| geometry(item, span))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(
        json!({"name":value.name,"parent":value.parent,"origin":pose(&value.origin, span)?,
        "mass":mass,"inertial_origin":value.inertial_origin.as_ref().map(|origin| pose(origin, span)).transpose()?,
        "inertia":entries.into_iter().map(|(text, _)| text).collect::<Vec<_>>(),"geometries":geometries}),
    )
}

fn joint(value: &RobotJointDecl, span: Span) -> Result<Value, FrontError> {
    if value.axis.len() != 3 {
        return Err(failure("S1600", "joint axis must have three entries", span));
    }
    let axis = value
        .axis
        .iter()
        .map(|raw| number(raw, span))
        .collect::<Result<Vec<_>, _>>()?;
    if value.kind != "fixed"
        && axis
            .iter()
            .all(|(_, part)| *part == BigRational::from_integer(0.into()))
    {
        return Err(failure("S1600", "movable joint axis cannot be zero", span));
    }
    let position_unit = if value.kind == "prismatic" {
        "m"
    } else {
        "rad"
    };
    let velocity_unit = if value.kind == "prismatic" {
        "m/s"
    } else {
        "rad/s"
    };
    let effort_unit = if value.kind == "prismatic" {
        "N"
    } else {
        "N*m"
    };
    let position = value
        .position
        .as_ref()
        .map(|(lower, upper)| {
            let lower_doc = measured(lower, position_unit)?;
            let upper_doc = measured(upper, position_unit)?;
            let lower_value = graph::decimal(lower_doc["value"].as_str().unwrap()).unwrap();
            let upper_value = graph::decimal(upper_doc["value"].as_str().unwrap()).unwrap();
            if lower_value > upper_value {
                return Err(failure(
                    "S1603",
                    "joint lower limit exceeds upper limit",
                    span,
                ));
            }
            Ok(json!({"lower":lower_doc,"upper":upper_doc}))
        })
        .transpose()?;
    if value.kind == "fixed"
        && (position.is_some() || value.velocity.is_some() || value.effort.is_some())
    {
        return Err(failure(
            "S1600",
            "fixed joint cannot declare motion limits",
            span,
        ));
    }
    let velocity = value
        .velocity
        .as_ref()
        .map(|limit| positive(limit, velocity_unit))
        .transpose()?;
    let effort = value
        .effort
        .as_ref()
        .map(|limit| positive(limit, effort_unit))
        .transpose()?;
    Ok(
        json!({"name":value.name,"kind":value.kind,"parent":value.parent,"child":value.child,
        "axis":axis.into_iter().map(|(text, _)| text).collect::<Vec<_>>(),
        "position":position,"velocity":velocity,"effort":effort}),
    )
}

fn actuator(value: &RobotActuatorDecl, unit: &str, span: Span) -> Result<Value, FrontError> {
    let (gear, ratio) = number(&value.gear, span)?;
    if ratio == BigRational::from_integer(0.into()) {
        return Err(failure("S1600", "actuator gear cannot be zero", span));
    }
    let lower = measured(&value.control.0, unit)?;
    let upper = measured(&value.control.1, unit)?;
    if graph::decimal(lower["value"].as_str().unwrap()).unwrap()
        > graph::decimal(upper["value"].as_str().unwrap()).unwrap()
    {
        return Err(failure(
            "S1603",
            "actuator control lower bound exceeds upper bound",
            span,
        ));
    }
    let mut document = json!({"name":value.name,"mode":value.mode,"target_kind":value.target_kind,"target":value.target,"gear":gear,
        "control":{"lower":lower,"upper":upper}});
    if let Some(hold) = &value.hold {
        let hold = measured(hold, unit)?;
        let decimal = |v: &Value| graph::decimal(v["value"].as_str().unwrap()).unwrap();
        if decimal(&hold) < decimal(&lower) || decimal(&hold) > decimal(&upper) {
            return Err(failure(
                "S1603",
                "actuator hold command lies outside its control range",
                span,
            ));
        }
        document["hold"] = hold;
    }
    Ok(document)
}

fn sensor(value: &RobotSensorDecl, span: Span) -> Result<Value, FrontError> {
    if (value.kind == "camera") != value.intrinsics.is_some() {
        return Err(failure(
            "S1600",
            "camera needs intrinsics; other sensors cannot declare them",
            span,
        ));
    }
    let intrinsics = value
        .intrinsics
        .as_ref()
        .map(|item| {
            let width = positive(&item.width, "px")?["value"]
                .as_str()
                .unwrap()
                .to_owned();
            let height = positive(&item.height, "px")?["value"]
                .as_str()
                .unwrap()
                .to_owned();
            let fx = positive(&item.fx, "px")?["value"]
                .as_str()
                .unwrap()
                .to_owned();
            let fy = positive(&item.fy, "px")?["value"]
                .as_str()
                .unwrap()
                .to_owned();
            let cx = measured(&item.cx, "px")?["value"]
                .as_str()
                .unwrap()
                .to_owned();
            let cy = measured(&item.cy, "px")?["value"]
                .as_str()
                .unwrap()
                .to_owned();
            CameraIntrinsics {
                scalar: "f64".to_owned(),
                width_px: width,
                height_px: height,
                fx_px: fx,
                fy_px: fy,
                cx_px: cx,
                cy_px: cy,
                distortion: Distortion::None,
            }
            .to_ir()
            .map_err(|message| failure("S1600", message, span))
        })
        .transpose()?;
    Ok(json!({"name":value.name,"kind":value.kind,"source":value.source,"intrinsics":intrinsics}))
}

type CompiledRobot = (Value, BTreeMap<String, Vec<u8>>, Vec<String>);

pub(super) fn compile_robot(
    value: &RobotModelDecl,
    id: &str,
    span: Span,
    source_dir: Option<&Path>,
) -> Result<CompiledRobot, FrontError> {
    if value.links.is_empty()
        || value.links.len() > 128
        || value.joints.len() > 128
        || value.actuators.len() > 128
        || value.sensors.len() > 128
    {
        return Err(failure("S1600", "robot model size is outside limits", span));
    }
    if value.license.len() > 256 || spdx::Expression::parse(&value.license).is_err() {
        return Err(failure("S1606", "robot license is not valid SPDX", span));
    }
    let asset = value
        .asset
        .as_ref()
        .map(|asset| {
            relative_path(&asset.path, span)?;
            digest(&asset.sha256, span)?;
            Ok::<_, FrontError>(json!({"kind":asset.kind,"path":asset.path,"sha256":asset.sha256}))
        })
        .transpose()?;
    let (source_tree, asset_files, asset_bytes, warnings) = if let Some(asset) = &value.asset {
        let source_dir = source_dir
            .ok_or_else(|| failure("S1605", "asset-backed robot needs a source directory", span))?;
        let root = source_dir
            .canonicalize()
            .map_err(|_| failure("S1605", "source directory is missing", span))?;
        let path = root
            .join(&asset.path)
            .canonicalize()
            .map_err(|_| failure("S1605", "robot asset is missing", span))?;
        if !path.starts_with(&root) {
            return Err(failure(
                "S1605",
                "robot asset escapes source directory",
                span,
            ));
        }
        let bytes =
            std::fs::read(&path).map_err(|_| failure("S1605", "cannot read robot asset", span))?;
        let actual_hash = hash_bytes(&bytes);
        if bytes.len() > 1_048_576 || actual_hash != asset.sha256.to_ascii_lowercase() {
            return Err(failure(
                "S1605",
                "robot asset SHA-256 differs from declaration",
                span,
            ));
        }
        let license = std::fs::read_to_string(path.parent().unwrap().join("LICENSE"))
            .map_err(|_| failure("S1606", "robot asset LICENSE is missing", span))?;
        let license_matches = match value.license.as_str() {
            "Apache-2.0" => license.contains("Apache License") && license.contains("Version 2.0"),
            "BSD-3-Clause" => {
                license.contains("Redistribution and use in source and binary forms")
                    && license.contains("Neither the name")
            }
            other => license.contains(&format!("SPDX-License-Identifier: {other}")),
        };
        if !license_matches {
            return Err(failure(
                "S1606",
                "robot declared license differs from asset LICENSE",
                span,
            ));
        }
        if asset.kind != "mjcf" {
            return Err(failure(
                "S1607",
                "URDF asset import is not yet supported",
                span,
            ));
        }
        let mut tree =
            xml::load_mjcf(&path).map_err(|issue| failure(issue.code, issue.message, span))?;
        let warnings = xml::validate_supported(&mut tree)
            .map_err(|issue| failure(issue.code, issue.message, span))?;
        let model_dir = path.parent().unwrap();
        let files = xml::asset_files(&tree, model_dir)
            .map_err(|issue| failure(issue.code, issue.message, span))?;
        for geometry in value
            .links
            .iter()
            .flat_map(|link| &link.geometries)
            .filter(|geometry| geometry.kind == "mesh")
        {
            let mesh_path = geometry.path.as_deref().unwrap_or("");
            let bytes = xml::model_file(model_dir, mesh_path)
                .map_err(|issue| failure(issue.code, issue.message, span))?;
            if hash_bytes(&bytes)
                != geometry
                    .sha256
                    .as_deref()
                    .unwrap_or("")
                    .to_ascii_lowercase()
            {
                return Err(failure(
                    "S1605",
                    "mesh SHA-256 differs from declaration",
                    span,
                ));
            }
        }
        let mut asset_bytes = BTreeMap::new();
        for (file, expected) in &files {
            let bytes = xml::model_file(model_dir, file)
                .map_err(|issue| failure(issue.code, issue.message, span))?;
            if hash_bytes(&bytes) != *expected {
                return Err(failure("S1605", "MJCF asset changed while checking", span));
            }
            asset_bytes.insert(file.clone(), bytes);
        }
        asset_bytes.insert("LICENSE".to_owned(), license.into_bytes());
        (
            Some(tree.to_value()),
            files
                .into_iter()
                .map(|(path, sha256)| json!({"path":path,"sha256":sha256}))
                .collect::<Vec<_>>(),
            asset_bytes,
            warnings,
        )
    } else {
        (None, Vec::new(), BTreeMap::new(), Vec::new())
    };
    let mut names = BTreeSet::new();
    let mut links = Vec::new();
    for item in &value.links {
        if !names.insert(item.name.as_str()) {
            return Err(failure("S1600", "duplicate robot link", span));
        }
        links.push(link(item, span)?);
    }
    if links
        .iter()
        .filter(|item| item["parent"] == "world")
        .count()
        != 1
    {
        return Err(failure("S1601", "robot needs exactly one root link", span));
    }
    for item in &value.links {
        let mut ancestors = BTreeSet::new();
        let mut current = item;
        while current.parent != "world" {
            if !ancestors.insert(current.name.as_str()) {
                return Err(failure("S1601", "kinematic loop", span));
            }
            current = value
                .links
                .iter()
                .find(|candidate| candidate.name == current.parent)
                .ok_or_else(|| failure("S1601", "link parent is missing", span))?;
        }
    }
    names.clear();
    let mut joints = Vec::new();
    for item in &value.joints {
        if !names.insert(item.name.as_str()) {
            return Err(failure("S1600", "duplicate robot joint", span));
        }
        if !value.links.iter().any(|link| link.name == item.parent)
            || !value.links.iter().any(|link| link.name == item.child)
        {
            return Err(failure(
                "S1601",
                "joint parent or child link is missing",
                span,
            ));
        }
        if !value
            .links
            .iter()
            .any(|link| link.name == item.child && link.parent == item.parent)
        {
            return Err(failure(
                "S1601",
                "joint does not follow link frame tree",
                span,
            ));
        }
        joints.push(joint(item, span)?);
    }
    names.clear();
    let mut actuators = Vec::new();
    for item in &value.actuators {
        if !names.insert(item.name.as_str()) {
            return Err(failure("S1600", "duplicate actuator name", span));
        }
        let unit = if item.target_kind == "tendon" {
            if value.asset.is_none() {
                return Err(failure(
                    "S1601",
                    "tendon target needs a checked MJCF asset",
                    span,
                ));
            }
            "one"
        } else {
            let joint = value
                .joints
                .iter()
                .find(|joint| joint.name == item.target)
                .ok_or_else(|| failure("S1601", "actuator joint does not resolve", span))?;
            let prismatic = joint.kind == "prismatic";
            match (item.mode.as_str(), prismatic) {
                ("position", true) => "m",
                ("position", false) => "rad",
                ("velocity", true) => "m/s",
                ("velocity", false) => "rad/s",
                ("motor-torque", true) => "N",
                _ => "N*m",
            }
        };
        actuators.push(actuator(item, unit, span)?);
    }
    names.clear();
    let mut sensors = Vec::new();
    for item in &value.sensors {
        if !names.insert(item.name.as_str()) {
            return Err(failure("S1600", "duplicate robot sensor", span));
        }
        let target_exists = if item.kind == "camera" {
            value.links.iter().any(|link| link.name == item.source)
        } else {
            value.joints.iter().any(|joint| joint.name == item.source)
        };
        if !target_exists {
            return Err(failure("S1601", "sensor source does not resolve", span));
        }
        sensors.push(sensor(item, span)?);
    }
    let tool = value
        .tool
        .as_ref()
        .map(|(link, tool_pose)| {
            if !value.links.iter().any(|item| item.name == *link) {
                return Err(failure("S1601", "tool link does not resolve", span));
            }
            Ok::<_, FrontError>(json!({"link":link,"pose":pose(tool_pose, span)?}))
        })
        .transpose()?;
    let mut robot = json!({"id":id,"name":value.name,"license":value.license,"asset":asset,"asset_files":asset_files,"source_tree":source_tree,
        "links":links,"joints":joints,"actuators":actuators,"sensors":sensors});
    if let Some(tool) = tool {
        robot["tool"] = tool;
    }
    if !robot["source_tree"].is_null() {
        let tree = xml::Element::from_value(&robot["source_tree"])
            .ok_or_else(|| failure("S1607", "checked MJCF source tree is malformed", span))?;
        xml::verify_robot(&tree, &robot)
            .map_err(|issue| failure(issue.code, issue.message, span))?;
    }
    Ok((robot, asset_bytes, warnings))
}

pub(super) fn compile_cell(
    value: &CellModelDecl,
    id: &str,
    span: Span,
) -> Result<Value, FrontError> {
    if value.objects.len() > 128 || value.io.len() > 128 {
        return Err(failure("S1600", "cell size limit exceeded", span));
    }
    let mut names = BTreeSet::new();
    let mut objects = Vec::new();
    for object in &value.objects {
        if !names.insert(object.name.as_str()) {
            return Err(failure("S1600", "duplicate cell object", span));
        }
        let geometries = object
            .geometries
            .iter()
            .map(|item| geometry(item, span))
            .collect::<Result<Vec<_>, _>>()?;
        let dynamic = object
            .dynamic
            .as_ref()
            .map(|mass| {
                if object.kind != "object"
                    || !object
                        .geometries
                        .iter()
                        .any(|geometry| geometry.role == "collision")
                {
                    return Err(failure(
                        "S1602",
                        "attachable dynamic object needs collision geometry",
                        span,
                    ));
                }
                Ok::<_, FrontError>(json!({"mass":positive(mass, "kg")?,"attachable":true}))
            })
            .transpose()?;
        let mut item = json!({"kind":object.kind,"name":object.name,"pose":pose(&object.pose, span)?,"geometries":geometries});
        if let Some(dynamic) = dynamic {
            item["dynamic"] = dynamic;
        }
        if let Some(grasp) = &object.grasp {
            if object.kind != "object" || object.dynamic.is_none() {
                return Err(failure(
                    "S1602",
                    "grasp requires an attachable dynamic object",
                    span,
                ));
            }
            item["grasp"] = pose(grasp, span)?;
        }
        objects.push(item);
    }
    for (name, direction) in &value.io {
        if !names.insert(name.as_str())
            || !matches!(direction.as_str(), "digital-in" | "digital-out")
        {
            return Err(failure("S1600", "duplicate or invalid cell IO", span));
        }
    }
    let mount = pose(&value.mount, span)?;
    let mut cell = json!({"id":id,"name":value.name,"robot":value.robot,"mount":mount,
        "objects":objects,"io":value.io.iter().map(|(name, direction)| json!({"name":name,"direction":direction})).collect::<Vec<_>>()});
    if let Some((fixture, exclude)) = &value.mount_surface {
        cell["mount_surface"] = super::mount::surface(&mount, &objects, fixture, exclude, span)?;
    }
    Ok(cell)
}

pub(super) fn check_license(robot: &Value, product: &Value, span: Span) -> Result<(), FrontError> {
    let expression = spdx::Expression::parse(robot["license"].as_str().unwrap_or(""))
        .map_err(|_| failure("S1606", "invalid robot SPDX license", span))?;
    let pseudo_component = json!({"license":{"expression":robot["license"],"link":"static","modified":false,
        "replaceable":false,"network_exposed":false,"grant":null},"isolation":"in-process"});
    let mut rejected = None;
    let allowed = expression.evaluate(|term| {
        match graph::license_term(term, product, &pseudo_component, "static") {
            Ok(()) => true,
            Err((_, message)) => {
                rejected.get_or_insert(message);
                false
            }
        }
    });
    if !allowed {
        return Err(failure(
            "S1606",
            rejected.unwrap_or("robot license rejected by product"),
            span,
        ));
    }
    Ok(())
}
