//! Bounded, exact-text XML tree for S1-05 MJCF import and canonicalization.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Component, Path},
};

use num_bigint::BigInt;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

mod cells;
mod import;
mod overlay;
mod synthesize;
#[cfg(test)]
mod tests;
mod verify;

pub(crate) use cells::apply_cells;
pub(crate) use import::{asset_files, import_mjcf, model_file};
pub(crate) use overlay::overlay_robot;
pub(crate) use synthesize::{apply_tool, synthesize_robot};
pub(crate) use verify::verify_robot;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Element {
    pub tag: String,
    pub attrs: BTreeMap<String, String>,
    pub children: Vec<Element>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct XmlError {
    pub code: &'static str,
    pub message: String,
}

fn fail(code: &'static str, message: impl Into<String>) -> XmlError {
    XmlError {
        code,
        message: message.into(),
    }
}

pub(crate) fn exact_decimal(value: &str) -> Option<String> {
    let (negative, raw) = value
        .strip_prefix('-')
        .map_or((false, value), |tail| (true, tail));
    let (mantissa, exponent) = raw
        .split_once(['e', 'E'])
        .map_or((raw, 0), |(mantissa, exponent)| {
            (mantissa, exponent.parse::<i32>().unwrap_or(i32::MAX))
        });
    if exponent.unsigned_abs() > 128 {
        return None;
    }
    let (whole, fraction) = mantissa
        .split_once('.')
        .map_or((mantissa, ""), |parts| parts);
    if whole.is_empty()
        || !whole.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
        || whole.len() + fraction.len() > 128
    {
        return None;
    }
    let mut digits = format!("{whole}{fraction}");
    let point = whole.len() as i32 + exponent;
    if point < 0 {
        digits = format!("0.{}{digits}", "0".repeat((-point) as usize));
    } else if point == 0 {
        digits = format!("0.{digits}");
    } else if point as usize >= digits.len() {
        digits.push_str(&"0".repeat(point as usize - digits.len()));
    } else {
        digits.insert(point as usize, '.');
    }
    let (whole, fraction) = digits
        .split_once('.')
        .map_or((digits.as_str(), ""), |parts| parts);
    let whole = whole.trim_start_matches('0');
    let whole = if whole.is_empty() { "0" } else { whole };
    let fraction = fraction.trim_end_matches('0');
    let sign = if negative && (whole != "0" || !fraction.is_empty()) {
        "-"
    } else {
        ""
    };
    Some(if fraction.is_empty() {
        format!("{sign}{whole}")
    } else {
        format!("{sign}{whole}.{fraction}")
    })
}

fn numeric_attribute(name: &str) -> bool {
    matches!(
        name,
        "pos"
            | "quat"
            | "axis"
            | "range"
            | "mass"
            | "fullinertia"
            | "diaginertia"
            | "size"
            | "gear"
            | "ctrlrange"
            | "forcerange"
            | "gainprm"
            | "biasprm"
            | "qpos"
            | "ctrl"
            | "rgba"
            | "rgb1"
            | "rgb2"
            | "markrgb"
            | "damping"
            | "armature"
            | "frictionloss"
            | "solimp"
            | "solref"
            | "coef"
            | "fromto"
            | "euler"
            | "xyaxes"
            | "zaxis"
            | "dir"
            | "diffuse"
            | "ambient"
            | "specular"
            | "shininess"
            | "reflectance"
            | "texrepeat"
            | "width"
            | "height"
            | "azimuth"
            | "elevation"
            | "fovy"
            | "focal"
            | "principal"
            | "sensorsize"
            | "ipd"
    )
}

fn parse_node(node: roxmltree::Node<'_, '_>, count: &mut usize) -> Result<Element, XmlError> {
    *count += 1;
    if *count > 4096 {
        return Err(fail("S1607", "MJCF element count limit exceeded"));
    }
    let mut attrs = BTreeMap::new();
    for attribute in node.attributes() {
        let value = if numeric_attribute(attribute.name()) {
            attribute
                .value()
                .split_whitespace()
                .map(|part| {
                    exact_decimal(part).ok_or_else(|| {
                        fail(
                            "S1607",
                            format!("invalid exact decimal in {}", attribute.name()),
                        )
                    })
                })
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
            children.push(parse_node(child, count)?);
        } else if child.is_text() && !child.text().unwrap_or("").trim().is_empty() {
            return Err(fail("S1607", "MJCF element text is unsupported"));
        }
    }
    Ok(Element {
        tag: node.tag_name().name().to_owned(),
        attrs,
        children,
    })
}

fn relative(path: &str) -> bool {
    !path.is_empty()
        && path.len() <= 512
        && !path
            .chars()
            .any(|ch| ch == '"' || ch == '\\' || ch.is_control())
        && Path::new(path)
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
}

fn read_file(
    path: &Path,
    root: &Path,
    seen: &mut BTreeSet<std::path::PathBuf>,
    total: &mut usize,
    depth: usize,
) -> Result<Element, XmlError> {
    if depth > 16 {
        return Err(fail("S1607", "MJCF include depth limit exceeded"));
    }
    let path = path
        .canonicalize()
        .map_err(|_| fail("S1605", "MJCF source file is missing"))?;
    if !path.starts_with(root) || !seen.insert(path.clone()) {
        return Err(fail(
            "S1605",
            "MJCF include escapes model root or repeats a file",
        ));
    }
    let bytes = std::fs::read(&path).map_err(|_| fail("S1605", "cannot read MJCF source"))?;
    *total += bytes.len();
    if *total > 1_048_576 {
        return Err(fail("S1607", "MJCF XML byte limit exceeded"));
    }
    let source = std::str::from_utf8(&bytes).map_err(|_| fail("S1607", "MJCF must be UTF-8"))?;
    let document = roxmltree::Document::parse(source)
        .map_err(|error| fail("S1607", format!("invalid MJCF XML: {error}")))?;
    let mut count = 0;
    let mut tree = parse_node(document.root_element(), &mut count)?;
    if tree.tag != "mujoco" {
        return Err(fail("S1607", "MJCF root must be mujoco"));
    }
    expand_includes(&mut tree, root, seen, total, depth)?;
    Ok(tree)
}

fn expand_includes(
    node: &mut Element,
    root: &Path,
    seen: &mut BTreeSet<std::path::PathBuf>,
    total: &mut usize,
    depth: usize,
) -> Result<(), XmlError> {
    let mut expanded = Vec::new();
    for mut child in std::mem::take(&mut node.children) {
        if child.tag == "include" {
            let file = child
                .attrs
                .get("file")
                .ok_or_else(|| fail("S1607", "MJCF include has no file"))?;
            if !relative(file) {
                return Err(fail("S1605", "MJCF include path is not relative"));
            }
            let included = read_file(&root.join(file), root, seen, total, depth + 1)?;
            expanded.extend(included.children);
        } else {
            expand_includes(&mut child, root, seen, total, depth)?;
            expanded.push(child);
        }
    }
    node.children = expanded;
    Ok(())
}

type Defaults = BTreeMap<String, BTreeMap<String, BTreeMap<String, String>>>;

fn collect_defaults(
    node: &Element,
    inherited: &BTreeMap<String, BTreeMap<String, String>>,
    classes: &mut Defaults,
) {
    let class = node.attrs.get("class").cloned().unwrap_or_default();
    let mut values = inherited.clone();
    for child in &node.children {
        if child.tag != "default" {
            let target = values.entry(child.tag.clone()).or_default();
            target.extend(
                child
                    .attrs
                    .iter()
                    .filter(|(key, _)| key.as_str() != "class")
                    .map(|(key, value)| (key.clone(), value.clone())),
            );
        }
    }
    classes.insert(class, values.clone());
    for child in node.children.iter().filter(|child| child.tag == "default") {
        collect_defaults(child, &values, classes);
    }
}

fn defaultable(tag: &str, parent: &str) -> bool {
    matches!(
        (parent, tag),
        ("body" | "worldbody", "geom" | "joint" | "site" | "camera")
            | ("actuator", "general" | "motor" | "position" | "velocity")
            | ("asset", "material" | "mesh" | "texture")
    )
}

fn apply_defaults(node: &mut Element, parent: &str, inherited_class: &str, classes: &Defaults) {
    let chosen = node
        .attrs
        .get("class")
        .map_or(inherited_class, String::as_str)
        .to_owned();
    let child_class = node
        .attrs
        .get("childclass")
        .map_or(inherited_class, String::as_str)
        .to_owned();
    if defaultable(&node.tag, parent)
        && let Some(defaults) = classes.get(&chosen).and_then(|class| class.get(&node.tag))
    {
        for (key, value) in defaults {
            node.attrs
                .entry(key.clone())
                .or_insert_with(|| value.clone());
        }
    }
    node.attrs.remove("class");
    node.attrs.remove("childclass");
    node.children.retain(|child| child.tag != "default");
    for child in &mut node.children {
        let next = if node.tag == "body" {
            child_class.as_str()
        } else {
            inherited_class
        };
        apply_defaults(child, &node.tag, next, classes);
    }
}

pub(crate) fn load_mjcf(path: &Path) -> Result<Element, XmlError> {
    let root = path
        .parent()
        .ok_or_else(|| fail("S1605", "MJCF source has no directory"))?
        .canonicalize()
        .map_err(|_| fail("S1605", "MJCF source directory is missing"))?;
    let mut tree = read_file(path, &root, &mut BTreeSet::new(), &mut 0, 0)?;
    let mut classes = Defaults::new();
    for child in tree.children.iter().filter(|child| child.tag == "default") {
        collect_defaults(child, &BTreeMap::new(), &mut classes);
    }
    apply_defaults(&mut tree, "", "", &classes);
    Ok(tree)
}

pub(crate) fn validate_supported(tree: &mut Element) -> Result<Vec<String>, XmlError> {
    fn visit(node: &mut Element, parent: &str, warnings: &mut Vec<String>) -> Result<(), XmlError> {
        let known = matches!(
            node.tag.as_str(),
            "mujoco"
                | "compiler"
                | "option"
                | "visual"
                | "headlight"
                | "rgba"
                | "global"
                | "asset"
                | "texture"
                | "material"
                | "mesh"
                | "worldbody"
                | "light"
                | "geom"
                | "body"
                | "inertial"
                | "joint"
                | "site"
                | "camera"
                | "actuator"
                | "general"
                | "motor"
                | "position"
                | "velocity"
                | "sensor"
                | "jointpos"
                | "jointvel"
                | "force"
                | "torque"
                | "tendon"
                | "fixed"
                | "spatial"
                | "equality"
                | "weld"
                | "connect"
                | "contact"
                | "exclude"
                | "pair"
                | "keyframe"
                | "key"
        );
        if !known {
            return Err(fail(
                "S1607",
                format!(
                    "unsupported MJCF element <{}> under <{parent}> affects the model",
                    node.tag
                ),
            ));
        }
        let mut retained = Vec::new();
        for mut child in std::mem::take(&mut node.children) {
            let child_known = matches!(
                child.tag.as_str(),
                "headlight"
                    | "rgba"
                    | "global"
                    | "texture"
                    | "material"
                    | "mesh"
                    | "light"
                    | "geom"
                    | "body"
                    | "inertial"
                    | "joint"
                    | "site"
                    | "camera"
                    | "general"
                    | "motor"
                    | "position"
                    | "velocity"
                    | "jointpos"
                    | "jointvel"
                    | "force"
                    | "torque"
                    | "fixed"
                    | "spatial"
                    | "weld"
                    | "connect"
                    | "exclude"
                    | "pair"
                    | "key"
                    | "compiler"
                    | "option"
                    | "visual"
                    | "asset"
                    | "worldbody"
                    | "actuator"
                    | "sensor"
                    | "tendon"
                    | "equality"
                    | "contact"
                    | "keyframe"
            );
            if !child_known && node.tag == "visual" {
                warnings.push(format!(
                    "unsupported visual MJCF element <{}> was omitted",
                    child.tag
                ));
                continue;
            }
            visit(&mut child, &node.tag, warnings)?;
            retained.push(child);
        }
        node.children = retained;
        Ok(())
    }
    let mut warnings = Vec::new();
    visit(tree, "", &mut warnings)?;
    Ok(warnings)
}

fn attr<'a>(node: &'a Element, name: &str) -> Option<&'a str> {
    node.attrs.get(name).map(String::as_str)
}

fn required<'a>(node: &'a Element, name: &str) -> Result<&'a str, XmlError> {
    attr(node, name).ok_or_else(|| fail("S1607", format!("{} needs {name}", node.tag)))
}

fn tuple(value: &str, count: usize) -> Result<String, XmlError> {
    let parts: Vec<_> = value.split_whitespace().collect();
    if parts.len() != count {
        return Err(fail("S1607", "MJCF vector has wrong length"));
    }
    Ok(format!("({})", parts.join(", ")))
}

fn pose(node: &Element, pos_name: &str) -> Result<String, XmlError> {
    if node.attrs.contains_key("euler")
        || node.attrs.contains_key("axisangle")
        || node.attrs.contains_key("xyaxes")
        || node.attrs.contains_key("zaxis")
    {
        return Err(fail("S1607", "MJCF alternate rotation affects kinematics"));
    }
    let pos = tuple(attr(node, pos_name).unwrap_or("0 0 0"), 3)?;
    let quat = tuple(attr(node, "quat").unwrap_or("1 0 0 0"), 4)?;
    let pos = pos
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split(", ")
        .map(|part| format!("{part} m"))
        .collect::<Vec<_>>()
        .join(", ");
    Ok(format!("xyz ({pos}) quat {quat}"))
}

fn identifier(raw: &str) -> Result<&str, XmlError> {
    let mut bytes = raw.bytes();
    if !bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(fail(
            "S1600",
            format!("MJCF name cannot be an Invar identifier: {raw}"),
        ));
    }
    Ok(raw)
}

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

impl Element {
    pub(crate) fn to_value(&self) -> Value {
        json!({"tag":self.tag,"attrs":self.attrs,"children":self.children.iter().map(Element::to_value).collect::<Vec<_>>()})
    }

    pub(crate) fn from_value(value: &Value) -> Option<Self> {
        Some(Self {
            tag: value["tag"].as_str()?.to_owned(),
            attrs: value["attrs"]
                .as_object()?
                .iter()
                .map(|(key, value)| Some((key.clone(), value.as_str()?.to_owned())))
                .collect::<Option<_>>()?,
            children: value["children"]
                .as_array()?
                .iter()
                .map(Self::from_value)
                .collect::<Option<_>>()?,
        })
    }

    pub(crate) fn render(&self) -> String {
        fn escape(value: &str) -> String {
            value
                .replace('&', "&amp;")
                .replace('"', "&quot;")
                .replace('<', "&lt;")
        }
        if self.tag == "#text" {
            return escape(self.attrs.get("value").map_or("", String::as_str));
        }
        let mut out = format!("<{}", self.tag);
        for (key, value) in &self.attrs {
            out.push_str(&format!(" {key}=\"{}\"", escape(value)));
        }
        if self.children.is_empty() {
            out.push_str("/>");
        } else {
            out.push('>');
            for child in &self.children {
                out.push_str(&child.render());
            }
            out.push_str(&format!("</{}>", self.tag));
        }
        out
    }
}
