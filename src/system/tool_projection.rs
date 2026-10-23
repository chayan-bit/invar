//! Source-derived adapter between the pinned reference tool and declared cell tool.
//! This changes neither the foreign ABI nor the cell's tool/weld definitions.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::xml::Element;

/// Rigid transform, metres and a unit xyzw quaternion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub translation_m: [f64; 3],
    pub quaternion_xyzw: [f64; 4],
}
impl Pose {
    pub const IDENTITY: Self = Self {
        translation_m: [0.0; 3],
        quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
    };
    pub fn valid(self) -> bool {
        self.translation_m
            .iter()
            .chain(&self.quaternion_xyzw)
            .all(|x| x.is_finite())
            && (self
                .quaternion_xyzw
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt()
                - 1.0)
                .abs()
                <= 1e-12
    }
    /// Checked composition: base_from_parent * parent_from_child.
    pub fn compose(self, rhs: Self) -> Result<Self, &'static str> {
        if !self.valid() || !rhs.valid() {
            return Err("invalid rigid transform");
        }
        let [x, y, z, w] = self.quaternion_xyzw;
        let [a, b, c, d] = rhs.quaternion_xyzw;
        let rotated = self.rotate(rhs.translation_m);
        let out = Self {
            translation_m: std::array::from_fn(|i| self.translation_m[i] + rotated[i]),
            quaternion_xyzw: [
                w * a + x * d + y * c - z * b,
                w * b - x * c + y * d + z * a,
                w * c + x * b - y * a + z * d,
                w * d - x * a - y * b - z * c,
            ],
        };
        if !out.valid() {
            return Err("nonfinite or nonunit composed transform");
        }
        Ok(out)
    }
    fn rotate(self, v: [f64; 3]) -> [f64; 3] {
        let [x, y, z, w] = self.quaternion_xyzw;
        let t = [
            2.0 * (y * v[2] - z * v[1]),
            2.0 * (z * v[0] - x * v[2]),
            2.0 * (x * v[1] - y * v[0]),
        ];
        [
            v[0] + w * t[0] + y * t[2] - z * t[1],
            v[1] + w * t[1] + z * t[0] - x * t[2],
            v[2] + w * t[2] + x * t[1] - y * t[0],
        ]
    }
    pub fn inverse(self) -> Result<Self, &'static str> {
        if !self.valid() {
            return Err("invalid rigid transform");
        }
        let [x, y, z, w] = self.quaternion_xyzw;
        let mut out = Self {
            translation_m: [0.0; 3],
            quaternion_xyzw: [-x, -y, -z, w],
        };
        out.translation_m = out.rotate(self.translation_m.map(|v| -v));
        if !out.valid() {
            return Err("nonfinite inverse transform");
        }
        Ok(out)
    }
}

/// Explicit provenance and direction of a checked cell adapter.
#[derive(Clone, Debug)]
pub struct Projection {
    pub schema: String,
    pub reference_frame: String,
    pub planner_reference_frame: String,
    pub anchor_body: String,
    pub source_scene_sha256: String,
    pub reference_model_sha256: String,
    pub reference_implementation_sha256: String,
    pub cell_tool: Value,
    pub reference_from_cell: Pose,
    pub reference_base_frame: String,
    pub source_base_from_reference_base: Pose,
}

/// (vendored cell model, pinned FK model) pairs with identical kinematics.
/// `issue_s1_06_panda_inertia` proves the vendored Panda differs from the upstream copy
/// the FK goldens pin only by the principal-inertia rewrite of `<inertial>` elements and
/// the removal of visual-only geoms; the vendored UR5e differs from its pinned original
/// only by that removal.
const SAME_KINEMATICS: [(&str, &str); 2] = [
    (
        "93a8d8e137ae43a33e7499ff627c1f633fb27d94f58ea7545b97c8f65027db1d",
        "96ad67da03710f17f798c9478fd9e9efdf24a3bf8359f05e456dd9fb158ea273",
    ),
    (
        "43b30279ed0f3e590de8a948504fb28582258a36014f5577971b8e0744644fdd",
        "ffd0a7b336d3c2d27c35a15cc0d4aaa95196769aca190ac8d6522d3318c345a0",
    ),
];

/// Whether the FK model bound to a planner record has the projection's reference
/// kinematics: the same bytes, or a pinned pair proven to differ only in inertia.
pub fn same_kinematics(reference_model_sha256: &str, fk_model_sha256: &str) -> bool {
    reference_model_sha256 == fk_model_sha256
        || SAME_KINEMATICS.contains(&(reference_model_sha256, fk_model_sha256))
}

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn values<const N: usize>(value: &str) -> Result<[f64; N], &'static str> {
    let a = value
        .split_whitespace()
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| "invalid source pose")?;
    let out: [f64; N] = a.try_into().map_err(|_| "invalid source pose width")?;
    if out.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite source pose");
    }
    Ok(out)
}
pub(super) fn source_pose(p: [f64; 3], wxyz: [f64; 4]) -> Result<Pose, &'static str> {
    let norm = wxyz.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err("invalid source quaternion");
    }
    let [w, x, y, z] = wxyz.map(|x| x / norm);
    let pose = Pose {
        translation_m: p,
        quaternion_xyzw: [x, y, z, w],
    };
    if !pose.valid() {
        return Err("invalid source transform");
    }
    Ok(pose)
}
fn named<'a>(tree: &'a Element, tag: &str, name: &str, found: &mut Vec<&'a Element>) {
    if tree.tag == tag && tree.attrs.get("name").is_some_and(|v| v == name) {
        found.push(tree);
    }
    for child in &tree.children {
        named(child, tag, name, found);
    }
}

/// Derive from a checked robot, the pinned scene convention and its actual
/// reference body/site. Unknown models are refused instead of guessing by DOF.
pub fn derive(robot: &Value) -> Result<Projection, &'static str> {
    derive_version(robot, true)
}

fn derive_version(robot: &Value, source_base: bool) -> Result<Projection, &'static str> {
    let panda_scene = include_bytes!(
        "../../examples/cells/reach_place/assets/menagerie/franka_emika_panda/scene.xml"
    );
    let ur_scene = include_bytes!(
        "../../examples/cells/reach_place/assets/menagerie/universal_robots_ur5e/scene.xml"
    );
    let (body, reference, planner, model): (&str, &str, &str, &[u8]) = if robot["asset"]["sha256"]
        == hash(panda_scene)
    {
        (
            "hand",
            "hand",
            "panda_hand",
            include_bytes!(
                "../../examples/cells/reach_place/assets/menagerie/franka_emika_panda/panda.xml"
            ),
        )
    } else if robot["asset"]["sha256"] == hash(ur_scene) {
        (
            "wrist_3_link",
            "attachment_site",
            "tool0",
            include_bytes!(
                "../../examples/cells/reach_place/assets/menagerie/universal_robots_ur5e/ur5e.xml"
            ),
        )
    } else {
        return Err("no checked reference-tool convention for selected model");
    };
    if robot["tool"]["link"] != body {
        return Err("cell tool and pinned reference must share the checked anchor body");
    }
    let tree = Element::from_value(&robot["source_tree"]).ok_or("missing source tree")?;
    if reference != body {
        let mut sites = Vec::new();
        named(&tree, "site", reference, &mut sites);
        if sites.len() != 1 {
            return Err("missing or duplicate reference site");
        }
    }
    let mut anchors = Vec::new();
    named(&tree, "body", body, &mut anchors);
    if anchors.len() != 1 {
        return Err("missing or duplicate reference anchor");
    }
    let body_from_reference = if reference == body {
        Pose::IDENTITY
    } else {
        let sites: Vec<_> = anchors[0]
            .children
            .iter()
            .filter(|s| s.tag == "site" && s.attrs.get("name").is_some_and(|n| n == reference))
            .collect();
        if sites.len() != 1 {
            return Err("reference site must be unique and directly on anchor body");
        }
        let site = sites[0];
        if site
            .attrs
            .keys()
            .any(|k| matches!(k.as_str(), "euler" | "axisangle" | "xyaxes" | "zaxis"))
        {
            return Err("unsupported reference-site orientation");
        }
        source_pose(
            values(site.attrs.get("pos").map_or("0 0 0", String::as_str))?,
            values(site.attrs.get("quat").map_or("1 0 0 0", String::as_str))?,
        )?
    };
    // Compare the actual source site with the pinned reference XML, not its name alone.
    let pinned =
        roxmltree::Document::parse(std::str::from_utf8(model).map_err(|_| "invalid pinned XML")?)
            .map_err(|_| "invalid pinned XML")?;
    let expected = if reference == body {
        Pose::IDENTITY
    } else {
        let sites: Vec<_> = pinned
            .descendants()
            .filter(|n| n.has_tag_name("site") && n.attribute("name") == Some(reference))
            .collect();
        if sites.len() != 1 || sites[0].parent().and_then(|n| n.attribute("name")) != Some(body) {
            return Err("pinned reference anchor changed");
        }
        source_pose(
            values(sites[0].attribute("pos").unwrap_or("0 0 0"))?,
            values(sites[0].attribute("quat").unwrap_or("1 0 0 0"))?,
        )?
    };
    if body_from_reference != expected {
        return Err("checked source reference site differs from pinned convention");
    }
    let xyz = robot["tool"]["pose"]["xyz"]
        .as_array()
        .ok_or("missing cell tool position")?;
    if xyz.len() != 3 || xyz.iter().any(|v| v["unit"] != "m") {
        return Err("invalid cell tool position units");
    }
    let p: [f64; 3] = xyz
        .iter()
        .map(|v| {
            v["value"]
                .as_str()
                .ok_or("invalid cell position")
                .and_then(|s| s.parse().map_err(|_| "invalid cell position"))
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "invalid cell position width")?;
    let quat = robot["tool"]["pose"]["quat"]
        .as_array()
        .ok_or("missing cell quaternion")?;
    let q: [f64; 4] = quat
        .iter()
        .map(|v| {
            v.as_str()
                .ok_or("invalid cell quaternion")
                .and_then(|s| s.parse().map_err(|_| "invalid cell quaternion"))
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "invalid cell quaternion width")?;
    let reference_from_cell = body_from_reference.inverse()?.compose(source_pose(p, q)?)?;
    let root_name = if reference == "hand" { "link0" } else { "base" };
    let base = if source_base {
        let mut roots = Vec::new();
        named(&tree, "body", root_name, &mut roots);
        if roots.len() != 1
            || roots[0]
                .children
                .iter()
                .any(|c| matches!(c.tag.as_str(), "joint" | "freejoint"))
        {
            return Err("missing, duplicate or moving reference root");
        }
        let direct_root = tree
            .children
            .iter()
            .filter(|c| c.tag == "worldbody")
            .flat_map(|c| &c.children)
            .any(|c| std::ptr::eq(c, roots[0]));
        if !direct_root {
            return Err("reference root must be directly in source worldbody");
        }
        let root = roots[0];
        if root
            .attrs
            .keys()
            .any(|k| matches!(k.as_str(), "euler" | "axisangle" | "xyaxes" | "zaxis"))
        {
            return Err("unsupported reference-root orientation");
        }
        let actual = source_pose(
            values(root.attrs.get("pos").map_or("0 0 0", String::as_str))?,
            values(root.attrs.get("quat").map_or("1 0 0 0", String::as_str))?,
        )?;
        let roots: Vec<_> = pinned
            .descendants()
            .filter(|n| n.has_tag_name("body") && n.attribute("name") == Some(root_name))
            .collect();
        if roots.len() != 1
            || !roots[0]
                .parent()
                .is_some_and(|n| n.has_tag_name("worldbody"))
        {
            return Err("pinned reference root changed");
        }
        let expected = source_pose(
            values(roots[0].attribute("pos").unwrap_or("0 0 0"))?,
            values(roots[0].attribute("quat").unwrap_or("1 0 0 0"))?,
        )?;
        if actual != expected {
            return Err("source reference root differs from pinned convention");
        }
        actual
    } else {
        Pose::IDENTITY
    };
    Ok(Projection {
        schema: if source_base {
            "invar-tool-projection-v2"
        } else {
            "invar-tool-projection-v1"
        }
        .into(),
        reference_frame: reference.into(),
        planner_reference_frame: planner.into(),
        anchor_body: body.into(),
        source_scene_sha256: robot["asset"]["sha256"]
            .as_str()
            .ok_or("missing source hash")?
            .into(),
        reference_model_sha256: hash(model),
        reference_implementation_sha256: hash(include_bytes!(
            "../../examples/cells/reach_place/components/pinocchio_fk/abi_reference.cpp"
        )),
        cell_tool: robot["tool"].clone(),
        reference_from_cell,
        reference_base_frame: root_name.into(),
        source_base_from_reference_base: base,
    })
}

/// Derive a selected product's adapter without modifying its checked IR.
pub fn selected(ir: &super::SystemIr) -> Result<Projection, &'static str> {
    let d = ir.checked_document()?;
    let product = d["products"]
        .as_array()
        .and_then(|p| p.iter().find(|p| p["id"] == d["system"]["product"]))
        .ok_or("missing selected product")?;
    let robot = d["robots"]
        .as_array()
        .and_then(|r| r.iter().find(|r| r["id"] == product["robot"]))
        .ok_or("missing selected robot")?;
    derive(robot)
}

impl Projection {
    /// Explicit compatibility for records that used the reference-base v1 convention.
    pub fn legacy(mut self) -> Self {
        self.schema = "invar-tool-projection-v1".into();
        self.source_base_from_reference_base = Pose::IDENTITY;
        self
    }
    pub fn value(&self) -> Value {
        let mut value = json!({"schema":self.schema,"reference_frame":self.reference_frame,"planner_reference_frame":self.planner_reference_frame,"anchor_body":self.anchor_body,"source_scene_sha256":self.source_scene_sha256,"reference_model_sha256":self.reference_model_sha256,"reference_implementation_sha256":self.reference_implementation_sha256,"cell_tool":self.cell_tool,"reference_from_cell":{"translation_m":self.reference_from_cell.translation_m,"quaternion_xyzw":self.reference_from_cell.quaternion_xyzw}});
        if self.schema == "invar-tool-projection-v2" {
            value["reference_base_frame"] = json!(self.reference_base_frame);
            value["source_base_from_reference_base"] = json!({"translation_m":self.source_base_from_reference_base.translation_m,"quaternion_xyzw":self.source_base_from_reference_base.quaternion_xyzw});
        }
        value
    }
    pub fn fk_to_source(&self, raw: Pose) -> Result<Pose, &'static str> {
        self.source_base_from_reference_base
            .compose(raw)?
            .compose(self.reference_from_cell)
    }
}
pub(crate) fn metadata(projection: &Projection) -> String {
    serde_json::to_string_pretty(&projection.value()).expect("finite checked projection")
}

/// Structural record validation replays the direction and pinned source
/// convention. Execution provenance still requires the hash-bound harness.
pub fn verify_value(value: &Value) -> Result<(), &'static str> {
    let scene = value["source_scene_sha256"]
        .as_str()
        .ok_or("missing source scene hash")?;
    let panda = scene
        == hash(include_bytes!(
            "../../examples/cells/reach_place/assets/menagerie/franka_emika_panda/scene.xml"
        ));
    let model: &[u8] = if panda {
        include_bytes!(
            "../../examples/cells/reach_place/assets/menagerie/franka_emika_panda/panda.xml"
        )
    } else {
        include_bytes!(
            "../../examples/cells/reach_place/assets/menagerie/universal_robots_ur5e/ur5e.xml"
        )
    };
    let body = if panda { "hand" } else { "wrist_3_link" };
    let mut children = Vec::new();
    if !panda {
        let doc = roxmltree::Document::parse(
            std::str::from_utf8(model).map_err(|_| "invalid pinned model")?,
        )
        .map_err(|_| "invalid pinned model")?;
        let site = doc
            .descendants()
            .find(|n| n.has_tag_name("site") && n.attribute("name") == Some("attachment_site"))
            .ok_or("missing pinned site")?;
        children.push(json!({"tag":"site","attrs":{"name":"attachment_site","pos":site.attribute("pos").ok_or("missing pinned position")?,"quat":site.attribute("quat").ok_or("missing pinned orientation")?},"children":[]}));
    }
    let source_base = match value["schema"].as_str() {
        Some("invar-tool-projection-v1") => false,
        Some("invar-tool-projection-v2") => true,
        _ => return Err("unsupported tool projection version"),
    };
    let anchor = json!({"tag":"body","attrs":{"name":body},"children":children});
    let tree = if source_base {
        let doc = roxmltree::Document::parse(
            std::str::from_utf8(model).map_err(|_| "invalid pinned model")?,
        )
        .map_err(|_| "invalid pinned model")?;
        let root_name = if panda { "link0" } else { "base" };
        let root = doc
            .descendants()
            .find(|n| n.has_tag_name("body") && n.attribute("name") == Some(root_name))
            .ok_or("missing pinned root")?;
        json!({"tag":"mujoco","attrs":{},"children":[{"tag":"worldbody","attrs":{},"children":[{"tag":"body","attrs":{"name":root_name,"pos":root.attribute("pos").unwrap_or("0 0 0"),"quat":root.attribute("quat").unwrap_or("1 0 0 0")},"children":[anchor]}]}]})
    } else {
        anchor
    };
    let robot = json!({"asset":{"sha256":scene},"tool":value["cell_tool"],"source_tree":tree});
    if derive_version(&robot, source_base)?.value() != *value {
        return Err("malformed, inverted or stale tool projection");
    }
    Ok(())
}

pub(crate) fn header(projection: Option<&Projection>) -> String {
    let pose = projection.map_or(Pose::IDENTITY, |p| p.reference_from_cell);
    let base = projection.map_or(Pose::IDENTITY, |p| p.source_base_from_reference_base);
    let model = projection.map_or("", |p| {
        if p.reference_frame == "hand" {
            include_str!(
                "../../examples/cells/reach_place/assets/menagerie/franka_emika_panda/panda.xml"
            )
        } else {
            include_str!(
                "../../examples/cells/reach_place/assets/menagerie/universal_robots_ur5e/ur5e.xml"
            )
        }
    });
    include_str!("ros2_templates/tool_projection.hpp.in")
        .replace(
            "@BASE_TRANSLATION@",
            &base
                .translation_m
                .iter()
                .map(|v| format!("{v:.17e}"))
                .collect::<Vec<_>>()
                .join(","),
        )
        .replace(
            "@BASE_QUATERNION@",
            &base
                .quaternion_xyzw
                .iter()
                .map(|v| format!("{v:.17e}"))
                .collect::<Vec<_>>()
                .join(","),
        )
        .replace("@REFERENCE_MODEL@", model)
        .replace(
            "@TRANSLATION@",
            &pose
                .translation_m
                .iter()
                .map(|v| format!("{v:.17e}"))
                .collect::<Vec<_>>()
                .join(","),
        )
        .replace(
            "@QUATERNION@",
            &pose
                .quaternion_xyzw
                .iter()
                .map(|v| format!("{v:.17e}"))
                .collect::<Vec<_>>()
                .join(","),
        )
}
