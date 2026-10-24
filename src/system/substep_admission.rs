//! Private source analysis for an opt-in policy whose native adapter is absent.
//! The predictor/ledger below exist only in tests; no engine, ROS or ABI caller.
use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;
use sha2::{Digest as _, Sha256};

use super::{SystemIr, xml};

mod render_checks;
use render_checks::{check_contacts, check_rendered, geometry_classes};

pub(super) const TARGET: &str = "invar::mujoco_substep_admission_v1";
const DT: f64 = 0.002;
/// Steps per run (60 s at dt 0.002 s).
#[cfg(test)]
const MAX_STEPS: u64 = 30_000;
/// Candidate evaluations per run, charged before prediction and never
/// refunded. Every step charges at least one evaluation, so this cap equals
/// `MAX_STEPS`: a full 30,000-step run admits ZERO rejected candidates, and a
/// run with r rejected candidates reaches at most `30_000 - r` steps before the
/// typed `CandidateBudget` stop. The cap is deliberately not raised; plans
/// that need rejections must budget shorter runs.
#[cfg(test)]
const MAX_RUN_CANDIDATES: u64 = 30_000;
/// The only `<option>` attributes the policy admits (values are checked
/// separately); any other attribute (gravity, cone, impratio, tolerance,
/// iterations, ...) and any child element (`<flag>`) is refused.
const OPTION_ATTRIBUTES: [&str; 4] = ["integrator", "solver", "jacobian", "timestep"];
#[cfg(test)]
const MAX_CONTACTS: usize = 1024;
const MAX_GEOMS: usize = 4096;
// Conservative accounting for each bounded model evidence record, not WCET or
// a native allocator measurement. Native frame/queue sizing remains unqualified.
#[cfg(test)]
const RECORD_BYTES: u64 = 65_536;

#[cfg_attr(
    not(test),
    allow(dead_code, reason = "faults of the sealed test-only state machine")
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    UncheckedSource,
    PolicySelection,
    Parameters,
    Resources,
    UnsupportedLayout,
    ContactVisibility,
    ModelBinding,
    Identity,
    Order,
    Overflow,
    Nonfinite,
    Bounds,
    Contact,
    Capacity,
    Credits,
    NoAdmissibleFiniteGrid,
    Predictor,
    Fuel,
    CandidateBudget,
    PredictionMismatch,
    Lifecycle,
    Incomplete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Digest([u8; 32]);
fn digest(bytes: &[u8]) -> Digest {
    Digest(Sha256::digest(bytes).into())
}
impl Digest {
    fn hex(self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

// Rendered model and asset-set identities qualified by the P5 suite. Both asset
// sets were unchanged from the P5 qualification commit 7ef6a553 until the visual
// strip below (evidence/s1-06-p5-contact-visibility). The Panda model was requalified after
// the 2026-10-13 decisions (principal inertias, source-declared gripper hold
// 253); the prior value was 46a6e125. See evidence/s1-06-p5-residuals. The UR5e
// model was requalified after the 2026-10-14 mount decision: its only change is
// the checked mount exclusion `<contact><exclude>` of table/base and
// table/shoulder_link; the prior value was a2cfae4d. It was requalified again
// after the 2026-10-14 tool-frame decision: its only change is the `invar_tool`
// site orientation (quat "1 0 0 0" to "-1 1 0 0"); the prior value was 7db51cfa.
// Both models were requalified after the 2026-10-15 decision D1 (gravity
// compensation generated on the arm bodies): their only change is `gravcomp="1"`
// on every body of the robot subtree; the prior values were Panda d1dd553c and
// UR5e 635a4b0d. Both models and asset sets were requalified after the 2026-10-23
// strip of visual-only robot geoms and meshes; the Panda home capture was
// regenerated (prior: Panda 1cc33d8a/5808f22e, UR5e 557f4a32/df5cbffd).
const QUALIFIED_PANDA_MODEL_XML: &str =
    "c5e4c42605bd020aff70f43d7f8ef8a1c9924db69a00cf6fb60e119750dbf1a7";
const QUALIFIED_PANDA_ASSETS: &str =
    "96ad2495bebc523c972277150aa9931b276a4def8921753b093fb1fd38810375";
const QUALIFIED_UR5E_MODEL_XML: &str =
    "f56abc0688946104c90b50789ffb480fb12eab588d7e82e21a315d5d016d0496";
const QUALIFIED_UR5E_ASSETS: &str =
    "4603249908b3f2991057aa8c6b9bb251c2cb05d1aa45b58095ea9d74b187a950";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    SevenArmTendon,
    SixArm,
}
impl Layout {
    fn arm_count(self) -> usize {
        match self {
            Self::SevenArmTendon => 7,
            Self::SixArm => 6,
        }
    }
    /// Effective collision role of each robot geom in binder walk order, from
    /// the vendored models with every default resolved: `c` is contype =
    /// conaffinity = 1, `v` is visual-only contype = conaffinity = 0 (none remain).
    fn robot_geoms(self) -> &'static str {
        match self {
            Self::SevenArmTendon => "ccccccccccccccccccccccc",
            Self::SixArm => "ccccccccc",
        }
    }
    /// SHA-256 of the qualified rendered model (`Identity::model_xml`) and of
    /// the vendored asset set (`Identity::assets`), from the reach_place
    /// qualification sources. P5 residuals 1 and 3: any geom margin, gap,
    /// solref or solimp, any equality or tendon change, and any mesh byte or
    /// name change alters one of these and is refused until requalified.
    fn qualified_identity(self) -> [&'static str; 2] {
        match self {
            Self::SevenArmTendon => [QUALIFIED_PANDA_MODEL_XML, QUALIFIED_PANDA_ASSETS],
            Self::SixArm => [QUALIFIED_UR5E_MODEL_XML, QUALIFIED_UR5E_ASSETS],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GeomClass {
    Environment,
    Robot,
    Table,
    Object,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Identity {
    source: Digest,
    model_xml: Digest,
    assets: Digest,
    controls: Digest,
    joints: Digest,
}

#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "profile of private analysis binding; adapter absent"
    )
)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct EngineProfile {
    version: [u32; 3],
    timestep_s: f64,
    integrator: &'static str,
    solver: &'static str,
    jacobian: &'static str,
}

// A checked analysis result, private until a separately reviewed adapter exists.
// Its identity/resource/state fields are consumed by the sealed test model only.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "private analysis binding; no native adapter")
)]
#[derive(Debug)]
struct Binding {
    layout: Layout,
    identity: Identity,
    joint_bounds: Vec<[f64; 2]>,
    speeds: Vec<f64>,
    control_bounds: Vec<[f64; 2]>,
    force_bounds: Vec<[f64; 2]>,
    geoms: Vec<GeomClass>,
    home_q: Vec<f64>,
    home_controls: Vec<f64>,
    memory_bytes: u64,
    profile: EngineProfile,
}

fn array(value: &Value) -> Result<&[Value], Fault> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or(Fault::ModelBinding)
}
fn text(value: &Value) -> Result<&str, Fault> {
    value.as_str().ok_or(Fault::ModelBinding)
}
fn finite(value: &str) -> Result<f64, Fault> {
    value
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .ok_or(Fault::Nonfinite)
}
fn measured(value: &Value, unit: &str) -> Result<f64, Fault> {
    if value["unit"] != unit {
        return Err(Fault::UnsupportedLayout);
    }
    finite(text(&value["value"])?)
}
fn exact(raw: &str, expected: f64) -> bool {
    xml::exact_decimal(raw).as_deref() == Some(expected.to_string().as_str())
}
fn exact_measured(value: &Value, unit: &str, expected: f64) -> Result<bool, Fault> {
    Ok(measured(value, unit)? == expected && exact(text(&value["value"])?, expected))
}
fn numbers(value: &str) -> Result<Vec<f64>, Fault> {
    value.split_whitespace().map(finite).collect()
}
fn pair(value: &str) -> Result<[f64; 2], Fault> {
    numbers(value)?.try_into().map_err(|_| Fault::ModelBinding)
}
fn exact_pair(value: &str, expected: [f64; 2]) -> Result<bool, Fault> {
    let parts = value.split_whitespace().collect::<Vec<_>>();
    let [left, right] = parts.as_slice() else {
        return Err(Fault::ModelBinding);
    };
    Ok(exact(left, expected[0]) && exact(right, expected[1]))
}
fn in_bounds(value: f64, bounds: [f64; 2]) -> bool {
    value.is_finite() && value >= bounds[0] && value <= bounds[1]
}
/// The gripper tendon command held at home, derived from the source: the
/// tendon actuator's declared `hold`, else the vendored `home` key value. It
/// must be an integral command because candidates step down from it.
fn source_hold(robot: &Value, home_controls: &[f64]) -> Result<f64, Fault> {
    let tendon = array(&robot["actuators"])?
        .last()
        .ok_or(Fault::ModelBinding)?;
    let hold = if tendon["hold"].is_null() {
        let key = array(&robot["source_tree"]["children"])?
            .iter()
            .filter(|n| n["tag"] == "keyframe")
            .flat_map(|n| n["children"].as_array().into_iter().flatten())
            .find(|n| n["tag"] == "key" && n["attrs"]["name"] == "home")
            .ok_or(Fault::ModelBinding)?;
        *numbers(text(&key["attrs"]["ctrl"])?)?
            .get(home_controls.len().saturating_sub(1))
            .ok_or(Fault::ModelBinding)?
    } else {
        measured(&tendon["hold"], "one")?
    };
    if hold.fract() != 0.0 || !in_bounds(hold, [0.0, 255.0]) {
        return Err(Fault::UnsupportedLayout);
    }
    Ok(hold)
}

pub(super) fn declared(document: &Value) -> bool {
    document["components"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|c| c["implementation"]["kind"] == "extern" && c["implementation"]["target"] == TARGET)
}

pub(super) fn ensure_export_supported(document: &Value) -> Result<(), &'static str> {
    if declared(document) {
        Err("substep admission native host adapter unavailable; policy is not active")
    } else {
        Ok(())
    }
}

/// The source checker calls this only after ordinary source lowering. Legacy
/// declarations do not enter this policy or change their existing error order.
pub(super) fn validate(ir: &SystemIr) -> Result<(), &'static str> {
    bind(ir).map(|_| ()).map_err(|fault| match fault {
        Fault::UncheckedSource => "substep policy requires ordinary source and passed graph checks",
        Fault::PolicySelection => "substep policy requires exactly one declaration and instance",
        Fault::Parameters => {
            "substep policy requires only fixed dt0.002, max256 and contact restriction1"
        }
        Fault::Resources => "substep policy requires one CPU core and 512 MiB declared memory",
        Fault::UnsupportedLayout => {
            "substep policy joint/control layout or exact bounds unsupported"
        }
        Fault::ContactVisibility => {
            "substep policy collision masks, defaults or contact exclusions unsupported"
        }
        Fault::Identity => {
            "substep policy rendered model or vendored assets differ from the qualified identity"
        }
        _ => "substep policy source-model binding unsupported",
    })
}

fn bind(ir: &SystemIr) -> Result<Option<Binding>, Fault> {
    if !declared(&ir.document) {
        return Ok(None);
    }
    let document = ir.checked_document().map_err(|_| Fault::UncheckedSource)?;
    let policies: Vec<_> = array(&document["components"])?
        .iter()
        .filter(|c| {
            c["implementation"]["kind"] == "extern" && c["implementation"]["target"] == TARGET
        })
        .collect();
    let [component] = policies.as_slice() else {
        return Err(Fault::PolicySelection);
    };
    if array(&document["system"]["instances"])?
        .iter()
        .filter(|i| i["component"] == component["id"])
        .count()
        != 1
    {
        return Err(Fault::PolicySelection);
    }
    let params = component["params"].as_object().ok_or(Fault::Parameters)?;
    if params.len() != 3 {
        return Err(Fault::Parameters);
    }
    for (name, value, unit) in [
        ("physics_dt", "0.002", "s"),
        ("maximum_candidates", "256", "one"),
        ("reject_robot_environment", "1", "one"),
    ] {
        let parameter = &component["params"][name];
        if parameter["value"] != value
            || parameter["unit"] != unit
            || parameter["range"]["min"] != value
            || parameter["range"]["max"] != value
        {
            return Err(Fault::Parameters);
        }
    }
    if component["resources"]["cpu_cores"]["value"] != "1"
        || component["resources"]["cpu_cores"]["unit"] != "one"
        || component["resources"]["memory"]["value"] != "512"
        || component["resources"]["memory"]["unit"] != "MiB"
        || component["resources"]
            .as_object()
            .is_none_or(|r| r.len() != 2)
    {
        return Err(Fault::Resources);
    }
    let ports = array(&component["ports"])?;
    let [port] = ports else {
        return Err(Fault::UnsupportedLayout);
    };
    if port["name"] != "cmd"
        || port["direction"] != "in"
        || port["type"] != "Quantity<i32,one>"
        || port["rate"]["value"] != "100"
        || port["rate"]["unit"] != "Hz"
    {
        return Err(Fault::UnsupportedLayout);
    }
    let product = array(&document["products"])?
        .iter()
        .find(|p| p["id"] == document["system"]["product"])
        .ok_or(Fault::ModelBinding)?;
    let robot = array(&document["robots"])?
        .iter()
        .find(|r| r["id"] == product["robot"])
        .ok_or(Fault::ModelBinding)?;
    let joints: Vec<_> = array(&robot["joints"])?
        .iter()
        .filter(|j| j["kind"] != "fixed")
        .collect();
    let actuators = array(&robot["actuators"])?;
    let (layout, bounds, speeds, efforts) = match (joints.len(), actuators.len()) {
        (9, 8) => (
            Layout::SevenArmTendon,
            vec![
                [-2.8973, 2.8973],
                [-1.7628, 1.7628],
                [-2.8973, 2.8973],
                [-3.0718, -0.0698],
                [-2.8973, 2.8973],
                [-0.0175, 3.7525],
                [-2.8973, 2.8973],
                [0.0, 0.04],
                [0.0, 0.04],
            ],
            vec![2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61, 0.05, 0.05],
            vec![87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 70.0, 70.0],
        ),
        (6, 6) => {
            // Preserve the frozen source's exact decimal limits.
            let joint_limit = finite("6.28319")?;
            let elbow_limit = finite("3.1415")?;
            (
                Layout::SixArm,
                vec![
                    [-joint_limit, joint_limit],
                    [-joint_limit, joint_limit],
                    [-elbow_limit, elbow_limit],
                    [-joint_limit, joint_limit],
                    [-joint_limit, joint_limit],
                    [-joint_limit, joint_limit],
                ],
                vec![finite("3.14159")?; 6],
                vec![150.0, 150.0, 150.0, 28.0, 28.0, 28.0],
            )
        }
        _ => return Err(Fault::UnsupportedLayout),
    };
    let arm_count = layout.arm_count();
    for (index, ((joint, bounds), (speed, effort))) in joints
        .iter()
        .zip(&bounds)
        .zip(speeds.iter().zip(&efforts))
        .enumerate()
    {
        let finger = index >= arm_count;
        let (kind, unit, velocity, force) = if finger {
            ("prismatic", "m", "m/s", "N")
        } else {
            ("revolute", "rad", "rad/s", "N*m")
        };
        if joint["kind"] != kind
            || !exact_measured(&joint["position"]["lower"], unit, bounds[0])?
            || !exact_measured(&joint["position"]["upper"], unit, bounds[1])?
            || !exact_measured(&joint["velocity"], velocity, *speed)?
            || !exact_measured(&joint["effort"], force, *effort)?
        {
            return Err(Fault::UnsupportedLayout);
        }
    }
    // Reuse the ordinary checked model transformations for private analysis;
    // this does not call an export API or publish a policy-free model artifact.
    let mut tree = xml::Element::from_value(&robot["source_tree"]).ok_or(Fault::ModelBinding)?;
    xml::overlay_robot(&mut tree, robot).map_err(|_| Fault::ModelBinding)?;
    xml::apply_tool(&mut tree, robot).map_err(|_| Fault::ModelBinding)?;
    let cells = array(&document["cells"])?;
    let [cell] = cells else {
        return Err(Fault::UnsupportedLayout);
    };
    // Match the ordinary generator: the product selects the robot, while the
    // sole cell supplies fixtures and objects. The frozen UR5e cell retains a
    // Panda annotation; changing that established selection rule is out of scope.
    let objects = array(&cell["objects"])?;
    let dynamic: Vec<_> = objects.iter().filter(|o| !o["dynamic"].is_null()).collect();
    let [object] = dynamic.as_slice() else {
        return Err(Fault::UnsupportedLayout);
    };
    if object["dynamic"]["attachable"] != true || object["grasp"].is_null() {
        return Err(Fault::UnsupportedLayout);
    }
    let roots: Vec<_> = array(&robot["links"])?
        .iter()
        .filter(|l| l["parent"] == "world")
        .collect();
    let [root] = roots.as_slice() else {
        return Err(Fault::ModelBinding);
    };
    let root_name = text(&root["name"])?;
    xml::apply_cells(&mut tree, cells, root_name, text(&robot["id"])?)
        .map_err(|_| Fault::ModelBinding)?;
    let model_xml = tree.render();
    #[cfg(test)]
    let model_xml = tests::forge_rendered(model_xml);
    let model = roxmltree::Document::parse(&model_xml).map_err(|_| Fault::ModelBinding)?;
    check_rendered(&model)?;
    let native_joints: Vec<_> = model
        .descendants()
        .filter(|n| n.has_tag_name("joint") && n.parent().is_some_and(|p| p.has_tag_name("body")))
        .collect();
    if native_joints.len() != joints.len()
        || model
            .descendants()
            .filter(|n| n.has_tag_name("freejoint"))
            .count()
            != 1
    {
        return Err(Fault::ModelBinding);
    }
    let mut names = BTreeSet::new();
    for (source, native) in joints.iter().zip(&native_joints) {
        if native.attribute("name") != Some(text(&source["name"])?)
            || !names.insert(text(&source["name"])?)
        {
            return Err(Fault::ModelBinding);
        }
    }
    let native_actuators: Vec<_> = model
        .descendants()
        .filter(|n| n.parent().is_some_and(|p| p.has_tag_name("actuator")))
        .collect();
    if native_actuators.len() != actuators.len() {
        return Err(Fault::ModelBinding);
    }
    let mut controls = Vec::new();
    let mut forces = Vec::new();
    for (index, (source, native)) in actuators.iter().zip(&native_actuators).enumerate() {
        let tendon = index >= arm_count;
        let unit = if tendon { "one" } else { "rad" };
        let expected = if tendon {
            [0.0, 255.0]
        } else if layout == Layout::SixArm && index != 2 {
            let limit = finite("6.2831")?;
            [-limit, limit]
        } else {
            *bounds.get(index).ok_or(Fault::UnsupportedLayout)?
        };
        let expected_force = if tendon {
            100.0
        } else {
            *efforts.get(index).ok_or(Fault::UnsupportedLayout)?
        };
        let kind = if tendon { "tendon" } else { "joint" };
        if source["mode"] != "position"
            || source["gear"] != "1"
            || source["target_kind"] != kind
            || native.attribute("name") != Some(text(&source["name"])?)
            || native.attribute(kind) != Some(text(&source["target"])?)
            || (!tendon
                && source["target"] != joints.get(index).ok_or(Fault::UnsupportedLayout)?["name"])
            || !exact_measured(&source["control"]["lower"], unit, expected[0])?
            || !exact_measured(&source["control"]["upper"], unit, expected[1])?
            || pair(native.attribute("ctrlrange").ok_or(Fault::ModelBinding)?)? != expected
            || pair(native.attribute("forcerange").ok_or(Fault::ModelBinding)?)?
                != [-expected_force, expected_force]
        {
            return Err(Fault::ModelBinding);
        }
        if !exact_pair(
            native.attribute("ctrlrange").ok_or(Fault::ModelBinding)?,
            expected,
        )? || !exact_pair(
            native.attribute("forcerange").ok_or(Fault::ModelBinding)?,
            [-expected_force, expected_force],
        )? {
            return Err(Fault::ModelBinding);
        }
        controls.push(expected);
        forces.push([-expected_force, expected_force]);
    }
    if layout == Layout::SevenArmTendon {
        let last = actuators.last().ok_or(Fault::ModelBinding)?;
        let tendons: Vec<_> = model
            .descendants()
            .filter(|n| {
                n.has_tag_name("fixed") && n.parent().is_some_and(|p| p.has_tag_name("tendon"))
            })
            .collect();
        let [tendon] = tendons.as_slice() else {
            return Err(Fault::ModelBinding);
        };
        if tendon.attribute("name") != Some(text(&last["target"])?) {
            return Err(Fault::ModelBinding);
        }
        let parts: Vec<_> = tendon.children().filter(|n| n.is_element()).collect();
        if parts.len() != 2 {
            return Err(Fault::ModelBinding);
        }
        for (part, joint) in parts.iter().zip(joints.iter().skip(arm_count)) {
            if !part.has_tag_name("joint")
                || part.attribute("coef").map(finite).transpose()? != Some(0.5)
                || part.attribute("joint") != Some(text(&joint["name"])?)
            {
                return Err(Fault::ModelBinding);
            }
        }
        let left = parts
            .first()
            .and_then(|n| n.attribute("joint"))
            .ok_or(Fault::ModelBinding)?;
        let right = parts
            .last()
            .and_then(|n| n.attribute("joint"))
            .ok_or(Fault::ModelBinding)?;
        let eqs: Vec<_> = model
            .descendants()
            .filter(|n| {
                n.has_tag_name("joint")
                    && n.parent().is_some_and(|p| p.has_tag_name("equality"))
                    && ((n.attribute("joint1") == Some(left)
                        && n.attribute("joint2") == Some(right))
                        || (n.attribute("joint1") == Some(right)
                            && n.attribute("joint2") == Some(left)))
            })
            .collect();
        let [eq] = eqs.as_slice() else {
            return Err(Fault::ModelBinding);
        };
        if eq
            .attribute("polycoef")
            .map(numbers)
            .transpose()?
            .is_some_and(|v| v != [0.0, 1.0, 0.0, 0.0, 0.0])
            || eq
                .attribute("active")
                .is_some_and(|v| !matches!(v, "true" | "1"))
        {
            return Err(Fault::ModelBinding);
        }
    }
    let homes: Vec<_> = model
        .descendants()
        .filter(|n| n.has_tag_name("key") && n.attribute("name") == Some("home"))
        .collect();
    let [home] = homes.as_slice() else {
        return Err(Fault::ModelBinding);
    };
    let home_q = numbers(home.attribute("qpos").ok_or(Fault::ModelBinding)?)?;
    let home_controls = numbers(home.attribute("ctrl").ok_or(Fault::ModelBinding)?)?;
    if home_q.len() != joints.len() + 7
        || home_controls.len() != actuators.len()
        || home_q.iter().zip(&bounds).any(|(q, b)| !in_bounds(*q, *b))
        || home_controls
            .iter()
            .zip(&controls)
            .any(|(q, b)| !in_bounds(*q, *b))
        || (layout == Layout::SevenArmTendon
            && home_controls.last().copied() != Some(source_hold(robot, &home_controls)?))
    {
        return Err(Fault::ModelBinding);
    }
    let mut roles = BTreeMap::new();
    roles.insert(root_name, GeomClass::Robot);
    for object in objects {
        let class = if !object["dynamic"].is_null() {
            GeomClass::Object
        } else if array(&object["geometries"])?
            .iter()
            .any(|g| g["role"] == "collision")
        {
            GeomClass::Table
        } else {
            GeomClass::Environment
        };
        if roles.insert(text(&object["name"])?, class).is_some() {
            return Err(Fault::ModelBinding);
        }
    }
    // P5 residual 2: the robot-table contact gate was qualified only on a cell
    // with exactly one table-class body; any other cell is outside its scope.
    if roles.values().filter(|c| **c == GeomClass::Table).count() != 1 {
        return Err(Fault::ContactVisibility);
    }
    let geoms = geometry_classes(&model, &roles)?;
    // Checked mount exclusions (S1608): only links that rotate about the surface
    // normal, so each excluded pair has a configuration-invariant distance.
    let fixture = cell["mount_surface"]["fixture"].as_str().unwrap_or("");
    let mounted: Vec<(&str, &str)> = super::mount::excluded_links(cell, text(&robot["id"])?)
        .into_iter()
        .map(|link| (fixture, link))
        .collect();
    check_contacts(&model, layout, root_name, &roles, objects, &mounted)?;
    let files = ir
        .assets
        .get(text(&robot["id"])?)
        .ok_or(Fault::ModelBinding)?;
    let asset_hashes: Vec<_> = files
        .iter()
        .map(|(name, bytes)| (name, digest(bytes).0))
        .collect();
    let encode = |value: &Value| serde_json::to_vec(value).map_err(|_| Fault::ModelBinding);
    let identity = Identity {
        source: digest(ir.canonical_json().as_bytes()),
        model_xml: digest(model_xml.as_bytes()),
        assets: digest(&serde_json::to_vec(&asset_hashes).map_err(|_| Fault::ModelBinding)?),
        controls: digest(&encode(&robot["actuators"])?),
        joints: digest(&encode(&robot["joints"])?),
    };
    let [model_xml, assets] = layout.qualified_identity();
    if identity.model_xml.hex() != model_xml || identity.assets.hex() != assets {
        return Err(Fault::Identity);
    }
    Ok(Some(Binding {
        layout,
        identity,
        joint_bounds: bounds,
        speeds,
        control_bounds: controls,
        force_bounds: forces,
        geoms,
        home_q,
        home_controls,
        memory_bytes: 512 * 1024 * 1024,
        profile: EngineProfile {
            version: [3, 12, 0],
            timestep_s: DT,
            integrator: "implicitfast",
            solver: "Newton",
            jacobian: "dense",
        },
    }))
}

/// Complete-state *model* protocol. State owns all its modeled storage and has
/// no interior sharing. This is neither an MjData clone nor native prediction.
#[cfg(test)]
mod model;

#[cfg(test)]
#[path = "../../tests/support/substep_admission_cases.rs"]
mod tests;
