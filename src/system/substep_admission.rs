//! Private source analysis for an opt-in policy whose native adapter is absent.
//! The predictor/ledger below exist only in tests; no engine, ROS or ABI caller.
use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;
use sha2::{Digest as _, Sha256};

use super::{SystemIr, xml};

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
mod model {
    use super::*;
    use std::collections::VecDeque;

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(super) struct Rad(pub(super) f64);
    #[derive(Clone, Debug, PartialEq)]
    pub(super) enum Arm {
        Seven([Rad; 7]),
        Six([Rad; 6]),
    }
    impl Arm {
        fn values(&self) -> &[Rad] {
            match self {
                Self::Seven(v) => v,
                Self::Six(v) => v,
            }
        }
        fn same(&self, other: &Self) -> bool {
            std::mem::discriminant(self) == std::mem::discriminant(other)
                && self
                    .values()
                    .iter()
                    .zip(other.values())
                    .all(|(a, b)| a.0.to_bits() == b.0.to_bits())
        }
    }
    #[derive(Clone, Debug, PartialEq)]
    pub(super) enum Controls {
        Panda { arm: [Rad; 7], tendon: u8 },
        Ur5e { arm: [Rad; 6] },
    }
    impl Controls {
        pub(super) fn raw(&self) -> Vec<f64> {
            match self {
                Self::Panda { arm, tendon } => arm
                    .iter()
                    .map(|v| v.0)
                    .chain([f64::from(*tendon)])
                    .collect(),
                Self::Ur5e { arm } => arm.iter().map(|v| v.0).collect(),
            }
        }
        pub(super) fn from_raw(layout: Layout, values: &[f64]) -> Result<Self, Fault> {
            match layout {
                Layout::SevenArmTendon => {
                    let [a, b, c, d, e, f, g, t] = values else {
                        return Err(Fault::Capacity);
                    };
                    if !t.is_finite() || !in_bounds(*t, [0.0, 255.0]) || t.fract() != 0.0 {
                        return Err(Fault::Bounds);
                    }
                    Ok(Self::Panda {
                        arm: [*a, *b, *c, *d, *e, *f, *g].map(Rad),
                        tendon: *t as u8,
                    })
                }
                Layout::SixArm => {
                    let [a, b, c, d, e, f] = values else {
                        return Err(Fault::Capacity);
                    };
                    Ok(Self::Ur5e {
                        arm: [*a, *b, *c, *d, *e, *f].map(Rad),
                    })
                }
            }
        }
        pub(super) fn arm(&self) -> Arm {
            match self {
                Self::Panda { arm, .. } => Arm::Seven(*arm),
                Self::Ur5e { arm } => Arm::Six(*arm),
            }
        }
        pub(super) fn tendon(&self) -> Option<u8> {
            match self {
                Self::Panda { tendon, .. } => Some(*tendon),
                Self::Ur5e { .. } => None,
            }
        }
        fn same(&self, other: &Self) -> bool {
            self.arm().same(&other.arm()) && self.tendon() == other.tendon()
        }
    }
    #[derive(Clone, Debug, PartialEq)]
    pub(super) struct State {
        pub(super) step: u64,
        pub(super) time: f64,
        pub(super) q: Vec<f64>,
        pub(super) dq: Vec<f64>,
        pub(super) forces: Vec<f64>,
        pub(super) contacts: Vec<(usize, usize)>,
        pub(super) applied: Controls,
        pub(super) attached: bool,
        // Owned, artificial hidden model state for clone/leak tests. A captured
        // native integration digest is only a fixture witness, not reconstruction.
        pub(super) latent: [u64; 8],
        pub(super) witness: Digest,
    }
    impl State {
        pub(super) fn home(binding: &Binding) -> Self {
            Self {
                step: 0,
                time: 0.0,
                q: binding
                    .home_q
                    .iter()
                    .take(binding.joint_bounds.len())
                    .copied()
                    .collect(),
                dq: vec![0.0; binding.joint_bounds.len()],
                forces: vec![0.0; binding.force_bounds.len()],
                contacts: Vec::new(),
                applied: Controls::from_raw(binding.layout, &binding.home_controls)
                    .expect("checked home controls"),
                attached: false,
                latent: [0; 8],
                witness: Digest([0; 32]),
            }
        }
        fn same(&self, other: &Self) -> bool {
            let bits = |a: &[f64], b: &[f64]| {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            };
            self.step == other.step
                && self.time.to_bits() == other.time.to_bits()
                && bits(&self.q, &other.q)
                && bits(&self.dq, &other.dq)
                && bits(&self.forces, &other.forces)
                && self.contacts == other.contacts
                && self.applied.same(&other.applied)
                && self.attached == other.attached
                && self.latent == other.latent
                && self.witness == other.witness
        }
    }
    impl Binding {
        /// Gripper candidates step down from the source-derived home hold, so
        /// the policy never commands more than the source declares.
        pub(super) fn tendon_start(&self) -> Option<u8> {
            match self.layout {
                Layout::SevenArmTendon => self.home_controls.last().map(|t| *t as u8),
                Layout::SixArm => None,
            }
        }
        pub(super) fn candidates(&self) -> u32 {
            self.tendon_start().map_or(1, |start| u32::from(start) + 1)
        }
        fn dimensions(&self, state: &State) -> Result<(), Fault> {
            let controls = state.applied.raw();
            if state.q.len() != self.joint_bounds.len()
                || state.dq.len() != self.speeds.len()
                || state.forces.len() != self.force_bounds.len()
                || controls.len() != self.control_bounds.len()
                || state.contacts.len() > MAX_CONTACTS
            {
                return Err(Fault::Capacity);
            }
            Ok(())
        }
        pub(super) fn gate(&self, state: &State) -> Result<(), Fault> {
            self.dimensions(state)?;
            let controls = state.applied.raw();
            if !state.time.is_finite()
                || state.time < 0.0
                || state
                    .q
                    .iter()
                    .chain(&state.dq)
                    .chain(&state.forces)
                    .chain(&controls)
                    .any(|v| !v.is_finite())
            {
                return Err(Fault::Nonfinite);
            }
            if state
                .q
                .iter()
                .zip(&self.joint_bounds)
                .any(|(v, b)| !in_bounds(*v, *b))
                || state.dq.iter().zip(&self.speeds).any(|(v, s)| v.abs() > *s)
                || controls
                    .iter()
                    .zip(&self.control_bounds)
                    .any(|(v, b)| !in_bounds(*v, *b))
                || state
                    .forces
                    .iter()
                    .zip(&self.force_bounds)
                    .any(|(v, b)| !in_bounds(*v, *b))
            {
                return Err(Fault::Bounds);
            }
            for (a, b) in &state.contacts {
                let a = self.geoms.get(*a).ok_or(Fault::Capacity)?;
                let b = self.geoms.get(*b).ok_or(Fault::Capacity)?;
                if (*a == GeomClass::Robot
                    && matches!(
                        b,
                        GeomClass::Robot | GeomClass::Table | GeomClass::Environment
                    ))
                    || (*b == GeomClass::Robot
                        && matches!(
                            a,
                            GeomClass::Robot | GeomClass::Table | GeomClass::Environment
                        ))
                {
                    return Err(Fault::Contact);
                }
            }
            Ok(())
        }
        fn nominal(&self, arm: &Arm) -> Result<(), Fault> {
            if !matches!(
                (self.layout, arm),
                (Layout::SevenArmTendon, Arm::Seven(_)) | (Layout::SixArm, Arm::Six(_))
            ) {
                return Err(Fault::Capacity);
            }
            if arm.values().iter().any(|v| !v.0.is_finite()) {
                return Err(Fault::Nonfinite);
            }
            if arm
                .values()
                .iter()
                .zip(&self.control_bounds)
                .any(|(v, b)| !in_bounds(v.0, *b))
            {
                return Err(Fault::Bounds);
            }
            Ok(())
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct RunId(pub(super) [u8; 16]);
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct Equality {
        pub(super) sequence: u64,
        pub(super) attached: bool,
    }
    #[derive(Clone, Debug)]
    pub(super) struct Request {
        pub(super) run: RunId,
        pub(super) identity: Identity,
        pub(super) next_step: u64,
        // Supplied test identity; current PlannerV1/JTC transport has no such ID.
        pub(super) arm_sequence: u64,
        pub(super) arm: Arm,
        pub(super) equality: Equality,
    }
    impl Request {
        fn same(&self, other: &Self) -> bool {
            self.run == other.run
                && self.identity == other.identity
                && self.next_step == other.next_step
                && self.arm_sequence == other.arm_sequence
                && self.arm.same(&other.arm)
                && self.equality == other.equality
        }
    }
    #[derive(Debug)]
    pub(super) struct Ticket {
        run: RunId,
        identity: Identity,
        pub(super) request: Request,
        pub(super) controls: Controls,
        pub(super) predicted: State,
        rank: u32,
    }
    #[derive(Clone, Debug)]
    pub(super) enum Kind {
        Candidate {
            step: u64,
            evaluation: u64,
            rank: u32,
            controls: Controls,
            prospective: State,
            predicted: Box<State>,
            result: Result<(), Fault>,
        },
        Command {
            step: u64,
            request: Request,
            controls: Controls,
        },
        Snapshot(State),
        Halt {
            step: u64,
            reason: Fault,
            evaluations: u64,
        },
        Final {
            step: u64,
            witness: Digest,
            outcome: Result<(), Fault>,
        },
    }
    #[derive(Clone, Debug)]
    pub(super) struct Record {
        pub(super) sequence: u64,
        pub(super) kind: Kind,
    }
    #[derive(Debug, PartialEq, Eq)]
    pub(super) enum Phase {
        Active,
        Prepared,
        Frozen,
        Complete,
        Halted(Fault),
    }
    #[derive(Debug)]
    pub(super) struct Completion {
        pub(super) commands: u64,
        pub(super) snapshots: u64,
        pub(super) completed_steps: u64,
        pub(super) final_sequence: u64,
    }
    pub(super) struct Run<'a> {
        pub(super) binding: &'a Binding,
        pub(super) id: RunId,
        pub(super) live: State,
        pub(super) phase: Phase,
        required: u64,
        capacity: u32,
        pub(super) available: u32,
        pub(super) held: u32,
        pub(super) queue: VecDeque<Record>,
        pub(super) last_record: u64,
        last_ack: u64,
        pub(super) commands: u64,
        pub(super) snapshots: u64,
        last_request: Option<Request>,
        pub(super) candidate_evaluations: u64,
        pending: Option<(u64, u32, Controls, State, Request)>,
        frozen_fault: Option<Fault>,
    }
    impl<'a> Run<'a> {
        pub(super) fn new(
            binding: &'a Binding,
            id: RunId,
            initial: State,
            required: u64,
            capacity: u32,
        ) -> Result<Self, Fault> {
            if id.0 == [0; 16] {
                return Err(Fault::Identity);
            }
            if required == 0
                || required > MAX_STEPS
                || initial.step != 0
                || initial.time.to_bits() != 0.0f64.to_bits()
            {
                return Err(Fault::Order);
            }
            binding.gate(&initial)?;
            let bytes = u64::from(capacity)
                .checked_add(3)
                .and_then(|v| v.checked_mul(RECORD_BYTES))
                .ok_or(Fault::Overflow)?;
            if capacity == 0 || bytes > binding.memory_bytes {
                return Err(Fault::Capacity);
            }
            let mut queue = VecDeque::new();
            queue
                .try_reserve_exact(usize::try_from(capacity).map_err(|_| Fault::Capacity)?)
                .map_err(|_| Fault::Capacity)?;
            let mut run = Self {
                binding,
                id,
                live: initial.clone(),
                phase: Phase::Active,
                required,
                capacity,
                available: capacity,
                held: 0,
                queue,
                last_record: 0,
                last_ack: 0,
                commands: 0,
                snapshots: 1,
                candidate_evaluations: 0,
                last_request: None,
                pending: None,
                frozen_fault: None,
            };
            run.reserve(1)?;
            run.emit(Kind::Snapshot(initial))?;
            Ok(run)
        }
        fn reserve(&mut self, count: u32) -> Result<(), Fault> {
            self.last_record
                .checked_add(u64::from(count))
                .ok_or(Fault::Overflow)?;
            if self.held != 0 || count > self.available {
                return Err(Fault::Credits);
            }
            self.available = self.available.checked_sub(count).ok_or(Fault::Credits)?;
            self.held = count;
            Ok(())
        }
        fn emit(&mut self, kind: Kind) -> Result<(), Fault> {
            if self.held == 0
                || self.queue.len()
                    >= usize::try_from(self.capacity).map_err(|_| Fault::Capacity)?
            {
                return Err(Fault::Credits);
            }
            let next = self.last_record.checked_add(1).ok_or(Fault::Overflow)?;
            self.queue.push_back(Record {
                sequence: next,
                kind,
            });
            self.last_record = next;
            self.held -= 1;
            Ok(())
        }
        fn release(&mut self) -> Result<(), Fault> {
            self.available = self
                .available
                .checked_add(self.held)
                .filter(|v| *v <= self.capacity)
                .ok_or(Fault::Credits)?;
            self.held = 0;
            Ok(())
        }
        fn halt<T>(&mut self, fault: Fault) -> Result<T, Fault> {
            self.pending = None;
            if self.held > 0 {
                self.emit(Kind::Halt {
                    step: self.live.step,
                    reason: fault,
                    evaluations: self.candidate_evaluations,
                })?;
            }
            self.release()?;
            self.phase = Phase::Halted(fault);
            Err(fault)
        }
        pub(super) fn prepare(
            &mut self,
            request: Request,
            predictor: &mut Predictor,
        ) -> Result<Ticket, Fault> {
            if self.phase != Phase::Active {
                return self.halt(Fault::Lifecycle);
            }
            if request.run != self.id || request.identity != self.binding.identity {
                return self.halt(Fault::Identity);
            }
            let next = match self.live.step.checked_add(1) {
                Some(v) => v,
                None => return self.halt(Fault::Overflow),
            };
            if request.next_step != next || next > self.required {
                return self.halt(Fault::Order);
            }
            if let Some(last) = &self.last_request
                && (request.arm_sequence < last.arm_sequence
                    || (request.arm_sequence == last.arm_sequence && !request.arm.same(&last.arm))
                    || request.equality.sequence < last.equality.sequence
                    || (request.equality.sequence == last.equality.sequence
                        && request.equality.attached != last.equality.attached))
            {
                return self.halt(Fault::Order);
            }
            if let Err(fault) = self.binding.nominal(&request.arm) {
                return self.halt(fault);
            }
            if let Err(fault) = self.binding.gate(&self.live) {
                return self.halt(fault);
            }
            if self.candidate_evaluations >= MAX_RUN_CANDIDATES {
                return self.halt(Fault::CandidateBudget);
            }
            // Worst-case candidate records plus command/state/halt. Unused slots
            // are released exactly; accepted commands hold three through commit.
            if let Err(fault) = self.reserve(self.binding.candidates() + 3) {
                return self.halt(fault);
            }
            let mut prospective = self.live.clone();
            prospective.attached = request.equality.attached;
            for rank in 0..self.binding.candidates() {
                if self.candidate_evaluations >= MAX_RUN_CANDIDATES {
                    return self.halt(Fault::CandidateBudget);
                }
                self.candidate_evaluations = self
                    .candidate_evaluations
                    .checked_add(1)
                    .ok_or(Fault::Overflow)?;
                let controls = match request.arm {
                    Arm::Seven(arm) => Controls::Panda {
                        arm,
                        tendon: self
                            .binding
                            .tendon_start()
                            .and_then(|start| start.checked_sub(u8::try_from(rank).ok()?))
                            .ok_or(Fault::Overflow)?,
                    },
                    Arm::Six(arm) => Controls::Ur5e { arm },
                };
                let predicted = match predictor.predict(prospective.clone(), &controls) {
                    Ok(v) => v,
                    Err(fault) => return self.halt(fault),
                };
                if let Err(fault) = self.binding.dimensions(&predicted) {
                    return self.halt(fault);
                }
                let verdict = if predicted.step != next
                    || predicted.time.to_bits() != (self.live.time + DT).to_bits()
                    || !predicted.applied.same(&controls)
                    || predicted.attached != prospective.attached
                {
                    Err(Fault::PredictionMismatch)
                } else {
                    self.binding.gate(&predicted)
                };
                self.emit(Kind::Candidate {
                    step: next,
                    evaluation: self.candidate_evaluations,
                    rank,
                    controls: controls.clone(),
                    prospective: prospective.clone(),
                    predicted: Box::new(predicted.clone()),
                    result: verdict,
                })?;
                match verdict {
                    Ok(()) => {
                        let unused = self.held.checked_sub(3).ok_or(Fault::Credits)?;
                        self.held -= unused;
                        self.available =
                            self.available.checked_add(unused).ok_or(Fault::Credits)?;
                        self.pending = Some((
                            next,
                            rank,
                            controls.clone(),
                            predicted.clone(),
                            request.clone(),
                        ));
                        self.phase = Phase::Prepared;
                        return Ok(Ticket {
                            run: self.id,
                            identity: self.binding.identity,
                            request,
                            controls,
                            predicted,
                            rank,
                        });
                    }
                    Err(Fault::Bounds | Fault::Contact) => (),
                    Err(fault) => return self.halt(fault),
                }
            }
            self.halt(Fault::NoAdmissibleFiniteGrid)
        }
        pub(super) fn commit(&mut self, ticket: Ticket, actual: State) -> Result<(), Fault> {
            let Some((step, rank, controls, predicted, frozen_request)) = self.pending.take()
            else {
                return self.halt(Fault::Lifecycle);
            };
            if self.phase != Phase::Prepared
                || ticket.run != self.id
                || ticket.identity != self.binding.identity
                || !ticket.request.same(&frozen_request)
                || ticket.request.next_step != step
                || ticket.rank != rank
                || !controls.same(&ticket.controls)
                || !predicted.same(&ticket.predicted)
            {
                return self.halt(Fault::Identity);
            }
            if actual.step != step {
                return self.halt(Fault::Order);
            }
            if let Err(fault) = self.binding.dimensions(&actual) {
                return self.halt(fault);
            }
            // Commit is a supplied actual-model observation, never native execution.
            // Preserve the completed observation on mismatch; do not undo a step.
            self.commands = self.commands.checked_add(1).ok_or(Fault::Overflow)?;
            self.snapshots = self.snapshots.checked_add(1).ok_or(Fault::Overflow)?;
            self.emit(Kind::Command {
                step,
                request: ticket.request.clone(),
                controls: actual.applied.clone(),
            })?;
            self.emit(Kind::Snapshot(actual.clone()))?;
            self.live = actual;
            if !self.live.same(&predicted) {
                return self.halt(Fault::PredictionMismatch);
            }
            if let Err(fault) = self.binding.gate(&self.live) {
                return self.halt(fault);
            }
            self.release()?;
            self.last_request = Some(frozen_request);
            self.phase = Phase::Active;
            Ok(())
        }
        pub(super) fn acknowledge(&mut self, last: u64) -> Result<(), Fault> {
            if last <= self.last_ack || last > self.last_record {
                return Err(Fault::Order);
            }
            let count = last.checked_sub(self.last_ack).ok_or(Fault::Overflow)?;
            if count > u64::try_from(self.queue.len()).map_err(|_| Fault::Overflow)? {
                return Err(Fault::Order);
            }
            let count = usize::try_from(count).map_err(|_| Fault::Overflow)?;
            if self
                .queue
                .iter()
                .take(count)
                .zip(self.last_ack + 1..=last)
                .any(|(r, expected)| r.sequence != expected)
            {
                return Err(Fault::Order);
            }
            let available = self
                .available
                .checked_add(u32::try_from(count).map_err(|_| Fault::Overflow)?)
                .filter(|v| *v <= self.capacity)
                .ok_or(Fault::Credits)?;
            for _ in 0..count {
                self.queue.pop_front();
            }
            self.available = available;
            self.last_ack = last;
            Ok(())
        }
        pub(super) fn quiesce(&mut self, final_capture: State) -> Result<(), Fault> {
            let reason = match self.phase {
                Phase::Active => {
                    if self.live.step == self.required {
                        None
                    } else {
                        Some(Fault::Incomplete)
                    }
                }
                Phase::Halted(fault) => Some(fault),
                _ => return self.halt(Fault::Lifecycle),
            };
            if !final_capture.same(&self.live) {
                return self.halt(Fault::PredictionMismatch);
            }
            if let Err(fault) = self.reserve(1) {
                return self.halt(fault);
            }
            self.emit(Kind::Final {
                step: self.live.step,
                witness: final_capture.witness,
                outcome: reason.map_or(Ok(()), Err),
            })?;
            self.frozen_fault = reason;
            self.phase = Phase::Frozen;
            Ok(())
        }
        pub(super) fn finish(&mut self) -> Result<Completion, Fault> {
            if self.phase != Phase::Frozen {
                return Err(Fault::Lifecycle);
            }
            if !self.queue.is_empty() || self.held != 0 || self.available != self.capacity {
                return Err(Fault::Credits);
            }
            if let Some(fault) = self.frozen_fault {
                return Err(fault);
            }
            if self.live.step != self.required
                || self.commands != self.required
                || self.snapshots != self.required.checked_add(1).ok_or(Fault::Overflow)?
            {
                return Err(Fault::Incomplete);
            }
            self.phase = Phase::Complete;
            Ok(Completion {
                commands: self.commands,
                snapshots: self.snapshots,
                completed_steps: self.live.step,
                final_sequence: self.last_ack,
            })
        }
        pub(super) fn credits_consistent(&self) -> bool {
            u64::from(self.available) + u64::from(self.held) + self.queue.len() as u64
                == u64::from(self.capacity)
        }
    }

    mod sealed {
        pub trait Sealed {}
    }
    trait TestPrediction: sealed::Sealed {
        fn predict(&mut self, state: State, controls: &Controls) -> Result<State, Fault>;
    }
    #[derive(Clone, Debug)]
    pub(super) struct Recorded {
        pub(super) step: u64,
        pub(super) controls: Controls,
        pub(super) state: State,
    }
    pub(super) enum Behavior {
        Pass,
        RejectAbove(u8),
        Exhaust,
        Recorded(Vec<Recorded>),
        Nonfinite,
        WrongTransition,
    }
    pub(super) struct Predictor {
        pub(super) behavior: Behavior,
        pub(super) fuel: u32,
        pub(super) inputs: Vec<State>,
        pub(super) calls: u64,
    }
    impl Predictor {
        pub(super) fn new(behavior: Behavior, fuel: u32) -> Self {
            Self {
                behavior,
                fuel,
                inputs: Vec::new(),
                calls: 0,
            }
        }
    }
    impl sealed::Sealed for Predictor {}
    impl TestPrediction for Predictor {
        fn predict(&mut self, mut state: State, controls: &Controls) -> Result<State, Fault> {
            if self.fuel == 0 {
                return Err(Fault::Fuel);
            }
            if self.inputs.first().is_some_and(|s| s.step != state.step) {
                self.inputs.clear();
            }
            if self.inputs.len() >= 256 {
                return Err(Fault::Capacity);
            }
            self.calls = self.calls.checked_add(1).ok_or(Fault::Overflow)?;
            self.fuel -= 1;
            self.inputs.push(state.clone());
            let next = state.step.checked_add(1).ok_or(Fault::Overflow)?;
            if let Behavior::Recorded(records) = &self.behavior {
                return records
                    .iter()
                    .find(|r| r.step == next && r.controls.same(controls))
                    .map(|r| r.state.clone())
                    .ok_or(Fault::Predictor);
            }
            state.step = next;
            state.time += DT;
            state.applied = controls.clone();
            state.latent[0] = state.latent[0].checked_add(1).ok_or(Fault::Overflow)?;
            match self.behavior {
                Behavior::RejectAbove(limit) if controls.tendon().is_some_and(|t| t > limit) => {
                    *state.q.last_mut().ok_or(Fault::Capacity)? = 0.041;
                    state.latent[1] = 999;
                }
                Behavior::Exhaust => {
                    *state.q.last_mut().ok_or(Fault::Capacity)? = 100.0;
                    state.latent[1] = 999;
                }
                Behavior::Nonfinite => {
                    *state.q.first_mut().ok_or(Fault::Capacity)? = f64::NAN;
                }
                Behavior::WrongTransition => state.attached = !state.attached,
                _ => (),
            }
            Ok(state)
        }
    }
}

/// Rendered-model checks independent of the source layout: unsupported
/// elements, including `<frame>`/`<attach>` nesting that the geom mapping does
/// not traverse; exactly one `<option>` with only approved attributes and no
/// children (so a `<flag>` cannot disable contact or other physics).
fn check_rendered(model: &roxmltree::Document<'_>) -> Result<(), Fault> {
    if model.descendants().any(|n| {
        matches!(
            n.tag_name().name(),
            "plugin" | "flex" | "composite" | "replicate" | "frame" | "attach"
        )
    }) {
        return Err(Fault::UnsupportedLayout);
    }
    let options = model
        .descendants()
        .filter(|n| n.has_tag_name("option"))
        .collect::<Vec<_>>();
    let [option] = options.as_slice() else {
        return Err(Fault::UnsupportedLayout);
    };
    if option
        .attributes()
        .any(|a| !OPTION_ATTRIBUTES.contains(&a.name()))
        || option.children().any(|n| n.is_element())
        || option.attribute("integrator") != Some("implicitfast")
        || option.attribute("solver").is_some_and(|s| s != "Newton")
        || option
            .attribute("jacobian")
            .is_some_and(|s| !matches!(s, "dense" | "auto"))
        || option
            .attribute("timestep")
            .map(finite)
            .transpose()?
            .is_some_and(|dt| dt != DT)
    {
        return Err(Fault::UnsupportedLayout);
    }
    Ok(())
}

fn geometry_classes(
    model: &roxmltree::Document<'_>,
    roles: &BTreeMap<&str, GeomClass>,
) -> Result<Vec<GeomClass>, Fault> {
    fn body(
        node: roxmltree::Node<'_, '_>,
        class: GeomClass,
        depth: usize,
        out: &mut Vec<GeomClass>,
    ) -> Result<(), Fault> {
        if depth > 64 {
            return Err(Fault::Capacity);
        }
        for geom in node.children().filter(|n| n.has_tag_name("geom")) {
            let _ = geom;
            if out.len() == MAX_GEOMS {
                return Err(Fault::Capacity);
            }
            out.push(class);
        }
        for child in node.children().filter(|n| n.has_tag_name("body")) {
            body(child, class, depth + 1, out)?;
        }
        Ok(())
    }
    let worlds: Vec<_> = model
        .root_element()
        .children()
        .filter(|n| n.has_tag_name("worldbody"))
        .collect();
    let mut out = Vec::new();
    // MuJoCo assigns world-owned geometry before body-owned geometry, including
    // when canonical input retains more than one worldbody section.
    for world in &worlds {
        for _ in world.children().filter(|n| n.has_tag_name("geom")) {
            out.push(GeomClass::Environment);
        }
    }
    for world in worlds {
        for root in world.children().filter(|n| n.has_tag_name("body")) {
            let name = root.attribute("name").ok_or(Fault::ModelBinding)?;
            body(
                root,
                *roles.get(name).ok_or(Fault::ModelBinding)?,
                0,
                &mut out,
            )?;
        }
    }
    if out.is_empty() || out.len() > MAX_GEOMS {
        return Err(Fault::Capacity);
    }
    // Every geom under a worldbody must have been classified: geometry the
    // walk above does not reach would shift native geom indices silently.
    let all = model
        .root_element()
        .children()
        .filter(|n| n.has_tag_name("worldbody"))
        .flat_map(|w| w.descendants().filter(|n| n.has_tag_name("geom")))
        .count();
    if all != out.len() {
        return Err(Fault::UnsupportedLayout);
    }
    Ok(out)
}

/// Contact visibility (review3 P5). Import inlines every MJCF default, so the
/// rendered model must carry no `<default>`, `class` or `childclass`; a geom's
/// effective `contype`/`conaffinity` is then its attribute or MuJoCo's built-in
/// value of one. Each geom is visual-only (0, 0) or full collision (1, 1).
/// Robot geoms must match the vendored layout, cell geoms their checked role,
/// and world geoms must collide. `<contact>` may only exclude a robot body and its robot
/// parent, a pair that MuJoCo's default parent filter already removes, or a checked
/// cell mount exclusion between the table-class mount fixture and a robot body mounted
/// to it. A mounted body only rotates about the surface normal, so its distance to the
/// table never changes and the exclusion cannot hide a motion-dependent contact.
/// Anything else is refused.
fn check_contacts(
    model: &roxmltree::Document<'_>,
    layout: Layout,
    root_name: &str,
    roles: &BTreeMap<&str, GeomClass>,
    objects: &[Value],
    mounted: &[(&str, &str)],
) -> Result<(), Fault> {
    fn collides(geom: roxmltree::Node<'_, '_>) -> Result<bool, Fault> {
        let mask = |name| match geom.attribute(name) {
            None | Some("1") => Ok(true),
            Some("0") => Ok(false),
            Some(_) => Err(Fault::ContactVisibility),
        };
        match (mask("contype")?, mask("conaffinity")?) {
            (true, true) => Ok(true),
            (false, false) => Ok(false),
            _ => Err(Fault::ContactVisibility),
        }
    }
    fn walk<'a, 'input>(
        body: roxmltree::Node<'a, 'input>,
        depth: usize,
        geoms: &mut Vec<bool>,
        bodies: &mut Vec<roxmltree::Node<'a, 'input>>,
    ) -> Result<(), Fault> {
        if depth > 64 {
            return Err(Fault::Capacity);
        }
        bodies.push(body);
        for geom in body.children().filter(|n| n.has_tag_name("geom")) {
            geoms.push(collides(geom)?);
        }
        for child in body.children().filter(|n| n.has_tag_name("body")) {
            walk(child, depth + 1, geoms, bodies)?;
        }
        Ok(())
    }
    if model.descendants().any(|n| {
        n.has_tag_name("default") || n.has_attribute("class") || n.has_attribute("childclass")
    }) {
        return Err(Fault::ContactVisibility);
    }
    // Named body -> (class, named parent body), for exclusion checks.
    let mut owners = BTreeMap::new();
    for world in model
        .root_element()
        .children()
        .filter(|n| n.has_tag_name("worldbody"))
    {
        for geom in world.children().filter(|n| n.has_tag_name("geom")) {
            if !collides(geom)? {
                return Err(Fault::ContactVisibility);
            }
        }
        for root in world.children().filter(|n| n.has_tag_name("body")) {
            let name = root.attribute("name").ok_or(Fault::ModelBinding)?;
            let class = *roles.get(name).ok_or(Fault::ModelBinding)?;
            let (mut geoms, mut bodies) = (Vec::new(), Vec::new());
            walk(root, 0, &mut geoms, &mut bodies)?;
            let expected = if name == root_name {
                layout.robot_geoms().bytes().map(|b| b == b'c').collect()
            } else {
                let object = objects
                    .iter()
                    .find(|o| o["name"] == name)
                    .ok_or(Fault::ModelBinding)?;
                array(&object["geometries"])?
                    .iter()
                    .map(|g| match g["role"].as_str() {
                        Some("collision") => Ok(true),
                        Some("visual") => Ok(false),
                        _ => Err(Fault::ContactVisibility),
                    })
                    .collect::<Result<Vec<_>, _>>()?
            };
            if geoms != expected {
                return Err(Fault::ContactVisibility);
            }
            for body in bodies {
                let Some(body_name) = body.attribute("name") else {
                    continue;
                };
                let parent = body
                    .parent()
                    .filter(|p| p.has_tag_name("body"))
                    .and_then(|p| p.attribute("name"));
                if owners.insert(body_name, (class, parent)).is_some() {
                    return Err(Fault::ContactVisibility);
                }
            }
        }
    }
    let robot_child_of = |child: &str, parent: &str| {
        owners.get(child) == Some(&(GeomClass::Robot, Some(parent)))
            && owners
                .get(parent)
                .is_some_and(|(class, _)| *class == GeomClass::Robot)
    };
    let mount_pair = |fixture: &str, link: &str| {
        mounted.contains(&(fixture, link))
            && owners
                .get(fixture)
                .is_some_and(|(class, _)| *class == GeomClass::Table)
            && owners
                .get(link)
                .is_some_and(|(class, _)| *class == GeomClass::Robot)
    };
    for node in model
        .descendants()
        .filter(|n| matches!(n.tag_name().name(), "contact" | "exclude" | "pair"))
    {
        let allowed = match node.tag_name().name() {
            "contact" => {
                node.parent() == Some(model.root_element())
                    && node.attributes().next().is_none()
                    && node
                        .children()
                        .all(|c| !c.is_element() || c.has_tag_name("exclude"))
            }
            "exclude" => {
                node.parent().is_some_and(|p| p.has_tag_name("contact"))
                    && node
                        .attributes()
                        .all(|a| matches!(a.name(), "name" | "body1" | "body2"))
                    && match (node.attribute("body1"), node.attribute("body2")) {
                        (Some(a), Some(b)) => {
                            robot_child_of(a, b)
                                || robot_child_of(b, a)
                                || mount_pair(a, b)
                                || mount_pair(b, a)
                        }
                        _ => false,
                    }
            }
            _ => false,
        };
        if !allowed {
            return Err(Fault::ContactVisibility);
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/support/substep_admission_cases.rs"]
mod tests;
