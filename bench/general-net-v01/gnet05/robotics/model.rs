//! Robot geometry and inertia extracted from the pinned Menagerie MJCF files
//! (S1-05/S1-06 models). Extraction is binary64 and exact to the source text;
//! constants enter Invar source rounded once to binary32.
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

pub const PANDA_MJCF: &str =
    "tests/fixtures/foreign-components/models/franka_emika_panda/panda.xml";
pub const PANDA_SHA256: &str = "96ad67da03710f17f798c9478fd9e9efdf24a3bf8359f05e456dd9fb158ea273";
pub const UR5E_MJCF: &str = "examples/cells/assets/menagerie/universal_robots_ur5e/ur5e.xml";
pub const UR5E_SHA256: &str = "ffd0a7b336d3c2d27c35a15cc0d4aaa95196769aca190ac8d6522d3318c345a0";
/// Inactive Panda finger slide position (contract `fixed_inactive_joints`).
pub const PANDA_FINGER_M: f64 = 0.04;

pub type Mat = [[f64; 3]; 3];
pub type Vec3 = [f64; 3];

/// A rigid transform: rotation (row-major) and translation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub r: Mat,
    pub p: Vec3,
}

/// Hinge axis of an active joint in its body frame: a positive coordinate axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Body {
    pub name: String,
    /// Index of the parent in `Model::bodies`; `None` for the base.
    pub parent: Option<usize>,
    /// Fixed transform from the parent frame to this body's joint frame,
    /// including any inactive slide displacement.
    pub fixed: Frame,
    /// Active hinge: (joint name, axis, lower, upper, armature).
    pub hinge: Option<(String, Axis, f64, f64, f64)>,
    pub mass: f64,
    /// Centre of mass in the body frame.
    pub com: Vec3,
    /// Rotational inertia about the centre of mass, body axes.
    pub inertia: Mat,
}

/// A primitive collision geometry in a body frame (capsule or cylinder).
#[derive(Clone, Debug, PartialEq)]
pub struct Geom {
    pub body: usize,
    pub capsule: bool,
    pub radius: f64,
    pub half_length: f64,
    pub center: Vec3,
    /// Unit axis of the primitive in the body frame.
    pub axis: Vec3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub name: &'static str,
    pub file: &'static str,
    pub sha256: &'static str,
    /// Base first; every parent precedes its children.
    pub bodies: Vec<Body>,
    /// Tool frame: body index and fixed offset.
    pub tool: (usize, Frame),
    pub tool_name: &'static str,
    /// Original collision primitives (UR5e only; Panda collision is meshes and pads).
    pub geoms: Vec<Geom>,
}

impl Model {
    /// Bodies with an active hinge, in source order.
    pub fn active(&self) -> Vec<usize> {
        (0..self.bodies.len())
            .filter(|&b| self.bodies[b].hinge.is_some())
            .collect()
    }
    /// Whether `ancestor` is `body` or one of its ancestors.
    pub fn moves(&self, ancestor: usize, body: usize) -> bool {
        let mut b = Some(body);
        while let Some(i) = b {
            if i == ancestor {
                return true;
            }
            b = self.bodies[i].parent;
        }
        false
    }
}

fn nums(text: &str) -> Vec<f64> {
    text.split_whitespace()
        .map(|t| t.parse::<f64>().unwrap_or_else(|_| panic!("number {t}")))
        .collect()
}

fn quat(q: &[f64]) -> Mat {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let (w, x, y, z) = (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

pub fn mul(a: &Mat, b: &Mat) -> Mat {
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    m
}

pub fn apply(a: &Mat, v: &Vec3) -> Vec3 {
    [0, 1, 2].map(|i| (0..3).map(|k| a[i][k] * v[k]).sum())
}

fn transpose(a: &Mat) -> Mat {
    [0, 1, 2].map(|i| [0, 1, 2].map(|j| a[j][i]))
}

/// MJCF default classes: class -> (parent class, tag -> attributes).
type Defaults = BTreeMap<String, (Option<String>, BTreeMap<String, BTreeMap<String, String>>)>;

fn defaults(root: roxmltree::Node) -> Defaults {
    fn walk(node: roxmltree::Node, parent: Option<String>, out: &mut Defaults) {
        let class = node.attribute("class").unwrap_or("main").to_owned();
        let mut tags = BTreeMap::new();
        for child in node.children().filter(|c| c.is_element()) {
            if child.tag_name().name() == "default" {
                continue;
            }
            let attrs = child
                .attributes()
                .map(|a| (a.name().to_owned(), a.value().to_owned()))
                .collect();
            tags.insert(child.tag_name().name().to_owned(), attrs);
        }
        out.insert(class.clone(), (parent, tags));
        for child in node
            .children()
            .filter(|c| c.is_element() && c.tag_name().name() == "default")
        {
            walk(child, Some(class.clone()), out);
        }
    }
    let mut out = Defaults::new();
    for d in root
        .children()
        .filter(|c| c.is_element() && c.tag_name().name() == "default")
    {
        walk(d, None, &mut out);
    }
    out
}

/// Resolve an attribute through the element, its class chain and `main`.
fn attr(node: roxmltree::Node, name: &str, d: &Defaults) -> Option<String> {
    if let Some(v) = node.attribute(name) {
        return Some(v.to_owned());
    }
    let mut class = node.attribute("class").map(str::to_owned).or_else(|| {
        node.ancestors()
            .filter(|a| a.tag_name().name() == "body")
            .find_map(|a| a.attribute("childclass").map(str::to_owned))
    });
    if class.is_none() {
        class = Some("main".to_owned());
    }
    let tag = node.tag_name().name();
    while let Some(c) = class {
        let (parent, tags) = d.get(&c).unwrap_or_else(|| panic!("class {c}"));
        if let Some(v) = tags.get(tag).and_then(|t| t.get(name)) {
            return Some(v.clone());
        }
        class = parent.clone();
    }
    None
}

fn child<'a>(node: roxmltree::Node<'a, 'a>, tag: &str) -> Vec<roxmltree::Node<'a, 'a>> {
    node.children()
        .filter(|c| c.is_element() && c.tag_name().name() == tag)
        .collect()
}

fn pose(node: roxmltree::Node) -> Frame {
    let p = node.attribute("pos").map_or([0.0; 3], |t| {
        let v = nums(t);
        [v[0], v[1], v[2]]
    });
    let r = node
        .attribute("quat")
        .map_or(quat(&[1.0, 0.0, 0.0, 0.0]), |t| quat(&nums(t)));
    Frame { r, p }
}

/// Load one model; panics on any unexpected source structure.
pub fn load(root: &Path, name: &'static str) -> Model {
    let (file, sha, base, tool_body, tool_site): (_, _, _, _, Option<&str>) = match name {
        "panda" => (PANDA_MJCF, PANDA_SHA256, "link0", "hand", None),
        "ur5e" => (
            UR5E_MJCF,
            UR5E_SHA256,
            "base",
            "wrist_3_link",
            Some("attachment_site"),
        ),
        other => panic!("model {other}"),
    };
    let text = std::fs::read_to_string(root.join(file)).unwrap();
    assert_eq!(
        hex(&Sha256::digest(text.as_bytes())),
        sha,
        "{file} is the pinned source"
    );
    let doc = roxmltree::Document::parse(&text).unwrap();
    let d = defaults(doc.root_element());
    let base_node = doc
        .descendants()
        .find(|n| n.tag_name().name() == "body" && n.attribute("name") == Some(base))
        .unwrap();
    let mut bodies = Vec::new();
    let mut geoms = Vec::new();
    let mut tool = None;
    let mut stack = vec![(base_node, None::<usize>)];
    while let Some((node, parent)) = stack.pop() {
        let index = bodies.len();
        let mut fixed = if parent.is_some() {
            pose(node)
        } else {
            Frame {
                r: quat(&[1.0, 0.0, 0.0, 0.0]),
                p: [0.0; 3],
            }
        };
        let mut hinge = None;
        for j in child(node, "joint") {
            let kind = attr(j, "type", &d).unwrap_or_else(|| "hinge".into());
            let a = nums(&attr(j, "axis", &d).unwrap_or_else(|| "0 0 1".into()));
            let range = nums(&attr(j, "range", &d).unwrap());
            let armature = attr(j, "armature", &d).map_or(0.0, |t| nums(&t)[0]);
            let jname = j.attribute("name").unwrap().to_owned();
            match kind.as_str() {
                "hinge" => {
                    let axis = match (a[0], a[1], a[2]) {
                        (1.0, 0.0, 0.0) => Axis::X,
                        (0.0, 1.0, 0.0) => Axis::Y,
                        (0.0, 0.0, 1.0) => Axis::Z,
                        other => panic!("{jname} axis {other:?}"),
                    };
                    assert!(hinge.is_none(), "one hinge per body");
                    hinge = Some((jname, axis, range[0], range[1], armature));
                }
                "slide" => {
                    assert_eq!(name, "panda", "only Panda fingers slide");
                    let shift = apply(&fixed.r, &[a[0], a[1], a[2]]);
                    for (p, s) in fixed.p.iter_mut().zip(shift) {
                        *p += s * PANDA_FINGER_M;
                    }
                }
                other => panic!("{jname} type {other}"),
            }
        }
        let inertial = child(node, "inertial")[0];
        let ipose = pose(inertial);
        let inertia_local = if let Some(t) = inertial.attribute("fullinertia") {
            let v = nums(t);
            [[v[0], v[3], v[4]], [v[3], v[1], v[5]], [v[4], v[5], v[2]]]
        } else {
            let v = nums(inertial.attribute("diaginertia").unwrap());
            [[v[0], 0.0, 0.0], [0.0, v[1], 0.0], [0.0, 0.0, v[2]]]
        };
        let inertia = mul(&mul(&ipose.r, &inertia_local), &transpose(&ipose.r));
        // Panda collision geometry is meshes and pads: excluded (stated limitation).
        for g in child(node, "geom").into_iter().filter(|_| name == "ur5e") {
            let class = attr(g, "class", &d).unwrap_or_default();
            if class != "collision" && class != "eef_collision" {
                continue;
            }
            let kind = attr(g, "type", &d).unwrap();
            let size = nums(&attr(g, "size", &d).unwrap());
            let gp = pose(g);
            geoms.push(Geom {
                body: index,
                capsule: match kind.as_str() {
                    "capsule" => true,
                    "cylinder" => false,
                    other => panic!("geom type {other}"),
                },
                radius: size[0],
                half_length: size[1],
                center: gp.p,
                axis: [gp.r[0][2], gp.r[1][2], gp.r[2][2]],
            });
        }
        let body_name = node.attribute("name").unwrap().to_owned();
        if body_name == tool_body {
            tool = Some((
                index,
                match tool_site {
                    None => Frame {
                        r: quat(&[1.0, 0.0, 0.0, 0.0]),
                        p: [0.0; 3],
                    },
                    Some(site) => pose(
                        child(node, "site")
                            .into_iter()
                            .find(|s| s.attribute("name") == Some(site))
                            .unwrap(),
                    ),
                },
            ));
        }
        bodies.push(Body {
            name: body_name,
            parent,
            fixed,
            hinge,
            mass: nums(inertial.attribute("mass").unwrap())[0],
            com: ipose.p,
            inertia,
        });
        // Depth-first in source order: push children reversed.
        for c in child(node, "body").into_iter().rev() {
            stack.push((c, Some(index)));
        }
    }
    Model {
        name,
        file,
        sha256: sha,
        bodies,
        tool: tool.unwrap(),
        tool_name: tool_site.unwrap_or(tool_body),
        geoms,
    }
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
