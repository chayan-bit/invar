//! Checked cell mount surface and mount contact exclusions (exact rational geometry).
//!
//! `mount POSE on FIXTURE exclude (ROBOT::LINK, ...)` declares that the robot is
//! mounted on the top face of FIXTURE's single collision box and that contact between
//! FIXTURE and the named links is excluded. Links are robot-qualified so one cell
//! source serves every arm; consumers apply the exclusions of the selected robot.
//! A link may be excluded only when it is mounted to that surface: the root link, or
//! a link whose chain to the root has only fixed joints and revolute joints whose axis
//! is parallel to the surface normal. Such a link only ever rotates about the normal,
//! so its distance to the surface is the same in every configuration and excluding
//! the pair cannot hide a motion-dependent collision.

use std::collections::BTreeMap;

use num_rational::BigRational;
use serde_json::{Value, json};

use crate::front::{FrontError, Span};

use super::{error, graph};

type R = BigRational;
type V = [R; 3];
/// Quaternion `[w, x, y, z]`; never normalized, rotations divide by the squared norm.
type Q = [R; 4];

const CODE: &str = "S1608";

fn invalid(message: impl Into<String>, span: Span) -> FrontError {
    error(CODE, message, span)
}

fn zero() -> R {
    R::from_integer(0.into())
}

fn number(value: &Value) -> Option<R> {
    graph::decimal(value.as_str()?)
}

fn pose(value: &Value) -> Option<(V, Q)> {
    let xyz = value["xyz"].as_array().filter(|v| v.len() == 3)?;
    let quat = value["quat"].as_array().filter(|v| v.len() == 4)?;
    let t = [
        number(&xyz[0]["value"])?,
        number(&xyz[1]["value"])?,
        number(&xyz[2]["value"])?,
    ];
    let q = [
        number(&quat[0])?,
        number(&quat[1])?,
        number(&quat[2])?,
        number(&quat[3])?,
    ];
    (!q.iter().all(|part| *part == zero())).then_some((t, q))
}

fn mul(a: &Q, b: &Q) -> Q {
    [
        &a[0] * &b[0] - &a[1] * &b[1] - &a[2] * &b[2] - &a[3] * &b[3],
        &a[0] * &b[1] + &a[1] * &b[0] + &a[2] * &b[3] - &a[3] * &b[2],
        &a[0] * &b[2] - &a[1] * &b[3] + &a[2] * &b[0] + &a[3] * &b[1],
        &a[0] * &b[3] + &a[1] * &b[2] - &a[2] * &b[1] + &a[3] * &b[0],
    ]
}

fn conjugate(q: &Q) -> Q {
    [q[0].clone(), -&q[1], -&q[2], -&q[3]]
}

/// Exact `q v q* / |q|^2`.
fn rotate(q: &Q, v: &V) -> V {
    let p = mul(
        &mul(q, &[zero(), v[0].clone(), v[1].clone(), v[2].clone()]),
        &conjugate(q),
    );
    let norm = q.iter().fold(zero(), |sum, part| sum + part * part);
    [&p[1] / &norm, &p[2] / &norm, &p[3] / &norm]
}

fn cross_is_zero(a: &V, b: &V) -> bool {
    &a[1] * &b[2] == &a[2] * &b[1]
        && &a[2] * &b[0] == &a[0] * &b[2]
        && &a[0] * &b[1] == &a[1] * &b[0]
}

fn unit_z() -> V {
    [zero(), zero(), R::from_integer(1.into())]
}

/// Checks the mount surface against the cell alone and returns its IR value.
pub(super) fn surface(
    cell_mount: &Value,
    objects: &[Value],
    fixture: &str,
    exclude: &[String],
    span: Span,
) -> Result<Value, FrontError> {
    let mut names = std::collections::BTreeSet::new();
    if exclude.iter().any(|link| !names.insert(link.as_str())) {
        return Err(error("S1600", "duplicate mount exclusion link", span));
    }
    let exclude = exclude
        .iter()
        .map(|path| {
            path.rsplit_once("::")
                .map(|(robot, link)| json!({"robot":robot,"link":link}))
                .ok_or_else(|| error("S1600", "mount exclusion must name ROBOT::LINK", span))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let object = objects
        .iter()
        .find(|object| object["name"] == fixture && object["kind"] == "fixture")
        .ok_or_else(|| {
            error(
                "S1601",
                format!("mount surface fixture '{fixture}' does not resolve"),
                span,
            )
        })?;
    let boxes: Vec<_> = object["geometries"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|geometry| geometry["role"] == "collision")
        .collect();
    if boxes.len() != 1 || boxes[0]["kind"] != "box" {
        return Err(invalid(
            format!("mount surface fixture '{fixture}' needs exactly one collision box"),
            span,
        ));
    }
    let half = boxes[0]["dimensions"]
        .as_array()
        .and_then(|d| Some([number(d.first()?)?, number(d.get(1)?)?, number(d.get(2)?)?]))
        .ok_or_else(|| invalid("mount surface box dimensions are malformed", span))?;
    let ((fixture_t, fixture_q), (mount_t, mount_q)) = pose(&object["pose"])
        .zip(pose(cell_mount))
        .ok_or_else(|| invalid("mount surface pose is malformed", span))?;
    let offset = [
        &mount_t[0] - &fixture_t[0],
        &mount_t[1] - &fixture_t[1],
        &mount_t[2] - &fixture_t[2],
    ];
    let local = rotate(&conjugate(&fixture_q), &offset);
    let normal = rotate(&fixture_q, &unit_z());
    let mount_z = rotate(&mount_q, &unit_z());
    let dot = (0..3).fold(zero(), |sum, i| sum + &normal[i] * &mount_z[i]);
    if local[2] != half[2]
        || local[0] > half[0]
        || -&local[0] > half[0]
        || local[1] > half[1]
        || -&local[1] > half[1]
        || !cross_is_zero(&normal, &mount_z)
        || dot <= zero()
    {
        return Err(invalid(
            format!(
                "cell mount is not on the top face of fixture '{fixture}' with its z axis along the face normal"
            ),
            span,
        ));
    }
    Ok(json!({"fixture":fixture,"exclude":exclude}))
}

/// Checks that every excluded link of `cell` resolves and is mounted to its surface.
pub(super) fn check_mounted(
    cell: &Value,
    robots: &BTreeMap<String, Value>,
    span: Span,
) -> Result<(), FrontError> {
    let fixture = cell["mount_surface"]["fixture"].as_str().unwrap_or("");
    for entry in cell["mount_surface"]["exclude"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let (robot_id, name) = (
            entry["robot"].as_str().unwrap_or(""),
            entry["link"].as_str().unwrap_or(""),
        );
        let robot = robots.get(robot_id).ok_or_else(|| {
            error(
                "S1601",
                format!("mount exclusion robot '{robot_id}' does not resolve"),
                span,
            )
        })?;
        let links: BTreeMap<&str, &Value> = robot["links"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|link| Some((link["name"].as_str()?, link)))
            .collect();
        let joints: BTreeMap<&str, &Value> = robot["joints"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|joint| Some((joint["child"].as_str()?, joint)))
            .collect();
        if !links.contains_key(name) {
            return Err(error(
                "S1601",
                format!("mount exclusion link '{robot_id}::{name}' does not resolve"),
                span,
            ));
        }
        if !mounted(name, &links, &joints)
            .ok_or_else(|| invalid("robot kinematic chain is malformed", span))?
        {
            return Err(invalid(
                format!(
                    "mount exclusion link '{robot_id}::{name}' is not mounted to fixture '{fixture}'"
                ),
                span,
            ));
        }
    }
    Ok(())
}

/// The checked excluded links of `cell` that belong to `robot_id`.
pub(crate) fn excluded_links<'a>(cell: &'a Value, robot_id: &str) -> Vec<&'a str> {
    cell["mount_surface"]["exclude"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|entry| entry["robot"] == robot_id)
        .filter_map(|entry| entry["link"].as_str())
        .collect()
}

/// Some(true) when every joint from the root to `name` is fixed, or revolute about
/// an axis parallel to the mount z axis (the checked surface normal).
fn mounted(
    name: &str,
    links: &BTreeMap<&str, &Value>,
    joints: &BTreeMap<&str, &Value>,
) -> Option<bool> {
    let mut chain = vec![name];
    while let Some(parent) = links.get(chain.last()?)?["parent"]
        .as_str()
        .filter(|p| *p != "world")
    {
        if chain.len() > links.len() {
            return None;
        }
        chain.push(parent);
    }
    let mut frame: Q = [R::from_integer(1.into()), zero(), zero(), zero()];
    for link in chain.iter().rev() {
        frame = mul(&frame, &pose(&links.get(link)?["origin"])?.1);
        let Some(joint) = joints.get(link) else {
            continue;
        };
        match joint["kind"].as_str()? {
            "fixed" => {}
            "revolute" => {
                let axis = joint["axis"].as_array().filter(|a| a.len() == 3)?;
                let axis = [number(&axis[0])?, number(&axis[1])?, number(&axis[2])?];
                if !cross_is_zero(&rotate(&frame, &axis), &unit_z()) {
                    return Some(false);
                }
            }
            _ => return Some(false),
        }
    }
    Some(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(xyz: [&str; 3], quat: [&str; 4]) -> Value {
        json!({"xyz":xyz.iter().map(|v| json!({"value":v,"unit":"m"})).collect::<Vec<_>>(),"quat":quat})
    }

    fn table() -> Vec<Value> {
        vec![
            json!({"kind":"fixture","name":"table","pose":p(["0.4","0","-0.05"],["1","0","0","0"]),
            "geometries":[{"role":"collision","kind":"box","dimensions":["0.5","0.5","0.05"]}]}),
        ]
    }

    fn robot() -> Value {
        // base rotated pi about z; pan about z; lift about a horizontal axis.
        json!({"links":[
            {"name":"base","parent":"world","origin":p(["0","0","0"],["0","0","0","-1"])},
            {"name":"shoulder","parent":"base","origin":p(["0","0","0.163"],["1","0","0","0"])},
            {"name":"upper","parent":"shoulder","origin":p(["0","0.138","0"],["1","0","1","0"])},
            {"name":"flange","parent":"base","origin":p(["0","0","0.01"],["1","1","0","0"])}],
          "joints":[
            {"kind":"revolute","child":"shoulder","axis":["0","0","1"]},
            {"kind":"revolute","child":"upper","axis":["0","1","0"]},
            {"kind":"fixed","child":"flange","axis":["0","0","0"]}]})
    }

    #[test]
    fn surface_requires_the_top_face_and_a_parallel_normal() {
        let span = Span { start: 0, end: 0 };
        let level = p(["0", "0", "0"], ["1", "0", "0", "0"]);
        let ok = surface(&level, &table(), "table", &["arm::base".into()], span);
        assert_eq!(
            ok.unwrap(),
            json!({"fixture":"table","exclude":[{"robot":"arm","link":"base"}]})
        );
        // Rotation about the normal keeps the mount valid; any other pose is refused.
        let turned = p(["0.1", "0.2", "0"], ["0", "0", "0", "3"]);
        assert!(surface(&turned, &table(), "table", &[], span).is_ok());
        for (xyz, quat) in [
            (["0", "0", "0.001"], ["1", "0", "0", "0"]),
            (["-0.2", "0", "0"], ["1", "0", "0", "0"]),
            (["0", "0", "0"], ["0", "1", "0", "0"]),
            (["0", "0", "0"], ["1", "1", "0", "0"]),
        ] {
            let refused = surface(&p(xyz, quat), &table(), "table", &[], span).unwrap_err();
            assert_eq!(refused.code, CODE);
        }
        let code = |fixture, exclude: &[String]| {
            surface(&level, &table(), fixture, exclude, span)
                .unwrap_err()
                .code
        };
        assert_eq!(code("cube", &[]), "S1601");
        assert_eq!(code("table", &["a::b".into(), "a::b".into()]), "S1600");
        assert_eq!(code("table", &["base".into()]), "S1600");
    }

    #[test]
    fn only_links_rotating_about_the_normal_are_mounted() {
        let span = Span { start: 0, end: 0 };
        let robots = BTreeMap::from([("arm".to_owned(), robot())]);
        let cell = |links: &[&str]| {
            let exclude: Vec<_> = links
                .iter()
                .map(|link| json!({"robot":"arm","link":link}))
                .collect();
            json!({"mount_surface":{"fixture":"table","exclude":exclude}})
        };
        let mounted = cell(&["base", "shoulder"]);
        assert!(check_mounted(&mounted, &robots, span).is_ok());
        assert_eq!(excluded_links(&mounted, "arm"), ["base", "shoulder"]);
        assert!(excluded_links(&mounted, "other").is_empty());
        let refused = check_mounted(&cell(&["upper"]), &robots, span).unwrap_err();
        assert_eq!(refused.code, CODE);
        assert!(
            refused
                .message
                .contains("'arm::upper' is not mounted to fixture 'table'")
        );
        // A fixed child is mounted; a missing link or robot is a resolution error.
        assert!(check_mounted(&cell(&["flange"]), &robots, span).is_ok());
        let missing = check_mounted(&cell(&["missing"]), &robots, span);
        assert_eq!(missing.unwrap_err().code, "S1601");
        let other =
            json!({"mount_surface":{"fixture":"table","exclude":[{"robot":"x","link":"base"}]}});
        assert_eq!(
            check_mounted(&other, &robots, span).unwrap_err().code,
            "S1601"
        );
    }
}
