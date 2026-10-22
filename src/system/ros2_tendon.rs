//! ROS-only actuator naming projection; physical MJCF attributes stay unchanged.
use std::collections::BTreeSet;

use super::xml::Element;

fn sections<'a>(tree: &'a Element, tag: &'a str) -> impl Iterator<Item = &'a Element> {
    tree.children
        .iter()
        .filter(move |node| node.tag == tag)
        .flat_map(|node| &node.children)
}

fn attribute<'a>(node: &'a Element, key: &str) -> Result<&'a str, &'static str> {
    node.attrs
        .get(key)
        .map(String::as_str)
        .ok_or("tendon naming requires explicit names")
}

fn joint_count(tree: &Element, name: &str) -> usize {
    usize::from(tree.tag == "joint" && tree.attrs.get("name").is_some_and(|value| value == name))
        + tree
            .children
            .iter()
            .map(|child| joint_count(child, name))
            .sum::<usize>()
}

fn references_actuator(tree: &Element, name: &str) -> bool {
    tree.attrs
        .get("actuator")
        .is_some_and(|value| value == name)
        || tree
            .children
            .iter()
            .any(|child| references_actuator(child, name))
}

/// Resolve a symmetric two-joint tendon to the primary joint of its explicit
/// identity equality. This is structural, not a robot/fixture-name heuristic.
/// Validate every mapping before mutation so rejected projections stay atomic.
pub(super) fn project(tree: &mut Element) -> Result<(), &'static str> {
    let mut mappings = Vec::new();
    let mut names = BTreeSet::new();
    for actuator in sections(tree, "actuator") {
        if let Some(name) = actuator.attrs.get("name")
            && !names.insert(name.as_str())
        {
            return Err("duplicate actuator name in ROS tendon projection");
        }
    }
    let mut targets = BTreeSet::new();
    for actuator in sections(tree, "actuator") {
        let Some(tendon_name) = actuator.attrs.get("tendon") else {
            continue;
        };
        let old = attribute(actuator, "name")?;
        let candidates: Vec<_> = sections(tree, "tendon")
            .filter(|node| node.attrs.get("name") == Some(tendon_name))
            .collect();
        let [tendon] = candidates.as_slice() else {
            return Err("ROS tendon projection needs one fixed tendon definition");
        };
        if tendon.tag != "fixed"
            || tendon.children.len() != 2
            || tendon.children.iter().any(|node| {
                node.tag != "joint" || node.attrs.get("coef").map(String::as_str) != Some("0.5")
            })
        {
            return Err("ROS tendon projection supports only symmetric two-joint tendons");
        }
        let left = attribute(&tendon.children[0], "joint")?;
        let right = attribute(&tendon.children[1], "joint")?;
        if left == right || joint_count(tree, left) != 1 || joint_count(tree, right) != 1 {
            return Err("ROS tendon projection needs two distinct existing joints");
        }
        let equalities: Vec<_> = sections(tree, "equality")
            .filter(|node| {
                let a = node.attrs.get("joint1").map(String::as_str);
                let b = node.attrs.get("joint2").map(String::as_str);
                node.tag == "joint"
                    && ((a == Some(left) && b == Some(right))
                        || (a == Some(right) && b == Some(left)))
            })
            .collect();
        let [equality] = equalities.as_slice() else {
            return Err("ROS tendon projection needs one unambiguous joint equality");
        };
        if equality
            .attrs
            .get("polycoef")
            .is_some_and(|value| value != "0 1 0 0 0")
            || equality
                .attrs
                .get("active")
                .is_some_and(|value| value != "true" && value != "1")
        {
            return Err("ROS tendon projection requires an active identity joint equality");
        }
        let primary = attribute(equality, "joint1")?;
        if !targets.insert(primary) || (primary != old && names.contains(primary)) {
            return Err("ROS tendon projection would collide with an actuator name");
        }
        if primary != old && references_actuator(tree, old) {
            return Err("ROS tendon projection cannot rename a referenced actuator");
        }
        mappings.push((old.to_owned(), primary.to_owned()));
    }
    for section in tree
        .children
        .iter_mut()
        .filter(|node| node.tag == "actuator")
    {
        for actuator in &mut section.children {
            if let Some((_, primary)) = mappings
                .iter()
                .find(|(old, _)| actuator.attrs.get("name") == Some(old))
            {
                actuator.attrs.insert("name".into(), primary.clone());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(tag: &str, attrs: &[(&str, &str)], children: Vec<Element>) -> Element {
        Element {
            tag: tag.into(),
            attrs: attrs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            children,
        }
    }

    fn model() -> Element {
        node(
            "mujoco",
            &[],
            vec![
                node(
                    "worldbody",
                    &[],
                    vec![node(
                        "body",
                        &[],
                        vec![
                            node("joint", &[("name", "left"), ("type", "slide")], vec![]),
                            node("joint", &[("name", "right"), ("type", "slide")], vec![]),
                        ],
                    )],
                ),
                node(
                    "tendon",
                    &[],
                    vec![node(
                        "fixed",
                        &[("name", "coupler")],
                        vec![
                            node("joint", &[("joint", "left"), ("coef", "0.5")], vec![]),
                            node("joint", &[("joint", "right"), ("coef", "0.5")], vec![]),
                        ],
                    )],
                ),
                node(
                    "equality",
                    &[],
                    vec![node(
                        "joint",
                        &[("joint1", "left"), ("joint2", "right")],
                        vec![],
                    )],
                ),
                node(
                    "actuator",
                    &[],
                    vec![node(
                        "general",
                        &[
                            ("name", "drive"),
                            ("tendon", "coupler"),
                            ("gainprm", "0.01568627451 0 0"),
                            ("ctrlrange", "0 255"),
                        ],
                        vec![],
                    )],
                ),
            ],
        )
    }

    #[test]
    fn names_only_projection_is_idempotent_and_not_robot_specific() {
        let original = model();
        let mut projected = original.clone();
        project(&mut projected).unwrap();
        let mut expected = original;
        expected.children[3].children[0]
            .attrs
            .insert("name".into(), "left".into());
        assert_eq!(projected, expected);
        project(&mut projected).unwrap();
        assert_eq!(projected, expected);
    }

    #[test]
    fn malformed_coupling_and_name_collisions_reject_without_mutation() {
        for kind in 0..5 {
            let mut bad = model();
            match kind {
                0 => {
                    bad.children[1].children[0].children[0]
                        .attrs
                        .insert("coef".into(), "0.4".into());
                }
                1 => {
                    bad.children[2].children[0]
                        .attrs
                        .insert("polycoef".into(), "0 -1 0 0 0".into());
                }
                2 => {
                    let duplicate = bad.children[2].children[0].clone();
                    bad.children[2].children.push(duplicate);
                }
                3 => {
                    bad.children[3].children.push(node(
                        "motor",
                        &[("name", "left"), ("joint", "left")],
                        vec![],
                    ));
                }
                _ => {
                    bad.children.push(node(
                        "sensor",
                        &[],
                        vec![node("actuatorfrc", &[("actuator", "drive")], vec![])],
                    ));
                }
            }
            let before = bad.clone();
            assert!(project(&mut bad).is_err(), "case {kind}");
            assert_eq!(bad, before, "case {kind}");
        }
    }
}
