use super::synthesize::element;
use super::*;

fn key_cell() -> Value {
    let pose = |x: &str, quat: [&str; 4]| json!({"xyz":[{"value":x,"unit":"m"},{"value":"0","unit":"m"},{"value":"0.025","unit":"m"}],"quat":quat});
    json!({"name":"KeyCell","mount":pose("0",["1","0","0","0"]),"objects":[
        {"name":"first","pose":pose("0.3",["1","0","0","0"]),"dynamic":{"mass":{"value":"0.1"}},"geometries":[{"kind":"box","role":"collision","dimensions":["0.025","0.025","0.025"]}]},
        {"name":"second","pose":pose("0.6",["0","0","0","2"]),"dynamic":{"mass":{"value":"0.1"}},"geometries":[{"kind":"box","role":"collision","dimensions":["0.025","0.025","0.025"]}]}
    ]})
}

fn key_tree() -> Element {
    element(
        "mujoco",
        &[],
        vec![
            element(
                "worldbody",
                &[],
                vec![element(
                    "body",
                    &[("name", "root".into())],
                    vec![
                        element(
                            "joint",
                            &[("name", "a".into()), ("ref", "0.2".into())],
                            vec![],
                        ),
                        element(
                            "joint",
                            &[
                                ("name", "b".into()),
                                ("type", "slide".into()),
                                ("ref", "-0.3".into()),
                            ],
                            vec![],
                        ),
                    ],
                )],
            ),
            element(
                "keyframe",
                &[],
                vec![
                    element(
                        "key",
                        &[
                            ("name", "partial".into()),
                            ("qpos", "0.7".into()),
                            ("qvel", "0.8".into()),
                            ("ctrl", "5".into()),
                        ],
                        vec![],
                    ),
                    element("key", &[("name", "implicit".into())], vec![]),
                ],
            ),
            element("compiler", &[("angle", "radian".into())], vec![]),
        ],
    )
}

#[test]
fn cell_keys_preserve_robot_prefix_defaults_and_free_pose_order() {
    let mut tree = key_tree();
    let cells = [key_cell()];
    apply_cells(&mut tree, &cells, "root", "robot").unwrap();
    let keys = &tree
        .children
        .iter()
        .find(|c| c.tag == "keyframe")
        .unwrap()
        .children;
    assert_eq!(
        attr(&keys[0], "qpos"),
        Some("0.7 -0.3 0.3 0 0.025 1 0 0 0 0.6 0 0.025 0 0 0 1")
    );
    assert_eq!(
        attr(&keys[1], "qpos"),
        Some("0.2 -0.3 0.3 0 0.025 1 0 0 0 0.6 0 0.025 0 0 0 1")
    );
    assert_eq!(
        attr(&keys[0], "qvel"),
        Some("0.8 0 0 0 0 0 0 0 0 0 0 0 0 0")
    );
    assert_eq!(attr(&keys[1], "qvel"), None);
    assert_eq!(attr(&keys[0], "ctrl"), Some("5"));
}

#[test]
fn cell_gravity_compensates_every_robot_body_and_no_cell_object() {
    let mut tree = key_tree();
    tree.children[0].children[0].children.push(element(
        "body",
        &[("name", "child".into())],
        vec![element("body", &[("name", "grandchild".into())], vec![])],
    ));
    apply_cells(&mut tree, &[key_cell()], "root", "robot").unwrap();
    let world = &tree.children[0];
    // The fixture mount is not identity, so the robot root sits in the mount body.
    let mount = &world.children[0];
    assert_eq!(attr(mount, "name"), Some("KeyCell_mount"));
    assert_eq!(attr(mount, "gravcomp"), None);
    let root = &mount.children[0];
    assert_eq!(attr(root, "gravcomp"), Some("1"));
    assert_eq!(attr(&root.children[2], "gravcomp"), Some("1"));
    assert_eq!(attr(&root.children[2].children[0], "gravcomp"), Some("1"));
    let objects: Vec<_> = world.children.iter().skip(1).collect();
    assert_eq!(objects.len(), 2);
    assert!(objects.iter().all(|o| attr(o, "gravcomp").is_none()));
    // An explicit full compensation is kept; a partial one is refused.
    for (value, accepted) in [("1", true), ("0.5", false)] {
        let mut tree = key_tree();
        tree.children[0].children[0]
            .attrs
            .insert("gravcomp".into(), value.into());
        assert_eq!(
            apply_cells(&mut tree, &[key_cell()], "root", "robot").is_ok(),
            accepted
        );
    }
    let mut tree = key_tree();
    assert_eq!(
        apply_cells(&mut tree, &[key_cell()], "missing", "robot")
            .unwrap_err()
            .code,
        "S1601"
    );
}

#[test]
fn cell_keys_refuse_ambiguous_joint_or_oversized_prefix() {
    let cells = [key_cell()];
    for field in ["qpos", "qvel"] {
        let mut tree = key_tree();
        tree.children[1].children[0]
            .attrs
            .insert(field.into(), "1 2 3".into());
        assert_eq!(
            apply_cells(&mut tree, &cells, "root", "robot")
                .unwrap_err()
                .code,
            "S1607"
        );
    }
    let mut tree = key_tree();
    tree.children[0].children[0].children[0]
        .attrs
        .insert("type".into(), "ball".into());
    assert_eq!(
        apply_cells(&mut tree, &cells, "root", "robot")
            .unwrap_err()
            .code,
        "S1607"
    );
}

#[test]
fn cell_key_defaults_convert_only_degree_hinge_references() {
    let cells = [key_cell()];
    let mut tree = key_tree();
    tree.children[2]
        .attrs
        .insert("angle".into(), "degree".into());
    apply_cells(&mut tree, &cells, "root", "robot").unwrap();
    let key = &tree.children[1].children[1];
    let values = attr(key, "qpos")
        .unwrap()
        .split_whitespace()
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(values[0], 0.2f64.to_radians());
    assert_eq!(values[1], -0.3);
    assert!(
        attr(&tree.children[1].children[0], "qpos")
            .unwrap()
            .starts_with("0.7 -0.3 ")
    );
}

#[test]
fn exact_exponent_normalization_preserves_digits() {
    assert_eq!(exact_decimal("8.2904e-7"), Some("0.00000082904".to_owned()));
    assert_eq!(exact_decimal("-0.000"), Some("0".to_owned()));
    assert_eq!(exact_decimal("2.8110e-2"), Some("0.02811".to_owned()));
}

#[test]
fn pinned_menagerie_includes_and_defaults_are_canonicalized() {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cells/reach_place/assets/menagerie");
    let panda = load_mjcf(&root.join("franka_emika_panda/scene.xml")).unwrap();
    let ur5e = load_mjcf(&root.join("universal_robots_ur5e/scene.xml")).unwrap();
    let panda_xml = panda.render();
    let ur5e_xml = ur5e.render();
    assert!(!panda_xml.contains("<include") && !panda_xml.contains("<default"));
    assert!(!ur5e_xml.contains("<include") && !ur5e_xml.contains("<default"));
    // link3 inertial pos "2.7518e-2 3.9252e-2 -6.6502e-2", exponents normalized.
    assert!(panda_xml.contains("0.027518 0.039252 -0.066502"));
    assert!(ur5e_xml.contains("name=\"elbow_joint\""));
    assert_eq!(
        Element::from_value(&panda.to_value()).unwrap().render(),
        panda_xml
    );
}
