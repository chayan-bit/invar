//! P5 tests: collision masks, contact defaults and the residual qualified model.

use super::*;

/// P5: the pinned robot roles are the vendored models' effective masks, and
/// the real Panda/UR5e policy sources bind with every robot geom colliding and
/// Panda's link0/link1 self exclusion.
#[test]
fn p5_vendored_collision_masks_are_resolved_and_pinned() {
    for (model, root, layout) in [
        (
            "franka_emika_panda/panda.xml",
            "link0",
            Layout::SevenArmTendon,
        ),
        ("universal_robots_ur5e/ur5e.xml", "base", Layout::SixArm),
    ] {
        let xml = vendored(model);
        assert_eq!(vendored_roles(&xml, root), layout.robot_geoms(), "{model}");
        // A default that zeroes the collision class changes the effective
        // roles, so a model edited that way cannot match the pin.
        const COLLISION: &str = r#"<default class="collision">"#;
        assert_eq!(xml.matches(COLLISION).count(), 1, "{model}");
        let zeroed = xml.replace(
            COLLISION,
            r#"<default class="collision"><geom contype="0" conaffinity="0"/>"#,
        );
        let roles = vendored_roles(&zeroed, root);
        assert!(
            !roles.contains('c') && roles != layout.robot_geoms(),
            "{model}"
        );
    }
    for n in [6, 7] {
        let ir = compile(&source(n)).unwrap();
        assert!(bind(&ir).unwrap().is_some(), "arm {n}");
        let xml = rendered_model(&ir);
        let doc = roxmltree::Document::parse(&xml).unwrap();
        let excludes: Vec<_> = doc
            .descendants()
            .filter(|e| e.has_tag_name("exclude"))
            .map(|e| (e.attribute("body1").unwrap(), e.attribute("body2").unwrap()))
            .collect();
        assert_eq!(
            excludes,
            if n == 7 {
                vec![("link0", "link1")]
            } else {
                // The cell's checked mount exclusions for the UR5e.
                vec![("table", "base"), ("table", "shoulder_link")]
            }
        );
        assert!(doc.descendants().any(|g| g.has_tag_name("geom")
            && g.attribute("contype") == Some("0")
            && g.attribute("conaffinity") == Some("0")));
    }
}

/// P5: direct or default-driven zero masks, partial or disjoint masks, role
/// flips, and exclusions that could hide robot contact are all refused.
#[test]
fn p5_contact_masks_defaults_and_exclusions_fail_closed() {
    const MESSAGE: &str =
        "substep policy collision masks, defaults or contact exclusions unsupported";
    for n in [6, 7] {
        let ir = compile(&source(n)).unwrap();
        let xml = rendered_model(&ir);
        let doc = roxmltree::Document::parse(&xml).unwrap();
        let root = if n == 7 { "link0" } else { "base" };
        let robot = doc
            .descendants()
            .find(|b| b.has_tag_name("body") && b.attribute("name") == Some(root))
            .unwrap();
        let mut chain = std::iter::successors(Some(robot), |b| {
            b.children().find(|c| c.has_tag_name("body"))
        })
        .map(|b| b.attribute("name").unwrap().to_owned());
        let (r0, r1, r2) = (
            chain.next().unwrap(),
            chain.next().unwrap(),
            chain.next().unwrap(),
        );
        let robot_collision =
            |g: roxmltree::Node<'_, '_>| in_body(g, root) && !g.has_attribute("contype");
        let table = |g: roxmltree::Node<'_, '_>| in_body(g, "table");
        let cube_collision =
            |g: roxmltree::Node<'_, '_>| in_body(g, "cube") && !g.has_attribute("contype");
        let floor =
            |g: roxmltree::Node<'_, '_>| g.parent().is_some_and(|p| p.has_tag_name("worldbody"));
        let contact = |body: &str| {
            xml.replacen(
                "</mujoco>",
                &format!("<contact>{body}</contact></mujoco>"),
                1,
            )
        };
        let open = xml.find("<mujoco").unwrap();
        let open = open + xml[open..].find('>').unwrap() + 1;
        let default =
            |inner: &str| format!("{}<default>{inner}</default>{}", &xml[..open], &xml[open..]);
        let refused = [
            (
                "robot contype 0",
                retag(&xml, robot_collision, &[("contype", "0")]),
            ),
            (
                "robot conaffinity 0",
                retag(&xml, robot_collision, &[("conaffinity", "0")]),
            ),
            (
                "robot made visual",
                retag(
                    &xml,
                    robot_collision,
                    &[("contype", "0"), ("conaffinity", "0")],
                ),
            ),
            (
                "robot disjoint mask 2",
                retag(
                    &xml,
                    robot_collision,
                    &[("contype", "2"), ("conaffinity", "2")],
                ),
            ),
            (
                "robot mask spelling",
                retag(&xml, robot_collision, &[("contype", "01")]),
            ),
            (
                "table masks 0",
                retag(&xml, table, &[("contype", "0"), ("conaffinity", "0")]),
            ),
            (
                "object conaffinity 0",
                retag(&xml, cube_collision, &[("conaffinity", "0")]),
            ),
            (
                "floor masks 0",
                retag(&xml, floor, &[("contype", "0"), ("conaffinity", "0")]),
            ),
            (
                "default main mask 0",
                default(r#"<geom contype="0" conaffinity="0"/>"#),
            ),
            (
                "default class",
                default(r#"<default class="quiet"><geom contype="0"/></default>"#),
            ),
            (
                "geom class",
                retag(&xml, robot_collision, &[("class", "quiet")]),
            ),
            ("body childclass", {
                // Insert after `<body`, whatever attributes the generator
                // renders before `name` (gravcomp since decision D1).
                let doc = roxmltree::Document::parse(&xml).unwrap();
                let at = doc
                    .descendants()
                    .find(|b| b.has_tag_name("body") && b.attribute("name") == Some(&r1))
                    .expect("second robot body")
                    .range()
                    .start
                    + "<body".len();
                format!("{} childclass=\"quiet\"{}", &xml[..at], &xml[at..])
            }),
            (
                "exclude robot-object",
                contact(&format!(r#"<exclude body1="{r2}" body2="cube"/>"#)),
            ),
            (
                "exclude robot-environment",
                contact(&format!(r#"<exclude body1="{r1}" body2="place_target"/>"#)),
            ),
            (
                "exclude robot-world",
                contact(&format!(r#"<exclude body1="world" body2="{r0}"/>"#)),
            ),
            (
                "exclude table-object",
                contact(r#"<exclude body1="table" body2="cube"/>"#),
            ),
            (
                "exclude robot non-adjacent",
                contact(&format!(r#"<exclude body1="{r0}" body2="{r2}"/>"#)),
            ),
            (
                "exclude robot self",
                contact(&format!(r#"<exclude body1="{r1}" body2="{r1}"/>"#)),
            ),
            (
                "exclude unknown body",
                contact(&format!(r#"<exclude body1="{r1}" body2="ghost"/>"#)),
            ),
            (
                "exclude extra attribute",
                contact(&format!(
                    r#"<exclude body1="{r1}" body2="{r2}" margin="1"/>"#
                )),
            ),
            ("contact pair", contact(r#"<pair geom1="a" geom2="b"/>"#)),
        ];
        for (name, forged) in refused {
            assert_ne!(forged, xml, "arm {n} {name}: forgery not applied");
            assert_eq!(
                bind_forged(&ir, move |_| forged.clone()),
                Err(Fault::ContactVisibility),
                "arm {n} {name}"
            );
        }
        let message = {
            let forged = retag(&xml, table, &[("contype", "0"), ("conaffinity", "0")]);
            FORGE.with(|f| *f.borrow_mut() = Some(Box::new(move |_| forged.clone())));
            let message = validate(&ir);
            FORGE.with(|f| *f.borrow_mut() = None);
            message
        };
        assert_eq!(message, Err(MESSAGE), "arm {n}");
        // Robot-adjacent self exclusions, in either order, pass every contact
        // rule; they change the rendered model, so only the qualified
        // identity pin (P5 residuals 1 and 3) refuses them.
        for (a, b) in [(&r1, &r2), (&r2, &r1), (&r0, &r1)] {
            let forged = contact(&format!(
                r#"<exclude name="adjacent" body1="{a}" body2="{b}"/>"#
            ));
            assert_eq!(
                bind_forged(&ir, move |_| forged.clone()),
                Err(Fault::Identity),
                "arm {n} {a}/{b}"
            );
        }
        // The cell's checked mount exclusions (UR5e base and shoulder on the table)
        // pass the contact rules in either order; forging them again changes the
        // rendered model, so only the identity pin refuses. Any other robot-table
        // exclusion, including every Panda one, is a contact-visibility refusal.
        for (body, declared) in [(&r0, n == 6), (&r1, n == 6), (&r2, false)] {
            for (a, b) in [(body.as_str(), "table"), ("table", body.as_str())] {
                let forged = contact(&format!(r#"<exclude body1="{a}" body2="{b}"/>"#));
                let expected = if declared {
                    Fault::Identity
                } else {
                    Fault::ContactVisibility
                };
                assert_eq!(
                    bind_forged(&ir, move |_| forged.clone()),
                    Err(expected),
                    "arm {n} {a}/{b}"
                );
            }
        }
        assert_eq!(bind_forged(&ir, |xml| xml.to_owned()), Ok(()), "arm {n}");
    }
}

/// P5 residuals 1 and 3: the binding identity is the qualified rendered model
/// and the qualified vendored asset set. Contact parameters, equality/tendon
/// additions and mesh edits that pass every structural rule are refused.
#[test]
fn p5_residual_model_and_assets_are_pinned_to_qualified_identity() {
    for n in [6, 7] {
        let mut ir = compile(&source(n)).unwrap();
        let b = bind(&ir).unwrap().unwrap();
        assert_eq!(
            [b.identity.model_xml.hex(), b.identity.assets.hex()],
            b.layout.qualified_identity(),
            "arm {n}"
        );
        let xml = rendered_model(&ir);
        let root = if n == 7 { "link0" } else { "base" };
        let robot = |g: roxmltree::Node<'_, '_>| in_body(g, root) && !g.has_attribute("contype");
        let table = |g: roxmltree::Node<'_, '_>| in_body(g, "table");
        let cube = |g: roxmltree::Node<'_, '_>| in_body(g, "cube") && !g.has_attribute("contype");
        let append = |inner: &str| xml.replacen("</mujoco>", &format!("{inner}</mujoco>"), 1);
        let forged = [
            ("robot margin", retag(&xml, robot, &[("margin", "0.01")])),
            ("robot gap", retag(&xml, robot, &[("gap", "0.01")])),
            ("table solref", retag(&xml, table, &[("solref", "0.5 1")])),
            (
                "table solimp",
                retag(&xml, table, &[("solimp", "0.1 0.2 0.001")]),
            ),
            ("cube margin", retag(&xml, cube, &[("margin", "0.02")])),
            (
                "extra equality",
                append(r#"<equality><weld name="pin" body1="cube" body2="table"/></equality>"#),
            ),
            (
                "extra spatial tendon",
                append(
                    r#"<tendon><spatial name="pull"><site site="invar_tool"/><site site="cube_grasp"/></spatial></tendon>"#,
                ),
            ),
        ];
        for (name, forged) in forged {
            assert_ne!(forged, xml, "arm {n} {name}: forgery not applied");
            assert_eq!(
                bind_forged(&ir, move |_| forged.clone()),
                Err(Fault::Identity),
                "arm {n} {name}"
            );
        }
        let robot_id = selected_robot_id(&ir);
        let files = ir.assets.get_mut(&robot_id).unwrap();
        // The UR5e model has no mesh asset, so its license file stands in.
        let mesh = files
            .keys()
            .find(|name| name.ends_with(".stl") || name.ends_with(".obj"))
            .or_else(|| files.keys().next())
            .cloned()
            .unwrap();
        let original = files[&mesh].clone();
        *files.get_mut(&mesh).unwrap().last_mut().unwrap() ^= 1;
        assert_eq!(bind(&ir).unwrap_err(), Fault::Identity, "arm {n} mesh byte");
        let files = ir.assets.get_mut(&robot_id).unwrap();
        files.insert(mesh.clone(), original.clone());
        assert!(bind(&ir).is_ok(), "arm {n} restored mesh");
        let files = ir.assets.get_mut(&robot_id).unwrap();
        files.remove(&mesh);
        files.insert(format!("{mesh}.renamed"), original);
        assert_eq!(bind(&ir).unwrap_err(), Fault::Identity, "arm {n} mesh name");
    }
}

/// P5 residual 2: the robot-table contact gate is qualified only for a cell
/// with exactly one table-class (collision fixture) body; it is not inferred.
#[test]
fn p5_residual_cells_without_exactly_one_table_body_are_refused() {
    const MESSAGE: &str =
        "substep policy collision masks, defaults or contact exclusions unsupported";
    const TABLE: &str = "    fixture table {\n        pose xyz (0.4 m, 0 m, -0.05 m) quat (1, 0, 0, 0);\n        collision box (0.5, 0.5, 0.05) m;\n    }\n";
    // The mount relation names the table; without it the substep gate is what refuses.
    const MOUNT: &str = " on table\n        exclude (menagerie::universal_robots_ur5e::base, menagerie::universal_robots_ur5e::shoulder_link)";
    for n in [6, 7] {
        let base = source(n).replace(MOUNT, "");
        assert!(base.contains(TABLE), "arm {n}");
        let second = TABLE
            .replace("fixture table", "fixture side_table")
            .replace("0.4 m, 0 m, -0.05 m", "0.4 m, 1.5 m, -0.05 m");
        for (name, changed) in [
            ("no table", base.replace(TABLE, "")),
            (
                "visual-only table",
                base.replace(TABLE, &TABLE.replace("collision box", "visual box")),
            ),
            (
                "two tables",
                base.replace(TABLE, &format!("{TABLE}{second}")),
            ),
        ] {
            let error = compile(&changed).unwrap_err();
            assert_eq!(error.code, "S1504", "arm {n} {name}: {}", error.message);
            assert!(
                error.message.contains(MESSAGE),
                "arm {n} {name}: {}",
                error.message
            );
        }
    }
}
