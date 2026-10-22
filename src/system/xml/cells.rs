use super::synthesize::{element, find_body, geometry_element, gravity_compensate, pose_attrs};
use super::*;

pub(crate) fn apply_cells(
    tree: &mut Element,
    cells: &[Value],
    robot_root: &str,
    robot_id: &str,
) -> Result<(), XmlError> {
    if cells.is_empty() {
        return Ok(());
    }
    if cells.len() > 1 {
        return Err(fail(
            "S1600",
            "MJCF generation needs one selected cell per robot",
        ));
    }
    // Cell free joints are appended after the selected robot's scalar joints.
    // Preserve every robot key prefix, but explicitly initialize the new qpos
    // coordinates: MuJoCo fills omitted key coordinates with zero, not qpos0.
    let mut robot_qpos0 = Vec::new();
    fn joint_defaults(
        node: &Element,
        values: &mut Vec<String>,
        degrees: bool,
    ) -> Result<(), XmlError> {
        if node.tag == "freejoint"
            || (node.tag == "joint"
                && !matches!(attr(node, "type"), None | Some("hinge" | "slide")))
        {
            return Err(fail(
                "S1607",
                "cell key extension needs scalar robot joints",
            ));
        }
        if node.tag == "joint" {
            let value = attr(node, "ref").unwrap_or("0");
            values.push(if degrees && attr(node, "type") != Some("slide") {
                let value = value
                    .parse::<f64>()
                    .map_err(|_| fail("S1607", "invalid robot joint reference"))?;
                if !value.is_finite() {
                    return Err(fail("S1607", "nonfinite robot joint reference"));
                }
                value.to_radians().to_string()
            } else {
                value.to_owned()
            });
        }
        for child in &node.children {
            joint_defaults(child, values, degrees)?;
        }
        Ok(())
    }
    let dynamic_count = cells[0]["objects"]
        .as_array()
        .ok_or_else(|| fail("S1607", "checked cell objects are missing"))?
        .iter()
        .filter(|object| !object["dynamic"].is_null())
        .count();
    if dynamic_count != 0 {
        let degrees = match tree
            .children
            .iter()
            .find(|child| child.tag == "compiler")
            .and_then(|compiler| attr(compiler, "angle"))
        {
            Some("radian") => false,
            None | Some("degree") => true,
            _ => return Err(fail("S1607", "unsupported robot compiler angle")),
        };
        for (index, world) in tree
            .children
            .iter()
            .filter(|child| child.tag == "worldbody")
            .enumerate()
        {
            let before = robot_qpos0.len();
            joint_defaults(world, &mut robot_qpos0, degrees)?;
            if index != 0 && robot_qpos0.len() != before {
                return Err(fail(
                    "S1607",
                    "cell key extension needs robot joints in one worldbody",
                ));
            }
        }
    }
    tree.attrs.insert(
        "model".to_owned(),
        cells[0]["name"]
            .as_str()
            .ok_or_else(|| fail("S1607", "checked cell name is missing"))?
            .to_owned(),
    );
    fn body_names(node: &Element, names: &mut BTreeSet<String>) {
        if node.tag == "body"
            && let Some(name) = attr(node, "name")
        {
            names.insert(name.to_owned());
        }
        for child in &node.children {
            body_names(child, names);
        }
    }
    let mut names = BTreeSet::new();
    body_names(tree, &mut names);
    for cell in cells {
        for object in cell["objects"]
            .as_array()
            .ok_or_else(|| fail("S1607", "checked cell objects are missing"))?
        {
            let name = object["name"]
                .as_str()
                .ok_or_else(|| fail("S1607", "cell object name is missing"))?;
            if !names.insert(name.to_owned()) {
                return Err(fail("S1600", format!("cell body name collides: {name}")));
            }
        }
    }
    let world = tree
        .children
        .iter_mut()
        .find(|child| child.tag == "worldbody")
        .ok_or_else(|| fail("S1607", "MJCF worldbody is missing"))?;
    // Decision D1 (S1-06): every body of the selected robot is gravity compensated in
    // the generated cell model, so position-controlled joints hold the arm and a held
    // cube without a gravity-load offset. Cell objects stay uncompensated.
    gravity_compensate(
        find_body(world, robot_root)
            .ok_or_else(|| fail("S1601", "selected robot root body is missing from MJCF"))?,
    )?;
    let mut welds = Vec::new();
    let mut cell_qpos0 = Vec::new();
    for cell in cells {
        let mount = &cell["mount"];
        let mount_attrs = pose_attrs(mount)?;
        let identity = mount_attrs[0].1 == "0 0 0" && mount_attrs[1].1 == "1 0 0 0";
        if !identity {
            let position = world
                .children
                .iter()
                .position(|child| child.tag == "body" && attr(child, "name") == Some(robot_root))
                .ok_or_else(|| fail("S1601", "selected robot root body is missing from MJCF"))?;
            let root = world.children.remove(position);
            let mount_name = format!("{}_mount", cell["name"].as_str().unwrap_or("cell"));
            if !names.insert(mount_name.clone()) {
                return Err(fail("S1600", "cell mount name collides"));
            }
            world.children.insert(
                position,
                element(
                    "body",
                    &[
                        ("name", mount_name),
                        mount_attrs[0].clone(),
                        mount_attrs[1].clone(),
                    ],
                    vec![root],
                ),
            );
        }
        for object in cell["objects"]
            .as_array()
            .ok_or_else(|| fail("S1607", "checked cell objects are missing"))?
        {
            let name = object["name"]
                .as_str()
                .ok_or_else(|| fail("S1607", "cell object name missing"))?;
            let mut attrs = vec![("name", name.to_owned())];
            attrs.extend(pose_attrs(&object["pose"])?);
            let mut children = object["geometries"]
                .as_array()
                .ok_or_else(|| fail("S1607", "cell geometry missing"))?
                .iter()
                .map(geometry_element)
                .collect::<Result<Vec<_>, _>>()?;
            if !object["dynamic"].is_null() {
                let pose = pose_attrs(&object["pose"])?;
                cell_qpos0.extend(pose[0].1.split_whitespace().map(str::to_owned));
                let quaternion = pose[1]
                    .1
                    .split_whitespace()
                    .map(str::parse::<f64>)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| fail("S1607", "invalid cell key quaternion"))?;
                let norm = quaternion.iter().map(|v| v * v).sum::<f64>().sqrt();
                if quaternion.len() != 4 || !norm.is_finite() || norm <= 0.0 {
                    return Err(fail("S1607", "invalid cell key quaternion"));
                }
                cell_qpos0.extend(quaternion.iter().map(|v| (v / norm).to_string()));
                let mass = object["dynamic"]["mass"]["value"]
                    .as_str()
                    .ok_or_else(|| fail("S1607", "dynamic object mass is missing"))?;
                children.insert(
                    0,
                    element("freejoint", &[("name", format!("{name}_free"))], Vec::new()),
                );
                let mut weighted = false;
                for child in &mut children {
                    if child.tag != "geom" {
                        continue;
                    }
                    let collision = attr(child, "contype") != Some("0");
                    child.attrs.insert(
                        "mass".to_owned(),
                        if collision && !weighted {
                            weighted = true;
                            mass.to_owned()
                        } else {
                            "0".to_owned()
                        },
                    );
                }
                if !weighted {
                    return Err(fail("S1602", "dynamic object needs collision geometry"));
                }
                children.push(element(
                    "site",
                    &[
                        ("name", format!("{name}_grasp")),
                        ("size", "0.001".to_owned()),
                    ],
                    Vec::new(),
                ));
                if let Some(grasp) = object.get("grasp") {
                    let attrs = pose_attrs(grasp)?;
                    children
                        .last_mut()
                        .ok_or_else(|| fail("S1607", "missing grasp site"))?
                        .attrs
                        .extend(
                            attrs
                                .into_iter()
                                .map(|(key, value)| (key.to_owned(), value)),
                        );
                }
                welds.push(element(
                    "weld",
                    &[
                        ("name", format!("{name}_attach")),
                        ("site1", "invar_tool".to_owned()),
                        ("site2", format!("{name}_grasp")),
                        ("active", "false".to_owned()),
                    ],
                    Vec::new(),
                ));
            }
            world.children.push(element("body", &attrs, children));
        }
    }
    if !cell_qpos0.is_empty() {
        for group in tree
            .children
            .iter_mut()
            .filter(|child| child.tag == "keyframe")
        {
            for key in &mut group.children {
                if key.tag != "key" {
                    continue;
                }
                let prefix = attr(key, "qpos")
                    .unwrap_or("")
                    .split_whitespace()
                    .collect::<Vec<_>>();
                if prefix.len() > robot_qpos0.len() {
                    return Err(fail("S1607", "robot key qpos exceeds scalar joint count"));
                }
                let mut qpos = robot_qpos0.clone();
                for (value, original) in qpos.iter_mut().zip(prefix) {
                    *value = original.to_owned();
                }
                qpos.extend(cell_qpos0.iter().cloned());
                key.attrs.insert("qpos".to_owned(), qpos.join(" "));
                if let Some(original) = attr(key, "qvel") {
                    let mut qvel = original
                        .split_whitespace()
                        .map(str::to_owned)
                        .collect::<Vec<_>>();
                    if qvel.len() > robot_qpos0.len() {
                        return Err(fail("S1607", "robot key qvel exceeds scalar joint count"));
                    }
                    qvel.resize(robot_qpos0.len() + 6 * dynamic_count, "0".to_owned());
                    key.attrs.insert("qvel".to_owned(), qvel.join(" "));
                }
            }
        }
    }
    let mut excludes = Vec::new();
    for cell in cells {
        let links = super::super::mount::excluded_links(cell, robot_id);
        if links.is_empty() {
            continue;
        }
        let fixture = cell["mount_surface"]["fixture"]
            .as_str()
            .filter(|fixture| names.contains(*fixture))
            .ok_or_else(|| fail("S1607", "mount surface fixture body is missing"))?;
        for link in links {
            if !names.contains(link) {
                return Err(fail(
                    "S1601",
                    "mount exclusion link is not a selected robot body",
                ));
            }
            excludes.push(element(
                "exclude",
                &[("body1", fixture.to_owned()), ("body2", link.to_owned())],
                Vec::new(),
            ));
        }
    }
    if !excludes.is_empty() {
        if let Some(contact) = tree
            .children
            .iter_mut()
            .find(|child| child.tag == "contact")
        {
            contact.children.extend(excludes);
        } else {
            tree.children.push(element("contact", &[], excludes));
        }
    }
    if !welds.is_empty() {
        if let Some(equality) = tree
            .children
            .iter_mut()
            .find(|child| child.tag == "equality")
        {
            equality.children.extend(welds);
        } else {
            tree.children.push(element("equality", &[], welds));
        }
    }
    Ok(())
}
