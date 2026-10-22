use super::*;

pub(crate) fn verify_robot(tree: &Element, robot: &Value) -> Result<(), XmlError> {
    let links = robot["links"]
        .as_array()
        .ok_or_else(|| fail("S1607", "checked links are missing"))?;
    let joints = robot["joints"]
        .as_array()
        .ok_or_else(|| fail("S1607", "checked joints are missing"))?;
    let actuators = robot["actuators"]
        .as_array()
        .ok_or_else(|| fail("S1607", "checked actuators are missing"))?;
    let sensors = robot["sensors"]
        .as_array()
        .ok_or_else(|| fail("S1607", "checked sensors are missing"))?;
    let meshdir = tree
        .children
        .iter()
        .find(|child| child.tag == "compiler")
        .and_then(|child| attr(child, "meshdir"))
        .unwrap_or("");
    let mut meshes = BTreeMap::new();
    for asset in tree.children.iter().filter(|child| child.tag == "asset") {
        for mesh in asset.children.iter().filter(|mesh| mesh.tag == "mesh") {
            let file = required(mesh, "file")?;
            let stem = Path::new(file)
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or_else(|| fail("S1605", "MJCF mesh filename is invalid"))?;
            let name = attr(mesh, "name").unwrap_or(stem);
            meshes.insert(
                name.to_owned(),
                if meshdir.is_empty() {
                    file.to_owned()
                } else {
                    format!("{meshdir}/{file}")
                },
            );
        }
    }
    let mut bodies_seen = BTreeSet::new();
    let mut joints_seen = BTreeSet::new();
    fn walk(
        node: &Element,
        parent: &str,
        links: &[Value],
        joints: &[Value],
        meshes: &BTreeMap<String, String>,
        bodies_seen: &mut BTreeSet<String>,
        joints_seen: &mut BTreeSet<String>,
    ) -> Result<(), XmlError> {
        let next_parent = if node.tag == "body" {
            let name = required(node, "name")?;
            let link = links
                .iter()
                .find(|link| link["name"] == name)
                .ok_or_else(|| fail("S1601", format!("MJCF body {name} has no typed link")))?;
            if link["parent"] != parent || !bodies_seen.insert(name.to_owned()) {
                return Err(fail(
                    "S1601",
                    format!("MJCF body {name} differs from typed frame tree"),
                ));
            }
            if node
                .children
                .iter()
                .filter(|child| child.tag == "geom")
                .count()
                != link["geometries"].as_array().map_or(0, Vec::len)
            {
                return Err(fail(
                    "S1607",
                    format!("MJCF body {name} geometry count differs from declaration"),
                ));
            }
            for (source, geometry) in node
                .children
                .iter()
                .filter(|child| child.tag == "geom")
                .zip(link["geometries"].as_array().unwrap())
            {
                if geometry["kind"] == "mesh" {
                    let source_mesh = required(source, "mesh")?;
                    let file = meshes
                        .get(source_mesh)
                        .ok_or_else(|| fail("S1605", "MJCF mesh reference is missing"))?;
                    if geometry["path"] != *file {
                        return Err(fail("S1607", "typed mesh differs from MJCF asset entry"));
                    }
                }
            }
            name
        } else {
            parent
        };
        if node.tag == "joint" && parent != "world" {
            let name = required(node, "name")?;
            let joint = joints
                .iter()
                .find(|joint| joint["name"] == name)
                .ok_or_else(|| {
                    fail(
                        "S1601",
                        format!("MJCF joint {name} has no typed declaration"),
                    )
                })?;
            if joint["child"] != parent || !joints_seen.insert(name.to_owned()) {
                return Err(fail(
                    "S1601",
                    format!("MJCF joint {name} differs from typed tree"),
                ));
            }
        }
        for child in &node.children {
            walk(
                child,
                next_parent,
                links,
                joints,
                meshes,
                bodies_seen,
                joints_seen,
            )?;
        }
        Ok(())
    }
    for world in tree
        .children
        .iter()
        .filter(|child| child.tag == "worldbody")
    {
        for child in &world.children {
            walk(
                child,
                "world",
                links,
                joints,
                &meshes,
                &mut bodies_seen,
                &mut joints_seen,
            )?;
        }
    }
    if bodies_seen.len() != links.len() || joints_seen.len() != joints.len() {
        return Err(fail("S1601", "typed robot omits source bodies or joints"));
    }
    let source_actuators = tree
        .children
        .iter()
        .filter(|child| child.tag == "actuator")
        .flat_map(|child| &child.children)
        .collect::<Vec<_>>();
    if source_actuators.len() != actuators.len() {
        return Err(fail(
            "S1601",
            "typed robot actuator count differs from source",
        ));
    }
    for source in source_actuators {
        let name = required(source, "name")?;
        let target_kind = if source.attrs.contains_key("joint") {
            "joint"
        } else if source.attrs.contains_key("tendon") {
            "tendon"
        } else {
            return Err(fail("S1601", "MJCF actuator target is missing"));
        };
        let target = required(source, target_kind)?;
        if !actuators.iter().any(|actuator| {
            actuator["name"] == name
                && actuator["target_kind"] == target_kind
                && actuator["target"] == target
        }) {
            return Err(fail(
                "S1601",
                format!("MJCF actuator {name} differs from declaration"),
            ));
        }
    }
    let source_sensors = tree
        .children
        .iter()
        .filter(|child| child.tag == "sensor")
        .flat_map(|child| &child.children)
        .collect::<Vec<_>>();
    if source_sensors.len() != sensors.len() {
        return Err(fail("S1601", "typed sensor count differs from MJCF source"));
    }
    for source in source_sensors {
        let name = required(source, "name")?;
        let kind = match source.tag.as_str() {
            "jointpos" => "joint-position",
            "jointvel" => "joint-velocity",
            _ => {
                return Err(fail(
                    "S1607",
                    "unsupported source sensor affects observations",
                ));
            }
        };
        let target = required(source, "joint")?;
        if !sensors.iter().any(|sensor| {
            sensor["name"] == name && sensor["kind"] == kind && sensor["source"] == target
        }) {
            return Err(fail(
                "S1601",
                format!("MJCF sensor {name} differs from declaration"),
            ));
        }
    }
    Ok(())
}
