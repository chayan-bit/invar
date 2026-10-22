use super::*;

pub(crate) fn asset_files(
    tree: &Element,
    model_dir: &Path,
) -> Result<BTreeMap<String, String>, XmlError> {
    let compiler = tree.children.iter().find(|child| child.tag == "compiler");
    let mut files = BTreeMap::new();
    for asset in tree.children.iter().filter(|child| child.tag == "asset") {
        for child in &asset.children {
            if let Some(file) = attr(child, "file") {
                let folder = match child.tag.as_str() {
                    "mesh" => compiler
                        .and_then(|compiler| attr(compiler, "meshdir"))
                        .unwrap_or(""),
                    "texture" => compiler
                        .and_then(|compiler| attr(compiler, "texturedir"))
                        .unwrap_or(""),
                    _ => return Err(fail("S1607", "unsupported file-backed MJCF asset")),
                };
                let path = if folder.is_empty() {
                    file.to_owned()
                } else {
                    format!("{folder}/{file}")
                };
                files.insert(path.clone(), hash(&model_file(model_dir, &path)?));
            }
        }
    }
    Ok(files)
}

pub(crate) fn model_file(root: &Path, relative_path: &str) -> Result<Vec<u8>, XmlError> {
    if !relative(relative_path) {
        return Err(fail("S1605", "MJCF asset path is not relative"));
    }
    let canonical_root = root
        .canonicalize()
        .map_err(|_| fail("S1605", "MJCF asset root is missing"))?;
    let path = root
        .join(relative_path)
        .canonicalize()
        .map_err(|_| fail("S1605", "MJCF mesh or texture is missing"))?;
    if !path.starts_with(canonical_root) {
        return Err(fail("S1605", "MJCF asset escapes model root"));
    }
    let bytes =
        std::fs::read(path).map_err(|_| fail("S1605", "cannot read MJCF mesh or texture"))?;
    if bytes.len() > 16_777_216 {
        return Err(fail("S1605", "MJCF mesh or texture exceeds 16 MiB"));
    }
    Ok(bytes)
}

pub(super) fn import_body(
    body: &Element,
    parent: &str,
    mesh_files: &BTreeMap<String, String>,
    model_dir: &Path,
    out: &mut String,
) -> Result<(), XmlError> {
    let name = identifier(required(body, "name")?)?;
    let inertial = body
        .children
        .iter()
        .find(|child| child.tag == "inertial")
        .ok_or_else(|| fail("S1607", format!("body {name} needs explicit inertial")))?;
    let inertia = if let Some(full) = attr(inertial, "fullinertia") {
        tuple(full, 6)?
    } else if let Some(diagonal) = attr(inertial, "diaginertia") {
        let values: Vec<_> = diagonal.split_whitespace().collect();
        if values.len() != 3 {
            return Err(fail("S1607", "diaginertia needs three values"));
        }
        format!("({}, {}, {}, 0, 0, 0)", values[0], values[1], values[2])
    } else {
        return Err(fail("S1607", format!("body {name} needs explicit inertia")));
    };
    out.push_str(&format!("    link {name} {{\n        parent {parent};\n        origin {};\n        mass {} kg;\n        inertial_origin {};\n        inertia {inertia} kg*m*m;\n",
        pose(body,"pos")?, required(inertial,"mass")?, pose(inertial,"pos")?));
    for geom in body.children.iter().filter(|child| child.tag == "geom") {
        let role = if attr(geom, "group") == Some("2") || attr(geom, "contype") == Some("0") {
            "visual"
        } else {
            "collision"
        };
        let kind = attr(geom, "type").unwrap_or(if geom.attrs.contains_key("mesh") {
            "mesh"
        } else {
            "sphere"
        });
        if kind == "mesh" {
            let mesh = required(geom, "mesh")?;
            let file = mesh_files
                .get(mesh)
                .ok_or_else(|| fail("S1605", format!("mesh {mesh} is missing from asset table")))?;
            let digest = hash(&model_file(model_dir, file)?);
            out.push_str(&format!(
                "        {role} mesh \"{file}\" sha256 \"{digest}\";\n"
            ));
        } else if matches!(kind, "box" | "sphere" | "cylinder" | "capsule") {
            let count = match kind {
                "box" => 3,
                "sphere" => 1,
                _ => 2,
            };
            let size = required(geom, "size")?;
            let values: Vec<_> = size.split_whitespace().collect();
            if values.len() < count {
                return Err(fail("S1607", "MJCF geometry size is incomplete"));
            }
            let dimensions = if count == 1 {
                values[0].to_owned()
            } else {
                tuple(&values[..count].join(" "), count)?
            };
            out.push_str(&format!("        {role} {kind} {dimensions} m;\n"));
        } else {
            return Err(fail("S1607", format!("unsupported robot geometry: {kind}")));
        }
    }
    if body.children.iter().any(|child| child.tag == "camera") {
        return Err(fail(
            "S1607",
            "MJCF camera pose and intrinsics cannot be imported losslessly",
        ));
    }
    out.push_str("    }\n");
    for joint in body.children.iter().filter(|child| child.tag == "joint") {
        let joint_name = identifier(required(joint, "name")?)?;
        let kind = match attr(joint, "type").unwrap_or("hinge") {
            "hinge" => "revolute",
            "slide" => "prismatic",
            _ => return Err(fail("S1607", "unsupported MJCF joint kind")),
        };
        let unit = if kind == "prismatic" { "m" } else { "rad" };
        let range = if let Some(range) = attr(joint, "range") {
            let values: Vec<_> = range.split_whitespace().collect();
            if values.len() != 2 {
                return Err(fail("S1607", "joint range needs two values"));
            }
            format!("{} {unit} .. {} {unit}", values[0], values[1])
        } else {
            "unbounded".to_owned()
        };
        out.push_str(&format!("    joint {joint_name} {{\n        kind {kind};\n        parent {parent};\n        child {name};\n        axis {};\n        position {range};\n        velocity unbounded;\n        effort unbounded;\n    }}\n",
            tuple(attr(joint,"axis").unwrap_or("0 0 1"),3)?));
    }
    for child in body.children.iter().filter(|child| child.tag == "body") {
        import_body(child, name, mesh_files, model_dir, out)?;
    }
    Ok(())
}

pub(crate) fn import_mjcf(path: &Path) -> Result<(String, Vec<String>), XmlError> {
    let mut tree = load_mjcf(path)?;
    let warnings = validate_supported(&mut tree)?;
    let model_dir = path
        .parent()
        .ok_or_else(|| fail("S1605", "MJCF model directory is missing"))?;
    let license_text = std::fs::read_to_string(model_dir.join("LICENSE"))
        .map_err(|_| fail("S1606", "MJCF model LICENSE is missing"))?;
    let license = if license_text.contains("Apache License") && license_text.contains("Version 2.0")
    {
        "Apache-2.0"
    } else if license_text.contains("Redistribution and use in source and binary forms")
        && license_text.contains("Neither the name")
    {
        "BSD-3-Clause"
    } else {
        return Err(fail("S1606", "unsupported or ambiguous MJCF asset license"));
    };
    let current =
        std::env::current_dir().map_err(|_| fail("S1605", "current directory is unavailable"))?;
    let full_path = path
        .canonicalize()
        .map_err(|_| fail("S1605", "MJCF source file is missing"))?;
    let relative_source = full_path
        .strip_prefix(&current)
        .map_err(|_| fail("S1605", "MJCF source must be below current directory"))?;
    let source_path = relative_source
        .to_str()
        .ok_or_else(|| fail("S1605", "MJCF path must be UTF-8"))?;
    if !relative(source_path) {
        return Err(fail("S1605", "MJCF path is not source-relative"));
    }
    let raw = std::fs::read(&full_path).map_err(|_| fail("S1605", "cannot read MJCF source"))?;
    let name = path
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .ok_or_else(|| fail("S1600", "MJCF model name is missing"))?;
    let name = identifier(name)?;
    let meshdir = tree
        .children
        .iter()
        .find(|child| child.tag == "compiler")
        .and_then(|child| attr(child, "meshdir"))
        .unwrap_or("");
    let mut mesh_files = BTreeMap::new();
    for asset in tree.children.iter().filter(|child| child.tag == "asset") {
        for mesh in asset.children.iter().filter(|child| child.tag == "mesh") {
            let file = required(mesh, "file")?;
            let stem = Path::new(file)
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or_else(|| fail("S1605", "mesh name is not UTF-8"))?;
            let mesh_name = attr(mesh, "name").unwrap_or(stem);
            let relative_file = if meshdir.is_empty() {
                file.to_owned()
            } else {
                format!("{meshdir}/{file}")
            };
            model_file(model_dir, &relative_file)?;
            mesh_files.insert(mesh_name.to_owned(), relative_file);
        }
    }
    let mut out = format!(
        "robot {name} {{\n    license \"{license}\";\n    asset mjcf \"{source_path}\" sha256 \"{}\";\n",
        hash(&raw)
    );
    for world in tree
        .children
        .iter()
        .filter(|child| child.tag == "worldbody")
    {
        for body in world.children.iter().filter(|child| child.tag == "body") {
            import_body(body, "world", &mesh_files, model_dir, &mut out)?;
        }
    }
    fn joint_kinds(node: &Element, kinds: &mut BTreeMap<String, bool>) {
        if node.tag == "joint"
            && let Some(name) = attr(node, "name")
        {
            kinds.insert(name.to_owned(), attr(node, "type") == Some("slide"));
        }
        for child in &node.children {
            joint_kinds(child, kinds);
        }
    }
    let mut joint_kinds_by_name = BTreeMap::new();
    for world in tree
        .children
        .iter()
        .filter(|child| child.tag == "worldbody")
    {
        joint_kinds(world, &mut joint_kinds_by_name);
    }
    for actuator in tree.children.iter().filter(|child| child.tag == "actuator") {
        for item in &actuator.children {
            let mode = match item.tag.as_str() {
                "position" => "position",
                "velocity" => "velocity",
                "motor" => "motor-torque",
                "general" => "position",
                _ => return Err(fail("S1607", "unsupported MJCF actuator")),
            };
            let name = identifier(required(item, "name")?)?;
            let (target_kind, target) = if let Some(joint) = attr(item, "joint") {
                ("joint", identifier(joint)?)
            } else if let Some(tendon) = attr(item, "tendon") {
                ("tendon", identifier(tendon)?)
            } else {
                return Err(fail("S1607", "actuator target is missing"));
            };
            let prismatic = target_kind == "joint"
                && *joint_kinds_by_name
                    .get(target)
                    .ok_or_else(|| fail("S1601", "MJCF actuator joint is missing"))?;
            let unit = match (target_kind, mode, prismatic) {
                ("tendon", _, _) => "one",
                (_, "position", true) => "m",
                (_, "position", false) => "rad",
                (_, "velocity", true) => "m/s",
                (_, "velocity", false) => "rad/s",
                (_, _, true) => "N",
                _ => "N*m",
            };
            let range = required(item, "ctrlrange")?
                .split_whitespace()
                .collect::<Vec<_>>();
            if range.len() != 2 {
                return Err(fail("S1607", "actuator control range needs two values"));
            }
            let gear = attr(item, "gear").unwrap_or("1");
            if gear.split_whitespace().count() != 1 {
                return Err(fail("S1607", "multi-axis actuator gear is unsupported"));
            }
            out.push_str(&format!("    actuator {name} {{\n        mode {mode};\n        {target_kind} {target};\n        gear {gear};\n        control {} {unit} .. {} {unit};\n    }}\n",range[0],range[1]));
        }
    }
    for sensors in tree.children.iter().filter(|child| child.tag == "sensor") {
        for sensor in &sensors.children {
            let kind = match sensor.tag.as_str() {
                "jointpos" => "joint-position",
                "jointvel" => "joint-velocity",
                _ => {
                    return Err(fail(
                        "S1607",
                        format!(
                            "unsupported MJCF sensor <{}> affects observations",
                            sensor.tag
                        ),
                    ));
                }
            };
            let name = identifier(required(sensor, "name")?)?;
            let source = identifier(required(sensor, "joint")?)?;
            out.push_str(&format!(
                "    sensor {name} {{\n        kind {kind};\n        source {source};\n    }}\n"
            ));
        }
    }
    out.push_str("}\n\n");
    out.push_str(&format!("product Imported {{\n    distribution internal;\n    commercial false;\n    robot {name};\n    capacity cpu_cores = 1, memory = 1 GiB;\n}}\nsystem imported for Imported {{}}\n"));
    Ok((out, warnings))
}
