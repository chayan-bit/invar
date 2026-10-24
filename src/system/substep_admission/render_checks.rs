//! Rendered-model checks: unsupported elements, geometry classes and contact visibility.

use super::*;

/// Rendered-model checks independent of the source layout: unsupported
/// elements, including `<frame>`/`<attach>` nesting that the geom mapping does
/// not traverse; exactly one `<option>` with only approved attributes and no
/// children (so a `<flag>` cannot disable contact or other physics).
pub(super) fn check_rendered(model: &roxmltree::Document<'_>) -> Result<(), Fault> {
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

pub(super) fn geometry_classes(
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
pub(super) fn check_contacts(
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
