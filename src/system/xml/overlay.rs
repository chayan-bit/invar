use super::*;

pub(super) fn quantity_values(values: &Value) -> Option<String> {
    Some(
        values
            .as_array()?
            .iter()
            .map(|value| value["value"].as_str().map(str::to_owned))
            .collect::<Option<Vec<_>>>()?
            .join(" "),
    )
}

pub(super) fn plain_values(values: &Value) -> Option<String> {
    Some(
        values
            .as_array()?
            .iter()
            .map(|value| value.as_str().map(str::to_owned))
            .collect::<Option<Vec<_>>>()?
            .join(" "),
    )
}

pub(super) fn set_attribute(node: &mut Element, key: &str, value: String, default: &str) {
    if node.attrs.contains_key(key) || value != default {
        node.attrs.insert(key.to_owned(), value);
    }
}

pub(crate) fn overlay_robot(tree: &mut Element, robot: &Value) -> Result<(), XmlError> {
    let links = robot["links"]
        .as_array()
        .ok_or_else(|| fail("S1607", "checked robot links are missing"))?;
    let joints = robot["joints"]
        .as_array()
        .ok_or_else(|| fail("S1607", "checked robot joints are missing"))?;
    let actuators = robot["actuators"]
        .as_array()
        .ok_or_else(|| fail("S1607", "checked robot actuators are missing"))?;
    fn visit(
        node: &mut Element,
        links: &[Value],
        joints: &[Value],
        actuators: &[Value],
    ) -> Result<(), XmlError> {
        if node.tag == "body"
            && let Some(name) = attr(node, "name")
            && let Some(link) = links.iter().find(|link| link["name"] == name)
        {
            let origin = &link["origin"];
            set_attribute(
                node,
                "pos",
                quantity_values(&origin["xyz"])
                    .ok_or_else(|| fail("S1607", "invalid checked link pose"))?,
                "0 0 0",
            );
            set_attribute(
                node,
                "quat",
                plain_values(&origin["quat"])
                    .ok_or_else(|| fail("S1607", "invalid checked link quaternion"))?,
                "1 0 0 0",
            );
            let inertial = node
                .children
                .iter_mut()
                .find(|child| child.tag == "inertial")
                .ok_or_else(|| fail("S1607", "source body has no inertial"))?;
            inertial.attrs.insert(
                "mass".to_owned(),
                link["mass"]["value"]
                    .as_str()
                    .ok_or_else(|| fail("S1607", "invalid checked link mass"))?
                    .to_owned(),
            );
            if !link["inertial_origin"].is_null() {
                set_attribute(
                    inertial,
                    "pos",
                    quantity_values(&link["inertial_origin"]["xyz"])
                        .ok_or_else(|| fail("S1607", "invalid inertial origin"))?,
                    "0 0 0",
                );
                set_attribute(
                    inertial,
                    "quat",
                    plain_values(&link["inertial_origin"]["quat"])
                        .ok_or_else(|| fail("S1607", "invalid inertial rotation"))?,
                    "1 0 0 0",
                );
            }
            let inertia = link["inertia"]
                .as_array()
                .ok_or_else(|| fail("S1607", "invalid checked inertia"))?;
            let values = plain_values(&link["inertia"])
                .ok_or_else(|| fail("S1607", "invalid checked inertia"))?;
            if inertial.attrs.contains_key("fullinertia")
                || inertia[3..].iter().any(|value| value != "0")
            {
                inertial.attrs.remove("diaginertia");
                inertial.attrs.insert("fullinertia".to_owned(), values);
            } else {
                inertial.attrs.remove("fullinertia");
                inertial.attrs.insert(
                    "diaginertia".to_owned(),
                    plain_values(&json!(&inertia[..3])).unwrap(),
                );
            }
            let declared = link["geometries"]
                .as_array()
                .ok_or_else(|| fail("S1607", "checked geometry list is missing"))?;
            for (source, geometry) in node
                .children
                .iter_mut()
                .filter(|child| child.tag == "geom")
                .zip(declared)
            {
                let kind = geometry["kind"]
                    .as_str()
                    .ok_or_else(|| fail("S1607", "checked geometry kind is missing"))?;
                if kind == "mesh" {
                    if !source.attrs.contains_key("mesh") {
                        return Err(fail("S1607", "new mesh needs a declared MJCF asset entry"));
                    }
                } else {
                    source.attrs.remove("mesh");
                    source.attrs.insert("type".to_owned(), kind.to_owned());
                    source.attrs.insert(
                        "size".to_owned(),
                        plain_values(&geometry["dimensions"]).ok_or_else(|| {
                            fail("S1607", "checked geometry dimensions are missing")
                        })?,
                    );
                }
                let source_visual =
                    attr(source, "group") == Some("2") || attr(source, "contype") == Some("0");
                let declared_visual = geometry["role"] == "visual";
                if source_visual != declared_visual {
                    let value = if declared_visual { "0" } else { "1" };
                    source.attrs.insert("contype".to_owned(), value.to_owned());
                    source
                        .attrs
                        .insert("conaffinity".to_owned(), value.to_owned());
                    if source.attrs.contains_key("group") {
                        source.attrs.insert(
                            "group".to_owned(),
                            if declared_visual { "2" } else { "3" }.to_owned(),
                        );
                    }
                }
            }
        } else if node.tag == "joint"
            && let Some(name) = attr(node, "name")
            && let Some(joint) = joints.iter().find(|joint| joint["name"] == name)
        {
            set_attribute(
                node,
                "axis",
                plain_values(&joint["axis"])
                    .ok_or_else(|| fail("S1607", "invalid checked joint axis"))?,
                "0 0 1",
            );
            if joint["position"].is_null() {
                node.attrs.remove("range");
                node.attrs.remove("limited");
            } else {
                let lower = joint["position"]["lower"]["value"]
                    .as_str()
                    .ok_or_else(|| fail("S1607", "invalid checked joint lower bound"))?;
                let upper = joint["position"]["upper"]["value"]
                    .as_str()
                    .ok_or_else(|| fail("S1607", "invalid checked joint upper bound"))?;
                node.attrs
                    .insert("range".to_owned(), format!("{lower} {upper}"));
            }
        } else if matches!(
            node.tag.as_str(),
            "general" | "motor" | "position" | "velocity"
        ) && let Some(name) = attr(node, "name")
            && let Some(actuator) = actuators.iter().find(|actuator| actuator["name"] == name)
        {
            let lower = actuator["control"]["lower"]["value"]
                .as_str()
                .ok_or_else(|| fail("S1607", "invalid checked actuator lower bound"))?;
            let upper = actuator["control"]["upper"]["value"]
                .as_str()
                .ok_or_else(|| fail("S1607", "invalid checked actuator upper bound"))?;
            node.attrs
                .insert("ctrlrange".to_owned(), format!("{lower} {upper}"));
            set_attribute(
                node,
                "gear",
                actuator["gear"]
                    .as_str()
                    .ok_or_else(|| fail("S1607", "invalid checked actuator gear"))?
                    .to_owned(),
                "1",
            );
        }
        for child in &mut node.children {
            visit(child, links, joints, actuators)?;
        }
        Ok(())
    }
    visit(tree, links, joints, actuators)?;
    apply_holds(tree, actuators)
}

/// A source-declared actuator `hold` replaces that actuator's entry in the
/// vendored `home` key control vector (the command held at home).
pub(super) fn apply_holds(tree: &mut Element, actuators: &[Value]) -> Result<(), XmlError> {
    let holds = actuators
        .iter()
        .filter(|actuator| !actuator["hold"].is_null())
        .collect::<Vec<_>>();
    if holds.is_empty() {
        return Ok(());
    }
    let order = tree
        .children
        .iter()
        .filter(|child| child.tag == "actuator")
        .flat_map(|group| group.children.iter().filter_map(|a| attr(a, "name")))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let mut homes = tree
        .children
        .iter_mut()
        .filter(|child| child.tag == "keyframe")
        .flat_map(|group| group.children.iter_mut())
        .filter(|key| key.tag == "key" && attr(key, "name") == Some("home"))
        .collect::<Vec<_>>();
    let [home] = homes.as_mut_slice() else {
        return Err(fail("S1607", "actuator hold needs exactly one home key"));
    };
    let mut ctrl = attr(home, "ctrl")
        .unwrap_or("")
        .split_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if ctrl.len() != order.len() {
        return Err(fail(
            "S1607",
            "home key controls do not match the actuators",
        ));
    }
    for actuator in holds {
        let index = order
            .iter()
            .position(|name| actuator["name"] == name.as_str())
            .ok_or_else(|| fail("S1607", "held actuator is missing from the model"))?;
        ctrl[index] = actuator["hold"]["value"]
            .as_str()
            .ok_or_else(|| fail("S1607", "invalid checked actuator hold"))?
            .to_owned();
    }
    home.attrs.insert("ctrl".to_owned(), ctrl.join(" "));
    Ok(())
}
