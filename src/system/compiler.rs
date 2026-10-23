use super::*;

/// Compile one source-backed foreign manifest without requiring a system root.
pub fn compile_manifest(program: Program) -> Result<Value, FrontError> {
    let [item] = program.system_items.as_slice() else {
        return Err(error(
            "S1400",
            "manifest must contain one extern component",
            Span { start: 0, end: 0 },
        ));
    };
    let SystemItemKind::Component {
        name,
        external: true,
        clauses,
    } = &item.kind
    else {
        return Err(error(
            "S1400",
            "manifest must contain one extern component",
            item.span,
        ));
    };
    let manifest = component(item, name, true, clauses, &BTreeSet::new(), true)?;
    let type_probes = manifest
        .ports
        .values()
        .map(|port| (port.ty.clone(), port.span))
        .collect();
    let span = item.span;
    let document = manifest.json;
    validate_port_types(program, type_probes, true, span)?;
    Ok(document)
}

pub fn compile(program: Program) -> Result<SystemIr, FrontError> {
    compile_with_source_dir(program, None)
}

pub fn compile_at(program: Program, source_dir: &Path) -> Result<SystemIr, FrontError> {
    compile_with_source_dir(program, Some(source_dir))
}

pub(super) fn compile_with_source_dir(
    program: Program,
    source_dir: Option<&Path>,
) -> Result<SystemIr, FrontError> {
    if program.system_items.len() > MAX_ITEMS {
        return Err(error(
            "S1009",
            "system item limit exceeded",
            program.system_items[MAX_ITEMS].span,
        ));
    }
    let robot = selected_robot(&program);
    let mut items = expanded_items(&program, source_dir)?;
    for item in &mut items {
        let SystemItemKind::Component {
            external: false,
            clauses,
            ..
        } = &mut item.kind
        else {
            continue;
        };
        for clause in clauses {
            let SystemClause::Port { ty, span, .. } = clause else {
                continue;
            };
            if let Type::Named(name) = ty {
                let legacy = name == "std::robotics::JointPositions<f64,robot_dof>";
                let planner = crate::robotics::generic_parts(name).is_some_and(|(head, args)| {
                    crate::robotics::is_planner_record(head) && args.get(1) == Some(&"robot_dof")
                });
                if !legacy && !planner {
                    continue;
                }
                let count = match robot {
                    Some("menagerie::franka_emika_panda") => 7,
                    Some("menagerie::universal_robots_ur5e") => 6,
                    _ => {
                        return Err(error(
                            "S1001",
                            "robot_dof requires one supported product robot",
                            *span,
                        ));
                    }
                };
                if legacy {
                    *name = format!("std::robotics::JointPositions<f64,{count}>");
                } else if let Some((head, args)) = crate::robotics::generic_parts(name) {
                    let args = args
                        .iter()
                        .enumerate()
                        .map(|(index, value)| {
                            if index == 1 {
                                count.to_string()
                            } else {
                                (*value).to_owned()
                            }
                        })
                        .collect::<Vec<_>>();
                    *name = format!("{head}<{}>", args.join(","));
                }
            }
        }
    }
    let planner_ports = items.iter().any(|item| {
        let SystemItemKind::Component { clauses, .. } = &item.kind else {
            return false;
        };
        clauses.iter().any(|clause| match clause {
            SystemClause::Port {
                ty: Type::Named(name),
                ..
            } => crate::robotics::generic_parts(name)
                .is_some_and(|(head, _)| crate::robotics::is_planner_record(head)),
            _ => false,
        })
    });
    // Robot and cell models: `robots`/`cells` in the IR and product robots
    // that name a declared model.
    let models = items.iter().any(|item| {
        matches!(
            item.kind,
            SystemItemKind::RobotModel(_) | SystemItemKind::CellModel(_)
        )
    });
    // The S1-02 foreign-manifest rules (C layouts, identifiers, manifest
    // relative paths, config and shim) apply once a source uses typed config,
    // input paths, golden tolerances, parameter ranges, planner records or models.
    let manifest_rules = items.iter().any(|item| {
        let SystemItemKind::Component { clauses, .. } = &item.kind else {
            return false;
        };
        clauses.iter().any(|clause| match clause {
            SystemClause::Config(_) | SystemClause::Input(_) => true,
            SystemClause::Golden { tolerances, .. } => !tolerances.is_empty(),
            SystemClause::Param { range: Some(_), .. } => true,
            _ => false,
        })
    }) || planner_ports
        || models;
    // Frame-tree failures report as graph checks (S1200) once a source uses a
    // bus, an identity transform, a license detail or a manifest rule feature.
    let graph_frame_codes = items.iter().any(|item| {
        let SystemItemKind::Component { clauses, .. } = &item.kind else {
            return false;
        };
        clauses.iter().any(|clause| match clause {
            SystemClause::Port { bus, .. } => bus.is_some(),
            SystemClause::IdentityParam { .. } => true,
            SystemClause::License {
                modified,
                replaceable,
                network_exposed,
                grant,
                ..
            } => {
                modified.is_some()
                    || replaceable.is_some()
                    || network_exposed.is_some()
                    || grant.is_some()
            }
            _ => false,
        })
    }) || manifest_rules;
    let mut origins = BTreeMap::new();
    for frame in &program.frames {
        origins.insert(frame.name.clone(), frame.span);
    }
    let mut frame_docs: Vec<_> = program
        .frames
        .iter()
        .map(|frame| json!({"id":frame.name,"name":frame.name,"parent":frame.parent}))
        .collect();
    frame_docs.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    let function_names: BTreeSet<_> = program.functions.iter().map(|f| f.name.clone()).collect();
    let mut components = BTreeMap::new();
    let mut products = BTreeMap::new();
    let mut robots = BTreeMap::new();
    let mut assets = BTreeMap::new();
    let mut robot_warnings = Vec::new();
    let mut cells = BTreeMap::new();
    let mut system = None;
    let mut connections = Vec::new();
    let mut flows = Vec::new();
    for item in &items {
        let module = item.module.as_deref();
        match &item.kind {
            SystemItemKind::ComponentImport { .. } => {
                return Err(error("S1400", "unexpanded component import", item.span));
            }
            SystemItemKind::Component {
                name,
                external,
                clauses,
            } => {
                let id = qualified(module, name);
                origins.insert(id.clone(), item.span);
                if components.contains_key(&id) {
                    return Err(error("S1000", "duplicate component", item.span));
                }
                components.insert(
                    id.clone(),
                    component(
                        item,
                        &id,
                        *external,
                        clauses,
                        &function_names,
                        manifest_rules,
                    )?,
                );
            }
            SystemItemKind::Product { name, clauses } => {
                let id = qualified(module, name);
                origins.insert(id.clone(), item.span);
                if products
                    .insert(id.clone(), product(item, &id, clauses, models)?)
                    .is_some()
                {
                    return Err(error("S1000", "duplicate product", item.span));
                }
            }
            SystemItemKind::System {
                name,
                products,
                instances,
            } => {
                origins.insert(qualified(module, name), item.span);
                if system
                    .replace((item, qualified(module, name), products, instances))
                    .is_some()
                {
                    return Err(error(
                        "S1005",
                        "exactly one system declaration is required",
                        item.span,
                    ));
                }
            }
            SystemItemKind::Connect { source, target, .. } => {
                origins.insert(format!("{source}->{target}"), item.span);
                connections.push(item);
            }
            SystemItemKind::Flow { name, .. } => {
                origins.insert(qualified(module, name), item.span);
                flows.push(item);
            }
            SystemItemKind::RobotModel(model) => {
                let id = qualified(module, &model.name);
                origins.insert(id.clone(), item.span);
                let (robot, files, warnings) =
                    robot::compile_robot(model, &id, item.span, source_dir)?;
                if robots.insert(id.clone(), robot).is_some() {
                    return Err(error("S1600", "duplicate robot declaration", item.span));
                }
                if !files.is_empty() {
                    assets.insert(id, files);
                }
                robot_warnings.extend(warnings.into_iter().map(|message| (message, item.span)));
            }
            SystemItemKind::CellModel(model) => {
                let id = qualified(module, &model.name);
                origins.insert(id.clone(), item.span);
                let mut cell = robot::compile_cell(model, &id, item.span)?;
                cell["robot"] = json!(qualified(module, &model.robot));
                // `get_mut`, not `IndexMut`: indexing would insert a null mount surface.
                let entries = cell
                    .get_mut("mount_surface")
                    .and_then(|surface| surface.get_mut("exclude"))
                    .and_then(Value::as_array_mut);
                for entry in entries.into_iter().flatten() {
                    entry["robot"] =
                        json!(qualified(module, entry["robot"].as_str().unwrap_or("")));
                }
                if cells.insert(id, cell).is_some() {
                    return Err(error("S1600", "duplicate cell declaration", item.span));
                }
            }
        }
    }
    if models {
        if robots.is_empty() {
            return Err(error(
                "S1600",
                "a cell model needs a robot declaration",
                items[0].span,
            ));
        }
        if !frame_docs.iter().any(|frame| frame["id"] == "world") {
            frame_docs.push(json!({"id":"world","name":"world","parent":null}));
        }
        let mut frame_names: BTreeSet<_> = frame_docs
            .iter()
            .filter_map(|frame| frame["id"].as_str().map(str::to_owned))
            .collect();
        for (id, model) in &robots {
            for link in model["links"].as_array().into_iter().flatten() {
                let name = format!("{id}::{}", link["name"].as_str().unwrap_or(""));
                let parent = if link["parent"] == "world" {
                    "world".to_owned()
                } else {
                    format!("{id}::{}", link["parent"].as_str().unwrap_or(""))
                };
                if !frame_names.insert(name.clone()) {
                    return Err(error(
                        "S1600",
                        "robot link frame collides with a declared frame",
                        origins[id],
                    ));
                }
                frame_docs.push(json!({"id":name,"name":name,"parent":parent}));
            }
        }
        frame_docs.sort_by(|left, right| left["id"].as_str().cmp(&right["id"].as_str()));
        for cell in cells.values() {
            let span = origins[cell["id"].as_str().unwrap_or("")];
            if !robots.contains_key(cell["robot"].as_str().unwrap_or("")) {
                return Err(error("S1601", "cell robot does not resolve", span));
            }
            mount::check_mounted(cell, &robots, span)?;
        }
    }
    let (system_item, system_name, system_products, system_instances) =
        system.ok_or_else(|| {
            error(
                "S1005",
                "system declaration is required",
                items
                    .first()
                    .map_or(Span { start: 0, end: 0 }, |item| item.span),
            )
        })?;
    if system_products.len() != 1 {
        return Err(error(
            "S1005",
            "system must name exactly one product",
            system_item.span,
        ));
    }
    let product_name = qualified(system_item.module.as_deref(), &system_products[0]);
    if !products.contains_key(&product_name) {
        return Err(error("S1001", "undefined system product", system_item.span));
    }
    if models {
        let selected = products[&product_name]["robot"].as_str().unwrap_or("");
        let model = robots
            .get(selected)
            .ok_or_else(|| error("S1601", "product robot does not resolve", system_item.span))?;
        robot::check_license(model, &products[&product_name], system_item.span)?;
        if cells.values().any(|cell| {
            cell["objects"]
                .as_array()
                .is_some_and(|objects| objects.iter().any(|object| !object["dynamic"].is_null()))
        }) && model["tool"].is_null()
        {
            return Err(error(
                "S1601",
                "attachable cell needs a selected robot tool",
                system_item.span,
            ));
        }
    }
    let mut instances = BTreeMap::new();
    for (instance, component_name) in system_instances {
        let component_name = qualified(system_item.module.as_deref(), component_name);
        if !components.contains_key(&component_name) {
            return Err(error(
                "S1001",
                format!("undefined component '{component_name}'"),
                system_item.span,
            ));
        }
        if instances.insert(instance.clone(), component_name).is_some() {
            return Err(error(
                "S1000",
                "duplicate system instance",
                system_item.span,
            ));
        }
    }
    for component in components.values() {
        if !component.external
            && let Some(target) = component.json["implementation"]["target"].as_str()
            && component.json["implementation"]["kind"] == "extern"
            && target != "invar::s1_06_reach_place_v1"
            && target != substep_admission::TARGET
            && !components.get(target).is_some_and(|found| found.external)
        {
            return Err(error(
                "S1001",
                "undefined extern component manifest",
                component.item.span,
            ));
        }
    }

    let mut type_probes = Vec::new();
    for component in components.values() {
        for port in component.ports.values() {
            type_probes.push((port.ty.clone(), port.span));
        }
    }
    for item in &items {
        if let SystemItemKind::Component { clauses, .. } = &item.kind {
            for clause in clauses {
                if let SystemClause::IdentityParam { ty, span, .. } = clause {
                    type_probes.push((ty.clone(), *span));
                }
            }
        }
    }
    validate_port_types(program, type_probes, graph_frame_codes, system_item.span)?;

    let mut connected = BTreeSet::new();
    let mut connection_docs = Vec::new();
    for item in connections {
        let SystemItemKind::Connect {
            source,
            target,
            policy,
        } = &item.kind
        else {
            unreachable!()
        };
        let resolve = |path: &str| -> Result<&Port, FrontError> {
            let (instance, port_name) = path
                .split_once('.')
                .ok_or_else(|| error("S1001", "invalid port path", item.span))?;
            let component_name = instances
                .get(instance)
                .ok_or_else(|| error("S1001", "undefined instance", item.span))?;
            let component = components
                .get(component_name)
                .ok_or_else(|| error("S1001", "undefined component", item.span))?;
            let port = component
                .ports
                .get(port_name)
                .ok_or_else(|| error("S1001", "undefined port", item.span))?;
            Ok(port)
        };
        let source_port = resolve(source)?;
        let target_port = resolve(target)?;
        if source_port.direction != "out" || target_port.direction != "in" {
            return Err(error("S1003", "connect requires out -> in", item.span));
        }
        if source_port.ty != target_port.ty {
            return Err(error(
                "S1002",
                "connected port types, units, or frames differ",
                item.span,
            ));
        }
        if !connected.insert(target.clone()) {
            return Err(error(
                "S1004",
                "input port has multiple producers",
                item.span,
            ));
        }
        connected.insert(source.clone());
        if policy != "sample-and-hold"
            && !policy.starts_with("queue(")
            && !policy.starts_with("decimate(")
        {
            return Err(error("S1000", "unknown connection policy", item.span));
        }
        if let Some(count) = policy
            .strip_prefix("queue(")
            .or_else(|| policy.strip_prefix("decimate("))
            && !count
                .strip_suffix(')')
                .and_then(|n| n.parse::<u32>().ok())
                .is_some_and(|n| (1..=1024).contains(&n))
        {
            return Err(error("S1000", "invalid policy capacity", item.span));
        }
        connection_docs.push(json!({"id":format!("{source}->{target}"),"source":source,"target":target,"policy":policy}));
    }
    connection_docs.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    let mut warnings = Vec::new();
    for (instance, component_name) in &instances {
        let component = &components[component_name];
        for (port_name, port) in &component.ports {
            let path = format!("{instance}.{port_name}");
            if !connected.contains(&path) && port.direction == "in" && !port.optional {
                return Err(error(
                    "S1010",
                    format!("input port '{path}' has no producer"),
                    port.span,
                ));
            }
            if !connected.contains(&path) && port.direction == "out" {
                warnings.push((format!("output port '{path}' has no consumer"), port.span));
            }
        }
    }
    let mut flow_docs = Vec::new();
    let mut flow_names = BTreeSet::new();
    for item in flows {
        let SystemItemKind::Flow {
            name,
            ports,
            within,
        } = &item.kind
        else {
            unreachable!()
        };
        if ports.len() < 2 {
            return Err(error("S1000", "flow needs at least two ports", item.span));
        }
        for path in ports {
            let (instance, port) = path
                .split_once('.')
                .ok_or_else(|| error("S1001", "invalid flow port", item.span))?;
            let component = instances
                .get(instance)
                .and_then(|id| components.get(id))
                .ok_or_else(|| error("S1001", "undefined flow instance", item.span))?;
            if !component.ports.contains_key(port) {
                return Err(error("S1001", "undefined flow port", item.span));
            }
        }
        let id = qualified(item.module.as_deref(), name);
        if !flow_names.insert(id.clone()) {
            return Err(error("S1000", "duplicate flow", item.span));
        }
        flow_docs.push(json!({"id":id,"name":name,"ports":ports,"within":duration(within)?}));
    }
    flow_docs.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    let instance_docs: Vec<_> = instances.iter().map(|(name, component)| json!({"id":format!("{system_name}::{name}"),"name":name,"component":component})).collect();
    let mut component_docs = Vec::new();
    let mut extern_docs = Vec::new();
    for component in components.values() {
        if component.external {
            extern_docs.push(component.json.clone());
        } else {
            component_docs.push(component.json.clone());
        }
    }
    let mut document = json!({"schema":"invar-system-v12","version":12,"system":{"id":system_name,"name":system_name,"product":product_name,"instances":instance_docs},"frames":frame_docs,"components":component_docs,"extern_components":extern_docs,"connections":connection_docs,"flows":flow_docs,"products":products.values().cloned().collect::<Vec<_>>(),"planner_contract":"invar-planner-wire-v1","planner_frame_policy":"request-consumer-v1"});
    if models {
        document["robots"] = json!(robots.values().collect::<Vec<_>>());
        document["cells"] = json!(cells.values().collect::<Vec<_>>());
    }
    if document.to_string().len() > MAX_IR_BYTES {
        return Err(error(
            "S1009",
            "system IR size limit exceeded",
            system_item.span,
        ));
    }
    let ir = SystemIr {
        document,
        warnings,
        robot_warnings,
        origins,
        checked_source: true,
        assets,
    };
    substep_admission::validate(&ir)
        .map_err(|message| error("S1504", message, system_item.span))?;
    Ok(ir)
}
