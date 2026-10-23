use super::*;

pub(super) fn component<'a>(
    item: &'a SystemItem,
    name: &str,
    external: bool,
    clauses: &'a [SystemClause],
    functions: &BTreeSet<String>,
    manifest_rules: bool,
) -> Result<Component<'a>, FrontError> {
    let mut ports = BTreeMap::new();
    let mut port_docs = Vec::new();
    let mut params = BTreeMap::new();
    let mut resource_doc = json!({});
    let mut resource_seen = false;
    let mut license = None;
    let mut isolation = None;
    let mut wcet = None;
    let mut deadline = None;
    let mut implementation = None;
    let mut abi = None;
    let mut lifecycle = None;
    let mut golden = Vec::new();
    let mut config = None;
    let mut inputs = BTreeSet::new();
    let mut properties = Vec::new();
    let mut shim = None;
    for clause in clauses {
        match clause {
            SystemClause::Port {
                direction,
                name,
                ty,
                rate,
                latency,
                optional,
                layout,
                bus,
                span,
            } => {
                if !matches!(direction.as_str(), "in" | "out") {
                    return Err(error("S1003", "port direction must be in or out", *span));
                }
                if ports
                    .insert(
                        name.clone(),
                        Port {
                            direction: direction.clone(),
                            ty: ty.clone(),
                            optional: *optional,
                            span: *span,
                        },
                    )
                    .is_some()
                {
                    return Err(error("S1000", format!("duplicate port '{name}'"), *span));
                }
                if external && layout.is_none() {
                    return Err(error("S1000", "extern port needs a C layout", *span));
                }
                if external
                    && manifest_rules
                    && !layout
                        .as_deref()
                        .is_some_and(|layout| foreign_port_layout(ty, layout))
                {
                    return Err(error(
                        "S1400",
                        "foreign port type and C layout disagree",
                        *span,
                    ));
                }
                let rate = positive(rate, Some("Hz"))?;
                let latency = latency.as_ref().map(duration).transpose()?;
                port_docs.push(json!({"name":name,"direction":direction,"type":type_name(ty),"rate":rate,"latency":latency,"optional":optional,"layout":layout,"bus":bus}));
            }
            SystemClause::Param {
                name,
                ty,
                value,
                range,
                span,
            } => {
                let expected = type_name(ty);
                let Some(unit) = expected
                    .strip_prefix("Quantity<dec,")
                    .and_then(|s| s.strip_suffix('>'))
                else {
                    return Err(error("S1006", "parameter must use Quantity<dec,U>", *span));
                };
                let actual_unit = if let Some((left, right)) = value.unit.split_once('/') {
                    format!("Div<{left},{right}>")
                } else if let Some((left, right)) = value.unit.split_once('*') {
                    format!("Mul<{left},{right}>")
                } else {
                    value.unit.clone()
                };
                if unit != actual_unit || !crate::check::system_unit_supported(unit) {
                    return Err(error(
                        "S1006",
                        "parameter default unit differs from its type",
                        *span,
                    ));
                }
                let mut parameter = quantity(value)?;
                if let Some((min, max)) = range {
                    if min.unit != value.unit || max.unit != value.unit {
                        return Err(error(
                            "S1500",
                            "parameter range unit differs from default",
                            *span,
                        ));
                    }
                    let min_value = decimal(&min.value, min.span)?;
                    let max_value = decimal(&max.value, max.span)?;
                    let default_value = decimal(&value.value, value.span)?;
                    let (Some(min_number), Some(max_number), Some(default_number)) = (
                        graph::decimal(&min_value),
                        graph::decimal(&max_value),
                        graph::decimal(&default_value),
                    ) else {
                        return Err(error("S1500", "invalid parameter range", *span));
                    };
                    if min_number > max_number
                        || default_number < min_number
                        || default_number > max_number
                    {
                        return Err(error(
                            "S1500",
                            "parameter default is outside inclusive range",
                            *span,
                        ));
                    }
                    parameter["range"] = json!({"min":min_value,"max":max_value});
                }
                if params.insert(name.clone(), parameter).is_some() {
                    return Err(error("S1000", "duplicate parameter", *span));
                }
            }
            SystemClause::IdentityParam { name, ty, span } => {
                let expected = type_name(ty);
                let Some(("Transform", args)) = crate::robotics::generic_parts(&expected) else {
                    return Err(error(
                        "S1203",
                        "identity constant requires Transform<Child,Parent>",
                        *span,
                    ));
                };
                let [child, parent] = args.as_slice() else {
                    return Err(error(
                        "S1203",
                        "identity constant requires two frames",
                        *span,
                    ));
                };
                let value = json!({"kind":"identity-transform","type":expected,"from":child,"to":parent,"translation":["0","0","0"],"rotation_wxyz":["1","0","0","0"]});
                if params.insert(name.clone(), value).is_some() {
                    return Err(error("S1000", "duplicate parameter", *span));
                }
            }
            SystemClause::Resource(values) => {
                if resource_seen {
                    return Err(error("S1000", "duplicate resource clause", item.span));
                }
                resource_seen = true;
                resource_doc = resources(values)?;
            }
            SystemClause::License {
                expression,
                link,
                modified,
                replaceable,
                network_exposed,
                grant,
                span,
            } => {
                if license.is_some() {
                    return Err(error("S1000", "duplicate license clause", *span));
                }
                if expression.len() > 4096 {
                    return Err(error("S1009", "SPDX expression size limit exceeded", *span));
                }
                if spdx::Expression::parse(expression).is_err() {
                    return Err(error("S1007", "invalid SPDX expression", *span));
                }
                if let Some(link) = link
                    && !matches!(link.as_str(), "static" | "dynamic" | "process")
                {
                    return Err(error("S1000", "invalid link mode", *span));
                }
                license = Some(
                    json!({"expression":expression,"link":link,"modified":modified,"replaceable":replaceable,"network_exposed":network_exposed,"grant":grant}),
                );
            }
            SystemClause::Isolation(value) => {
                if isolation.is_some() {
                    return Err(error("S1000", "duplicate isolation clause", item.span));
                }
                if !matches!(
                    value.as_str(),
                    "in-process" | "separate-process" | "separate-partition"
                ) {
                    return Err(error("S1000", "invalid isolation mode", item.span));
                }
                isolation = Some(value.clone());
            }
            SystemClause::Wcet(value) => {
                if wcet.is_some() {
                    return Err(error("S1000", "duplicate wcet clause", value.span));
                }
                wcet = Some(duration(value)?);
            }
            SystemClause::Deadline(value) => {
                if deadline.is_some() {
                    return Err(error("S1000", "duplicate deadline clause", value.span));
                }
                deadline = Some(duration(value)?);
            }
            SystemClause::Impl { kind, target } => {
                if implementation.is_some() {
                    return Err(error("S1000", "duplicate implementation clause", item.span));
                }
                if external || !matches!(kind.as_str(), "extern" | "invar") {
                    return Err(error(
                        "S1000",
                        "invalid component implementation",
                        item.span,
                    ));
                }
                let target = qualified(item.module.as_deref(), target);
                if kind == "invar" && !functions.contains(&target) {
                    return Err(error(
                        "S1001",
                        "undefined Invar implementation function",
                        item.span,
                    ));
                }
                implementation = Some(
                    json!({"kind":kind,"target":target,"link":if kind == "invar" {"static"} else {"manifest"}}),
                );
            }
            SystemClause::Abi { kind, prefix } => {
                if abi.is_some() {
                    return Err(error("S1000", "duplicate ABI clause", item.span));
                }
                if !external || kind != "c" || prefix.is_empty() {
                    return Err(error("S1000", "invalid extern C ABI", item.span));
                }
                if manifest_rules && !c_identifier(prefix) {
                    return Err(error(
                        "S1400",
                        "foreign ABI prefix is not a bounded C identifier",
                        item.span,
                    ));
                }
                abi = Some(json!({"kind":"c","prefix":prefix}));
            }
            SystemClause::Lifecycle(calls) => {
                if lifecycle.is_some() {
                    return Err(error("S1000", "duplicate lifecycle clause", item.span));
                }
                if calls != &["init", "configure", "step", "shutdown"] {
                    return Err(error(
                        "S1000",
                        "lifecycle must be init, configure, step, shutdown",
                        item.span,
                    ));
                }
                lifecycle = Some(json!(calls));
            }
            SystemClause::Config(layout) => {
                if !external || config.is_some() {
                    return Err(error(
                        "S1400",
                        "invalid or duplicate foreign config",
                        item.span,
                    ));
                }
                config =
                    Some(foreign_config_layout(layout).ok_or_else(|| {
                        error("S1400", "unknown foreign config C layout", item.span)
                    })?);
            }
            SystemClause::Input(path) => {
                if !external
                    || !manifest_dependency_path(path)
                    || !inputs.insert(path.clone())
                    || inputs.len() > 64
                {
                    return Err(error(
                        "S1400",
                        "invalid or duplicate manifest input path",
                        item.span,
                    ));
                }
            }
            SystemClause::Golden { path, tolerances } => {
                if manifest_rules && !manifest_relative_file(path) {
                    return Err(error(
                        "S1400",
                        "golden path must be relative to manifest",
                        item.span,
                    ));
                }
                let tolerances = tolerances
                    .iter()
                    .map(|value| {
                        let value_doc = quantity(value)?;
                        let number = value_doc["value"].as_str().unwrap_or("");
                        if number.starts_with('-') || !matches!(value.unit.as_str(), "m" | "rad") {
                            return Err(error("S1400", "invalid absolute tolerance", value.span));
                        }
                        Ok(value_doc)
                    })
                    .collect::<Result<Vec<_>, FrontError>>()?;
                golden.push((path.clone(), tolerances));
            }
            SystemClause::Property(value) => properties.push(value.clone()),
            SystemClause::Shim(value) => {
                if shim.is_some() {
                    return Err(error("S1000", "duplicate shim clause", item.span));
                }
                if manifest_rules && !manifest_relative_file(value) {
                    return Err(error(
                        "S1400",
                        "shim path must be relative to manifest",
                        item.span,
                    ));
                }
                shim = Some(value.clone());
            }
            _ => {
                return Err(error(
                    "S1000",
                    "clause is not allowed in a component",
                    item.span,
                ));
            }
        }
    }
    if license.is_none()
        || isolation.is_none()
        || (external && (abi.is_none() || lifecycle.is_none()))
        || (!external && implementation.is_none())
    {
        return Err(error(
            "S1000",
            "component is missing required clauses",
            item.span,
        ));
    }
    if manifest_rules && external {
        if config.is_none() || shim.is_none() {
            return Err(error(
                "S1400",
                "foreign v3 manifest needs config and shim",
                item.span,
            ));
        }
        for property in &properties {
            checked_foreign_property(property, &ports, item.span)?;
        }
        if config
            .as_ref()
            .is_some_and(|value| value["layout"] == "inv_planner_config_v1")
        {
            if isolation.as_deref() != Some("separate-process") {
                return Err(error(
                    "S1400",
                    "planner-v1 requires separate-process isolation",
                    item.span,
                ));
            }
            crate::robotics::planner::manifest_joints(&port_docs)
                .map_err(|message| error("S1400", message, item.span))?;
        }
    }
    port_docs.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    golden.sort_by(|left, right| left.0.cmp(&right.0));
    properties.sort();
    let golden: Vec<_> = golden
        .into_iter()
        .map(|(path, tolerances)| json!({"path":path,"tolerances":tolerances}))
        .collect();
    let document = json!({"id":name,"name":name,"external":external,"ports":port_docs,"params":params,"resources":resource_doc,"license":license,"isolation":isolation,"wcet":wcet,"deadline":deadline,"implementation":implementation,"abi":abi,"lifecycle":lifecycle,"golden":golden,"properties":properties,"shim":shim,"config":config,"inputs":inputs});
    Ok(Component {
        item,
        external,
        ports,
        json: document,
    })
}
