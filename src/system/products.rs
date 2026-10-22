use super::*;

pub(super) fn product(
    item: &SystemItem,
    name: &str,
    clauses: &[SystemClause],
    use_v5: bool,
) -> Result<Value, FrontError> {
    let mut distribution = None;
    let mut commercial = None;
    let mut robot = None;
    let mut capacity = json!({});
    let mut capacity_seen = false;
    for clause in clauses {
        match clause {
            SystemClause::Distribution(value)
                if matches!(value.as_str(), "internal" | "proprietary-binary" | "open") =>
            {
                if distribution.replace(value.clone()).is_some() {
                    return Err(error("S1000", "duplicate distribution clause", item.span));
                }
            }
            SystemClause::Commercial(value) => {
                if commercial.replace(*value).is_some() {
                    return Err(error("S1000", "duplicate commercial clause", item.span));
                }
            }
            SystemClause::Robot(value) => {
                if robot.is_some() {
                    return Err(error("S1000", "duplicate robot clause", item.span));
                }
                if !use_v5
                    && !matches!(
                        value.as_str(),
                        "menagerie::franka_emika_panda" | "menagerie::universal_robots_ur5e"
                    )
                {
                    return Err(error("S1001", "undefined robot model reference", item.span));
                }
                robot = Some(if use_v5 {
                    qualified(item.module.as_deref(), value)
                } else {
                    value.clone()
                });
            }
            SystemClause::Capacity(values) => {
                if capacity_seen {
                    return Err(error("S1000", "duplicate capacity clause", item.span));
                }
                capacity_seen = true;
                capacity = resources(values)?;
            }
            _ => return Err(error("S1000", "invalid product clause", item.span)),
        }
    }
    if distribution.is_none() || commercial.is_none() || robot.is_none() {
        return Err(error(
            "S1000",
            "product is missing required clauses",
            item.span,
        ));
    }
    Ok(
        json!({"id":name,"name":name,"distribution":distribution,"commercial":commercial,"robot":robot,"capacity":capacity}),
    )
}

pub(super) fn validate_port_types(
    mut program: Program,
    type_probes: Vec<(Type, Span)>,
    use_v2: bool,
    fallback_span: Span,
) -> Result<(), FrontError> {
    if type_probes.len().saturating_add(program.functions.len()) > 128 {
        return Err(error(
            "S1009",
            "system port type limit exceeded",
            fallback_span,
        ));
    }
    let probe = front::parse(b"fn __system_port_type() -> i32 { return 0; }")?
        .functions
        .remove(0);
    for (index, (ty, span)) in type_probes.into_iter().enumerate() {
        let mut function = probe.clone();
        function.name = format!("__system_port_type_{index}");
        let mode = if matches!(ty, Type::I32 | Type::U32 | Type::Bool)
            || matches!(&ty, Type::Named(name) if name.starts_with("Quantity<i32,") || name.starts_with("Quantity<u32,") || name.starts_with("Vector3<") || name.starts_with("Point<"))
        {
            None
        } else {
            Some(Mode::Read)
        };
        function.params = vec![Param {
            name: "value".into(),
            mode,
            ty,
            span,
        }];
        program.functions.push(function);
    }
    crate::check::check_port_types(program).map_err(|failure| {
        let code = if use_v2
            && matches!(
                failure.code,
                "FRAME_ROOTS" | "FRAME_CYCLE" | "FRAME_PARENT_UNKNOWN"
            ) {
            "S1200"
        } else if matches!(failure.code, "FRAME_UNKNOWN" | "FRAME_PARENT_UNKNOWN") {
            "S1001"
        } else {
            "S1002"
        };
        error(code, failure.message, failure.span.unwrap_or(fallback_span))
    })
}
