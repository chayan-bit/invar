use super::*;

pub(super) struct ImportedComponent {
    pub(super) declaration: SystemItem,
    pub(super) manifest: Value,
    pub(super) path: PathBuf,
    pub(super) source: Vec<u8>,
    pub(super) allowed_root: PathBuf,
    pub(super) robot: Option<String>,
}

pub(super) fn imported_component(
    item: &SystemItem,
    source_dir: Option<&Path>,
) -> Result<ImportedComponent, FrontError> {
    let SystemItemKind::ComponentImport { path, robot } = &item.kind else {
        return Err(error("S1400", "expected component import", item.span));
    };
    let relative = Path::new(path);
    if path.len() > 256
        || !path.ends_with(".inv")
        || !relative
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_)))
    {
        return Err(error(
            "S1400",
            "component import needs a bounded local .inv path",
            item.span,
        ));
    }
    let source_dir = source_dir.ok_or_else(|| {
        error(
            "S1405",
            "component import needs a source directory",
            item.span,
        )
    })?;
    let source_root = source_dir.canonicalize().map_err(|_| {
        error(
            "S1405",
            "component source directory is unavailable",
            item.span,
        )
    })?;
    let manifest_path = source_root
        .join(relative)
        .canonicalize()
        .map_err(|_| error("S1405", "component manifest is unavailable", item.span))?;
    if !manifest_path.starts_with(&source_root)
        || !manifest_path.is_file()
        || !manifest_path
            .metadata()
            .is_ok_and(|metadata| metadata.len() <= 1_048_576)
    {
        return Err(error(
            "S1405",
            "component manifest escapes the cell or exceeds its size limit",
            item.span,
        ));
    }
    let bytes = std::fs::read(&manifest_path)
        .map_err(|_| error("S1405", "component manifest cannot be read", item.span))?;
    let program = front::parse_system(&bytes)
        .map_err(|failure| error("S1400", failure.message, item.span))?;
    let mut declaration = program
        .system_items
        .first()
        .cloned()
        .ok_or_else(|| error("S1400", "component manifest has no declaration", item.span))?;
    let manifest =
        compile_manifest(program).map_err(|failure| error("S1400", failure.message, item.span))?;
    declaration.module = item.module.clone();
    declaration.span = item.span;
    let allowed_root = source_root
        .ancestors()
        .find(|ancestor| ancestor.join(".git").exists())
        .unwrap_or(&source_root)
        .to_path_buf();
    Ok(ImportedComponent {
        declaration,
        manifest,
        path: manifest_path,
        source: bytes,
        allowed_root,
        robot: robot.clone(),
    })
}

pub(super) fn verify_import(candidate: &ImportedComponent) -> Result<SystemItem, FrontError> {
    let span = candidate.declaration.span;
    let header =
        foreign::header(&candidate.manifest).map_err(|message| error("S1400", message, span))?;
    let (hash, _) = foreign::manifest_hash(
        &candidate.allowed_root,
        &candidate.path,
        &candidate.manifest,
        &header,
    )
    .map_err(|message| error("S1405", message, span))?;
    let name = candidate.manifest["name"]
        .as_str()
        .ok_or_else(|| error("S1400", "component manifest has no name", span))?;
    let evidence = candidate
        .path
        .parent()
        .ok_or_else(|| error("S1405", "component manifest has no directory", span))?
        .join("evidence")
        .join(format!("{name}-{hash}.json"));
    foreign::verify_evidence(&evidence, &hash).map_err(|message| error("S1405", message, span))?;
    if std::fs::read(&candidate.path)
        .map_err(|_| error("S1405", "component manifest cannot be reread", span))?
        != candidate.source
    {
        return Err(error(
            "S1405",
            "component manifest changed during import",
            span,
        ));
    }
    Ok(candidate.declaration.clone())
}

pub(super) fn selected_robot(program: &Program) -> Option<&str> {
    let system = program
        .system_items
        .iter()
        .find(|item| matches!(item.kind, SystemItemKind::System { .. }))?;
    let SystemItemKind::System { products, .. } = &system.kind else {
        return None;
    };
    let [product] = products.as_slice() else {
        return None;
    };
    let product_id = qualified(system.module.as_deref(), product);
    let declaration = program.system_items.iter().find(|item| {
        matches!(&item.kind, SystemItemKind::Product { name, .. } if qualified(item.module.as_deref(), name) == product_id)
    })?;
    let SystemItemKind::Product { clauses, .. } = &declaration.kind else {
        return None;
    };
    clauses.iter().find_map(|clause| match clause {
        SystemClause::Robot(name) => Some(name.as_str()),
        _ => None,
    })
}

pub(super) fn expanded_items(
    program: &Program,
    source_dir: Option<&Path>,
) -> Result<Vec<SystemItem>, FrontError> {
    let candidates = program
        .system_items
        .iter()
        .filter(|item| matches!(item.kind, SystemItemKind::ComponentImport { .. }))
        .map(|item| imported_component(item, source_dir))
        .collect::<Result<Vec<_>, _>>()?;
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, candidate) in candidates.iter().enumerate() {
        let SystemItemKind::Component { name, .. } = &candidate.declaration.kind else {
            unreachable!("checked foreign manifest")
        };
        groups
            .entry(qualified(candidate.declaration.module.as_deref(), name))
            .or_default()
            .push(index);
    }
    let selected_robot = selected_robot(program);
    let mut selected = BTreeSet::new();
    for indices in groups.values() {
        let mut selectors = BTreeSet::new();
        for &index in indices {
            if let Some(robot) = candidates[index].robot.as_deref()
                && !selectors.insert(robot)
            {
                return Err(error(
                    "S1000",
                    "duplicate robot-selected component import",
                    candidates[index].declaration.span,
                ));
            }
        }
        if indices.len() > 1
            && indices
                .iter()
                .any(|&index| candidates[index].robot.is_none())
        {
            return Err(error(
                "S1000",
                "mixed unconditional and robot-selected imports",
                candidates[indices[0]].declaration.span,
            ));
        }
        let matches = indices
            .iter()
            .copied()
            .filter(|&index| {
                candidates[index]
                    .robot
                    .as_deref()
                    .is_none_or(|robot| selected_robot == Some(robot))
            })
            .collect::<Vec<_>>();
        let [match_index] = matches.as_slice() else {
            let code = if matches.is_empty() { "S1001" } else { "S1000" };
            return Err(error(
                code,
                "component import must match exactly one selected robot",
                candidates[indices[0]].declaration.span,
            ));
        };
        selected.insert(*match_index);
    }
    let mut items = Vec::with_capacity(program.system_items.len());
    let mut cursor = 0;
    for item in &program.system_items {
        if matches!(item.kind, SystemItemKind::ComponentImport { .. }) {
            if selected.contains(&cursor) {
                items.push(verify_import(&candidates[cursor])?);
            }
            cursor += 1;
        } else {
            items.push(item.clone());
        }
    }
    Ok(items)
}
