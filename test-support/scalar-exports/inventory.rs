//! Reproducible source/function inventory, using the real front end and MIR.
//! Candidate classification is preparation, never frozen export admission.
use invar::{check, front, mir, ownership};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

fn files(root: &Path, extensions: &[&str], out: &mut Vec<PathBuf>) {
    if !root.exists() {
        return;
    }
    for entry in fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            if !matches!(
                path.file_name().and_then(|s| s.to_str()),
                Some("target" | ".git" | "__pycache__")
            ) {
                files(&path, extensions, out);
            }
        } else if path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|e| extensions.contains(&e))
        {
            out.push(path);
        }
    }
}

/// Read actual Rust string literals; skip comments and character literals.
/// Provenance uses byte offsets, so duplicate strings remain separate entries.
fn literals(text: &str) -> Vec<(usize, String)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i..].starts_with(b"//") {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if bytes[i..].starts_with(b"/*") {
            i += 2;
            let mut depth = 1;
            while i < bytes.len() && depth > 0 {
                if bytes[i..].starts_with(b"/*") {
                    depth += 1;
                    i += 2;
                } else if bytes[i..].starts_with(b"*/") {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            continue;
        }
        if bytes[i] == b'\'' {
            let mut end = i + 1;
            if bytes.get(end) == Some(&b'\\') {
                end += 2;
                if bytes.get(end - 1) == Some(&b'u') && bytes.get(end) == Some(&b'{') {
                    while end < bytes.len() && bytes[end] != b'}' {
                        end += 1;
                    }
                    end += 1;
                } else if bytes.get(end - 1) == Some(&b'x') {
                    end += 2;
                }
            } else if let Some(ch) = text[end..].chars().next() {
                end += ch.len_utf8();
            }
            if bytes.get(end) == Some(&b'\'') {
                i = end + 1;
                continue;
            }
        }
        if bytes[i] == b'r' {
            let mut quote = i + 1;
            while bytes.get(quote) == Some(&b'#') {
                quote += 1;
            }
            if bytes.get(quote) == Some(&b'"') {
                let hashes = quote - i - 1;
                let end_marker = format!("\"{}", "#".repeat(hashes));
                let start = quote + 1;
                if let Some(relative) = text[start..].find(&end_marker) {
                    out.push((i, text[start..start + relative].to_string()));
                    i = start + relative + end_marker.len();
                    continue;
                }
                panic!("unterminated Rust raw string at byte {i}");
            }
        }
        if bytes[i] != b'"' {
            i += 1;
            continue;
        }
        let origin = i;
        i += 1;
        let mut decoded = String::new();
        while i < bytes.len() && bytes[i] != b'"' {
            if bytes[i] != b'\\' {
                let ch = text[i..].chars().next().unwrap();
                decoded.push(ch);
                i += ch.len_utf8();
                continue;
            }
            i += 1;
            match bytes[i] {
                b'n' => decoded.push('\n'),
                b'r' => decoded.push('\r'),
                b't' => decoded.push('\t'),
                b'0' => decoded.push('\0'),
                b'\\' => decoded.push('\\'),
                b'"' => decoded.push('"'),
                b'\'' => decoded.push('\''),
                b'\n' | b'\r' => {
                    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                        i += 1;
                    }
                    continue;
                }
                b'x' => {
                    decoded.push(u8::from_str_radix(&text[i + 1..i + 3], 16).unwrap() as char);
                    i += 2;
                }
                b'u' => {
                    assert_eq!(bytes[i + 1], b'{');
                    let end = text[i + 2..].find('}').unwrap() + i + 2;
                    let code = text[i + 2..end].replace('_', "");
                    decoded.push(char::from_u32(u32::from_str_radix(&code, 16).unwrap()).unwrap());
                    i = end;
                }
                other => panic!("unexpected Rust string escape {other} at byte {i}"),
            }
            i += 1;
        }
        assert_eq!(bytes.get(i), Some(&b'"'), "unterminated string at {origin}");
        out.push((origin, decoded));
        i += 1;
    }
    out
}

fn integer(ty: &front::Type) -> bool {
    matches!(ty, front::Type::I32 | front::Type::U32)
}

fn internal_scalar(ty: &front::Type) -> bool {
    integer(ty) || matches!(ty, front::Type::Bool | front::Type::Unit)
}

/// Conservative reachable-MIR screen. It does not infer effects or bound loops.
fn reasons(program: &mir::Program, root: &mir::Function) -> BTreeSet<String> {
    let mut reasons = BTreeSet::new();
    if !integer(&root.result)
        || root
            .params
            .iter()
            .any(|p| !integer(&p.ty) || p.mode.is_some())
    {
        reasons.insert("interface_not_i32_u32".to_string());
        return reasons;
    }
    let mut pending = vec![root.name.as_str()];
    let mut seen = BTreeSet::new();
    while let Some(name) = pending.pop() {
        if !seen.insert(name) {
            continue;
        }
        let Some(function) = program.function(name) else {
            if !matches!(name, "wrapping_add" | "wrapping_mul") {
                reasons.insert(format!("unsupported_intrinsic:{name}"));
            }
            continue;
        };
        if function.foreign_oracle.is_some() {
            reasons.insert(format!("reachable_foreign_call:{name}"));
            continue;
        }
        if !internal_scalar(&function.result)
            || function.params.iter().any(|p| !internal_scalar(&p.ty))
        {
            reasons.insert(format!("reachable_nonscalar_signature:{name}"));
        }
        let mut blocks = vec![function.entry];
        let mut visited = BTreeSet::new();
        while let Some(id) = blocks.pop() {
            if !visited.insert(id) {
                continue;
            }
            let block = &function.blocks[id];
            for instruction in &block.instructions {
                if instruction
                    .output
                    .as_ref()
                    .is_some_and(|(_, ty)| !internal_scalar(ty))
                {
                    reasons.insert(format!("reachable_nonscalar_value:{name}"));
                }
                match &instruction.kind {
                    mir::Operation::Call { name, .. } => pending.push(name),
                    mir::Operation::Tick
                    | mir::Operation::I32(_)
                    | mir::Operation::U32(_)
                    | mir::Operation::Bool(_)
                    | mir::Operation::Read(_)
                    | mir::Operation::Unary { .. }
                    | mir::Operation::Binary { .. }
                    | mir::Operation::Bind { .. }
                    | mir::Operation::Store { .. }
                    | mir::Operation::Drop(_)
                    | mir::Operation::ForNext(_) => {}
                    _ => {
                        reasons.insert(format!("reachable_nonscalar_operation:{name}"));
                    }
                }
            }
            match block.end.as_ref().unwrap() {
                mir::Terminator::Goto { target, .. } => blocks.push(*target),
                mir::Terminator::Branch { yes, no, .. } => {
                    blocks.push(*yes);
                    blocks.push(*no);
                }
                mir::Terminator::Return { .. } => {}
            }
        }
    }
    reasons
}

pub fn classify(id: String, source: String) -> Value {
    let parsed = match front::parse(source.as_bytes()) {
        Ok(parsed) => parsed,
        Err(error) => {
            return json!({"id":id,"status":"parse_rejected","reason":error.code,"source":source});
        }
    };
    let source_functions: Vec<_> = parsed.functions.iter().map(|f| json!({
        "name":f.name,"module":f.module,"parameters":f.params.iter().map(|p|format!("{:?}",p.ty)).collect::<Vec<_>>(),
        "result":format!("{:?}",f.result),"generic":!f.generic_params.is_empty(),
    })).collect();
    let checked = match check::check(parsed, false).and_then(|checked| {
        ownership::check(&checked)?;
        Ok(checked)
    }) {
        Ok(checked) => checked,
        Err(error) => {
            return json!({"id":id,"status":"source_rejected","reason":error.code,"source_functions":source_functions,"source":source});
        }
    };
    let lowered = match mir::lower(&checked) {
        Ok(lowered) => lowered,
        Err(error) => {
            return json!({"id":id,"status":"mir_rejected","reason":error.code,"source_functions":source_functions,"source":source});
        }
    };
    let functions: Vec<_> = lowered.functions.iter().map(|f| {
        let excluded = reasons(&lowered, f);
        json!({"name":f.name,"parameters":f.params.iter().map(|p|format!("{:?}",p.ty)).collect::<Vec<_>>(),
            "result":format!("{:?}",f.result),"status":if excluded.is_empty(){"scalar_mir_candidate"}else{"excluded_by_preparation_screen"},
            "reasons":excluded,"export_admission":"preparation_screen_only"})
    }).collect();
    json!({"id":id,"status":"checked","source_functions":source_functions,"functions":functions,"source":source})
}

pub fn enumerate() -> Value {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut paths = Vec::new();
    for dir in ["examples", "tests/fixtures", "bench", "tools/bench", "lib"] {
        files(&root.join(dir), &["inv", "invar"], &mut paths);
    }
    paths.sort();
    let standalone_count = paths.len();
    let mut entries: Vec<_> = paths
        .iter()
        .map(|path| {
            classify(
                path.strip_prefix(root).unwrap().display().to_string(),
                fs::read_to_string(path).unwrap(),
            )
        })
        .collect();
    let benchmark: Value =
        serde_json::from_str(&fs::read_to_string(root.join("tools/bench/corpus-v1.json")).unwrap())
            .unwrap();
    let benchmark_count = benchmark["programs"].as_array().unwrap().len();
    for program in benchmark["programs"].as_array().unwrap() {
        entries.push(classify(
            format!(
                "tools/bench/corpus-v1.json#{}",
                program["id"].as_str().unwrap()
            ),
            program["source"].as_str().unwrap().to_string(),
        ));
    }
    let m2: Value = serde_json::from_str(include_str!("generated-m2-v1.json")).unwrap();
    let m2_count = m2["programs"].as_array().unwrap().len();
    assert_eq!(m2_count, 252);
    for program in m2["programs"].as_array().unwrap() {
        entries.push(classify(
            format!("generated-m2#{}", program["id"].as_str().unwrap()),
            program["source"].as_str().unwrap().to_string(),
        ));
    }
    let mut rust_paths = Vec::new();
    for dir in ["src", "tests", "crates", "bench", "tools"] {
        files(&root.join(dir), &["rs"], &mut rust_paths);
    }
    rust_paths.sort();
    let mut inline_count = 0;
    for path in rust_paths {
        let relative = path.strip_prefix(root).unwrap().display().to_string();
        if matches!(
            relative.as_str(),
            "tests/exports/scalar_export_corpus.rs" | "src/eval/named.rs"
        ) {
            continue;
        }
        let text = fs::read_to_string(&path).unwrap();
        for (offset, source) in literals(&text) {
            if source.contains("fn ") || source.contains("extern c ") {
                entries.push(classify(
                    format!("{relative}#literal-byte-{offset}"),
                    source,
                ));
                inline_count += 1;
            }
        }
    }
    entries.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    let mut statuses = std::collections::BTreeMap::new();
    let mut rejection_reasons = std::collections::BTreeMap::new();
    let mut checked_functions = 0;
    let mut candidates = 0;
    let mut excluded = 0;
    for entry in &entries {
        *statuses
            .entry(entry["status"].as_str().unwrap())
            .or_insert(0) += 1;
        if let Some(reason) = entry["reason"].as_str() {
            *rejection_reasons.entry(reason.to_string()).or_insert(0) += 1;
        }
        if let Some(functions) = entry["functions"].as_array() {
            checked_functions += functions.len();
            for function in functions {
                if function["status"] == "scalar_mir_candidate" {
                    candidates += 1;
                } else {
                    excluded += 1;
                }
                for reason in function["reasons"].as_array().unwrap() {
                    *rejection_reasons
                        .entry(reason.as_str().unwrap().to_string())
                        .or_insert(0) += 1;
                }
            }
        }
    }
    json!({"schema_version":1,"implementation_base":"d1fa19b54ab879fb121d129b3d643345839e1632",
        "scope":"All .inv/.invar in examples, tests/fixtures, bench, tools/bench, lib; all 240 benchmark embedded programs; 252 exported M2 generated programs; Rust string literals containing fn/extern c in src/tests/crates/bench/tools (excluding this lane's new test and named module). Input occurrences are counted, not deduplicated.",
        "limitations":["Rust literals include incomplete fragments and uninstantiated format templates; parse rejection does not classify their runtime expansions.","Runtime-generated programs outside the explicitly replayed M2 corpus are not enumerated; full-repository callable corpus cannot be claimed until generators and frozen API are covered.","Reachable-MIR scalar screening is conservative and is not the export owner's admission policy.","No named-call/native-export executions or acceptance count yet."],
        "compiled_export_admission":"separate_actual_facade_qualification",
        "summary":{"input_entries":entries.len(),"standalone_sources":standalone_count,"benchmark_embedded":benchmark_count,"generated_m2":m2_count,"rust_literal_occurrences":inline_count,
            "input_statuses":statuses,"checked_functions":checked_functions,"scalar_mir_candidates":candidates,"excluded_by_preparation_screen":excluded,"reasons":rejection_reasons},
        "entries":entries})
}
