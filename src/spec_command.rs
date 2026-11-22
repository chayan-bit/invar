//! `invar spec [--topic T]... [--chapter ID]... [--budget N] [--format md|json]`: the
//! core chapters plus the chosen ones (DX-25, #86; `src/spec_docs.rs`).

use invar::spec_docs::Spec;
use serde_json::json;
use std::ffi::OsString;

pub(crate) fn command(args: &[OsString]) -> i32 {
    let spec = Spec::embedded();
    let (mut topics, mut chapters, mut budget, mut json) = (Vec::new(), Vec::new(), None, None);
    let mut rest = args.iter().map(|arg| arg.to_str());
    while let Some(flag) = rest.next() {
        let value = rest.next().flatten().map(str::to_owned);
        match (flag, value) {
            (Some("--topic"), Some(topic)) => topics.push(topic),
            (Some("--chapter"), Some(id)) => chapters.push(id),
            (Some("--budget"), Some(n)) if budget.is_none() => match n.parse::<usize>() {
                Ok(n) => budget = Some(n),
                Err(_) => return crate::cli::usage(),
            },
            (Some("--format"), Some(format))
                if json.is_none() && (format == "md" || format == "json") =>
            {
                json = Some(format == "json");
            }
            _ => return crate::cli::usage(),
        }
    }
    let (budget, json) = (
        budget.unwrap_or(spec.default_budget()),
        json.unwrap_or(false),
    );
    match spec.assemble(&topics, &chapters, budget) {
        Ok(assembly) if json => {
            let chapters: Vec<_> = assembly
                .chapters
                .iter()
                .map(|(id, path, tokens)| json!({"id": id, "path": path, "tokens": tokens}))
                .collect();
            let document = json!({
                "schema": "invar-spec-assembly-v1",
                "budget": budget,
                "tokens": assembly.tokens,
                "chapters": chapters,
                "text": assembly.text,
            });
            println!("{document}");
            0
        }
        Ok(assembly) => {
            print!("{}", assembly.text);
            0
        }
        Err(error) if json => {
            let diagnostic = invar::diagnostics::diagnostic(
                error.code(),
                &error.to_string(),
                "docs/spec/chapters.json",
                &[],
                None,
            );
            println!(
                "{}",
                invar::diagnostics::document("tool_error", vec![diagnostic])
            );
            2
        }
        Err(error) => {
            eprintln!("error[{}]: {error}", error.code());
            2
        }
    }
}
