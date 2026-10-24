#![forbid(unsafe_code)]

#[cfg(feature = "cuda")]
use invar::engine::nvrtc_helper;
mod auto_command;
mod calibrate_command;
mod cli;
mod component_test;
mod fix_cli;
mod gnet_device_command;
mod metal_build;
mod native_command;
mod source_device_command;

use cli::{source_directory, usage};
use std::ffi::OsString;
use std::io::Read;

use invar::{
    eval::{EvalError, Value},
    front::{ErrorKind, FrontError},
};

fn main() {
    std::process::exit(run(std::env::args_os().collect()));
}

fn run(args: Vec<OsString>) -> i32 {
    if let Some(code) = native_command::debug_command(&args) {
        return code;
    }
    if args.len() >= 3
        && (args[1] == "run" || args[1] == "explain")
        && args[3..]
            .windows(2)
            .any(|p| p[0] == "--engine" && p[1] == "auto")
        && auto_command::placement_requested(&args[3..])
    {
        return auto_command::command(&args);
    }
    if args.get(1).is_some_and(|arg| arg == "calibrate") {
        return calibrate_command::command(&args[2..]);
    }
    if args.get(1).is_some_and(|arg| arg == "source-device") {
        return source_device_command::command(&args[2..]);
    }
    if args.get(1).is_some_and(|arg| arg == "gnet-device") {
        return gnet_device_command::command(&args[2..]);
    }
    if args.len() == 2 && (args[1] == "--help" || args[1] == "-h") {
        print!("{}", invar::help());
        return 0;
    }
    if args.len() == 4 && args[1] == "import" && (args[2] == "mjcf" || args[2] == "urdf") {
        return match if args[2] == "urdf" {
            invar::system::import_urdf(std::path::Path::new(&args[3]))
        } else {
            invar::system::import_mjcf(std::path::Path::new(&args[3]))
        } {
            Ok((source, warnings)) => {
                print!("{source}");
                for warning in warnings {
                    eprintln!("S1604 warning: {warning}");
                }
                0
            }
            Err(error) => report_front(error, &args[3], false),
        };
    }
    if args.len() == 6 && args[1] == "component" && args[2] == "generate" && args[4] == "--output" {
        return component_generate(&args[3], &args[5]);
    }
    if args.len() >= 4 && args[1] == "component" && args[2] == "test" {
        return component_test::command(&args[3..]);
    }
    if args.len() == 4 && args[1] == "system" && args[2] == "ir" {
        return system_command(&args[3], SystemAction::Ir);
    }
    if args.len() == 4 && args[1] == "system" && args[2] == "report" {
        return system_command(&args[3], SystemAction::Report(None));
    }
    if args.len() == 6 && args[1] == "system" && args[2] == "report" && args[4] == "--run-evidence"
    {
        return system_command(&args[3], SystemAction::Report(Some(&args[5])));
    }
    if args.len() == 7
        && args[1] == "system"
        && args[2] == "gen"
        && args[3] == "ros2"
        && args[5] == "--out"
    {
        return system_gen_ros2(&args[4], &args[6]);
    }
    if args.len() == 7
        && args[1] == "system"
        && args[2] == "gen"
        && args[3] == "mjcf"
        && args[5] == "--out"
    {
        return system_gen_robot(&args[4], &args[6], "mjcf");
    }
    if args.len() == 7
        && args[1] == "system"
        && args[2] == "gen"
        && args[3] == "urdf"
        && args[5] == "--out"
    {
        return system_gen_robot(&args[4], &args[6], "urdf");
    }
    if args.len() == 6
        && args[1] == "system"
        && args[2] == "check"
        && args[4] == "--format"
        && args[5] == "json"
    {
        return system_command(&args[3], SystemAction::Check);
    }
    if args.len() >= 3 && args[1] == "check" {
        if args.get(3).is_some_and(|flag| flag == "--fix") {
            return fix_cli::run(&args[2], &args[4..]);
        }
        if args.len() != 3 && (args.len() != 5 || args[3] != "--format" || args[4] != "json") {
            return usage();
        }
        if let Ok(source) = read_source(&args[2])
            && invar::front::contains_system_declaration(&source)
        {
            return system_command(&args[2], SystemAction::Check);
        }
        return execute(&args[2], args.len() == 5, None, None, None, None, None);
    }
    if args.len() == 5 && args[1] == "explain" && args[3] == "--format" && args[4] == "json" {
        return explain(&args[2], None, None);
    }
    if args.len() == 7
        && args[1] == "explain"
        && args[3] == "--engine"
        && (args[4] == "cuda" || args[4] == "metal" || args[4] == "auto")
        && args[5] == "--format"
        && args[6] == "json"
    {
        return explain(
            &args[2],
            Some(if args[4] == "cuda" {
                "cuda"
            } else if args[4] == "metal" {
                "metal"
            } else {
                "auto"
            }),
            None,
        );
    }
    if args.len() == 11
        && args[1] == "build"
        && args[3] == "--target"
        && args[4] == "cuda"
        && args[5] == "--entry"
        && args[7] == "--sm"
        && args[9] == "--output"
    {
        return gnet_device_command::build(&args[2], &args[6], &args[8], &args[10]);
    }
    if args.len() == 9
        && args[1] == "explain"
        && args[3] == "--engine"
        && (args[4] == "metal" || args[4] == "auto")
        && args[5] == "--artifact"
        && args[7] == "--format"
        && args[8] == "json"
    {
        let engine = if args[4] == "metal" { "metal" } else { "auto" };
        return explain(&args[2], Some(engine), Some(&args[6]));
    }
    if args.len() == 7
        && args[1] == "build"
        && args[3] == "--target"
        && args[4] == "metal"
        && args[5] == "--output"
    {
        return metal_build::run(&args[2], None, &args[6]);
    }
    if args.len() == 9
        && args[1] == "build"
        && args[3] == "--target"
        && args[4] == "metal"
        && args[5] == "--entry"
        && args[7] == "--output"
    {
        return metal_build::run(&args[2], Some(&args[6]), &args[8]);
    }
    if args.len() == 13
        && args[1] == "build"
        && args[3] == "--target"
        && args[4] == "metal"
        && args[5] == "--entry"
        && args[7] == "--family"
        && args[8] == "tensor"
        && args[9] == "--fuel"
        && args[11] == "--output"
    {
        return metal_build::run_tensor(&args[2], &args[6], &args[10], &args[12]);
    }
    if args.len() == 7
        && args[1] == "build"
        && args[3] == "--target"
        && args[4] == "native"
        && args[5] == "--output"
    {
        return native_command::build(&args[2], &args[6], None);
    }
    if args.len() == 9
        && args[1] == "build"
        && args[3] == "--target"
        && args[4] == "native"
        && args[5] == "--output"
        && args[7] == "--c-library"
    {
        return native_command::build(&args[2], &args[6], Some(&args[8]));
    }
    if args.len() >= 5 && args[1] == "run" {
        let mut engine = None;
        let mut json = false;
        let mut fuel = invar::eval::DEFAULT_FUEL;
        let mut fuel_seen = false;
        let mut workers = None;
        let mut c_library = None;
        let mut artifact = None;
        let mut options = args[3..].chunks_exact(2);
        if !options.remainder().is_empty() {
            return usage();
        }
        for pair in &mut options {
            if pair[0] == "--engine"
                && engine.is_none()
                && let Some(selected) = pair[1].to_str().and_then(invar::engine::Engine::parse)
            {
                engine = Some(selected.name());
            } else if pair[0] == "--workers" && workers.is_none() {
                let Some(text) = pair[1].to_str() else {
                    return usage();
                };
                let Ok(count) = text.parse::<usize>() else {
                    return usage();
                };
                if !(1..=invar::schedule::MAX_WORKERS).contains(&count) {
                    return usage();
                }
                workers = Some(count);
            } else if pair[0] == "--format" && pair[1] == "json" && !json {
                json = true;
            } else if pair[0] == "--c-library" && c_library.is_none() {
                c_library = Some(&pair[1]);
            } else if pair[0] == "--artifact" && artifact.is_none() {
                artifact = Some(&pair[1]);
            } else if pair[0] == "--fuel" && !fuel_seen {
                let Some(text) = pair[1].to_str() else {
                    return usage();
                };
                if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
                    return usage();
                }
                let Ok(parsed) = text.parse::<u64>() else {
                    return usage();
                };
                if parsed == 0 {
                    return usage();
                }
                fuel = parsed;
                fuel_seen = true;
            } else {
                return usage();
            }
        }
        let Some(engine) = engine else {
            return usage();
        };
        if (engine == "cpu-parallel") != workers.is_some()
            || (artifact.is_some() && engine != "metal" && engine != "auto")
        {
            return usage();
        }
        return execute(
            &args[2],
            json,
            Some(fuel),
            Some(engine),
            workers,
            c_library,
            artifact,
        );
    }
    usage()
}

fn component_generate(path: &OsString, output: &OsString) -> i32 {
    let source = match read_source(path) {
        Ok(source) => source,
        Err(error) => return report_front(error, path, true),
    };
    let manifest =
        match invar::front::parse_system(&source).and_then(invar::system::compile_manifest) {
            Ok(manifest) => manifest,
            Err(error) => return report_front(error, path, true),
        };
    let header = match invar::system::foreign::header(&manifest) {
        Ok(header) => header,
        Err(message) => return report_front(FrontError::tool("S1400", message), path, true),
    };
    let Some(prefix) = manifest["abi"]["prefix"].as_str() else {
        return report_front(
            FrontError::tool("S1400", "checked ABI prefix missing"),
            path,
            true,
        );
    };
    let directory = std::path::Path::new(output);
    if std::fs::create_dir(directory).is_err() {
        return report_front(
            FrontError::tool(
                "S1400",
                "output directory must not exist and its parent must be writable",
            ),
            path,
            true,
        );
    }
    for (name, bytes) in [
        (format!("{prefix}.h"), header.into_bytes()),
        (
            format!("{prefix}.manifest.json"),
            manifest.to_string().into_bytes(),
        ),
    ] {
        let result = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join(name))
            .and_then(|mut file| std::io::Write::write_all(&mut file, &bytes));
        if result.is_err() {
            return report_front(
                FrontError::tool("S1400", "cannot write generated component file"),
                path,
                true,
            );
        }
    }
    println!(
        "{}",
        serde_json::json!({"status":"ok","abi":prefix,"generated_files":2})
    );
    0
}

enum SystemAction<'a> {
    Check,
    Ir,
    Report(Option<&'a OsString>),
}

fn system_gen_ros2(path: &OsString, output: &OsString) -> i32 {
    let source = match read_source(path) {
        Ok(source) => source,
        Err(error) => return report_front(error, path, true),
    };
    let program = match invar::front::parse_system(&source) {
        Ok(program) => program,
        Err(error) => return report_front(error, path, true),
    };
    let ir = match invar::system::compile_at(program, source_directory(path)) {
        Ok(ir) => ir,
        Err(error) => return report_front(error, path, true),
    };
    if let Some((code, message, span)) = ir.graph_issues().into_iter().next() {
        return report_generated_source(code, &message, path, &source, span);
    }
    let files = match invar::system::ros2::generate(&ir) {
        Ok(files) => files,
        Err(error) => {
            return report_generated_source(
                error.code,
                &error.message,
                path,
                &source,
                invar::front::Span { start: 0, end: 0 },
            );
        }
    };
    let mut files: std::collections::BTreeMap<String, Vec<u8>> = files
        .into_iter()
        .map(|(name, content)| (name, content.into_bytes()))
        .collect();
    if ir
        .checked_document()
        .is_ok_and(|document| document["robots"].is_array())
    {
        let model_files = match invar::system::generate_ros2_mjcf(&ir) {
            Ok(files) => files,
            Err(error) => return report_front(error, path, true),
        };
        files.extend(model_files.into_iter().map(|(name, content)| {
            (
                format!("src/invar_cell_bringup/config/mjcf/{name}"),
                content,
            )
        }));
    }
    write_generated_files(path, output, files, "ros2", "S1500")
}

fn system_gen_robot(path: &OsString, output: &OsString, kind: &str) -> i32 {
    let source = match read_source(path) {
        Ok(source) => source,
        Err(error) => return report_front(error, path, true),
    };
    let program = match invar::front::parse_system(&source) {
        Ok(program) => program,
        Err(error) => return report_front(error, path, true),
    };
    let ir = match invar::system::compile_at(program, source_directory(path)) {
        Ok(ir) => ir,
        Err(error) => return report_front(error, path, true),
    };
    if let Some((code, message, span)) = ir.graph_issues().into_iter().next() {
        return report_generated_source(code, &message, path, &source, span);
    }
    let files = match if kind == "urdf" {
        invar::system::generate_urdf(&ir)
    } else {
        invar::system::generate_mjcf(&ir)
    } {
        Ok(files) => files,
        Err(error) => return report_front(error, path, true),
    };
    write_generated_files(path, output, files, kind, "S1605")
}

fn write_generated_files(
    path: &OsString,
    output: &OsString,
    files: std::collections::BTreeMap<String, Vec<u8>>,
    label: &str,
    code: &'static str,
) -> i32 {
    let destination = std::path::Path::new(output);
    if destination.exists() || destination.file_name().is_none() {
        return report_front(
            FrontError::tool(code, "output directory must be new"),
            path,
            true,
        );
    }
    let parent = destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let mut temporary = None;
    for attempt in 0..16 {
        let candidate = parent.join(format!(".invar-{label}-{}-{attempt}", std::process::id()));
        if std::fs::create_dir(&candidate).is_ok() {
            temporary = Some(candidate);
            break;
        }
    }
    let Some(temporary) = temporary else {
        return report_front(
            FrontError::tool(code, "cannot create temporary output directory"),
            path,
            true,
        );
    };
    let write_result = (|| -> std::io::Result<()> {
        for (name, contents) in &files {
            let file = temporary.join(name);
            if let Some(parent) = file.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(file, contents)?;
        }
        if destination.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "output directory exists",
            ));
        }
        std::fs::rename(&temporary, destination)
    })();
    if write_result.is_err() {
        let _ = std::fs::remove_dir_all(&temporary);
        return report_front(
            FrontError::tool(code, "cannot publish generated workspace"),
            path,
            true,
        );
    }
    println!(
        "{}",
        serde_json::json!({"status":"ok","generated_files":files.len()})
    );
    0
}

fn report_generated_source(
    code: &'static str,
    message: &str,
    path: &OsString,
    source: &[u8],
    span: invar::front::Span,
) -> i32 {
    let diagnostic =
        invar::diagnostics::diagnostic(code, message, &path.to_string_lossy(), source, Some(span));
    println!(
        "{}",
        invar::diagnostics::document("source_error", vec![diagnostic])
    );
    1
}

fn system_command(path: &OsString, action: SystemAction<'_>) -> i32 {
    let source = match read_source(path) {
        Ok(source) => source,
        Err(error) => return report_front(error, path, true),
    };
    let program = match invar::front::parse_system(&source) {
        Ok(program) => program,
        Err(error) => return report_front(error, path, true),
    };
    if matches!(action, SystemAction::Check)
        && matches!(
            program.system_items.as_slice(),
            [invar::front::SystemItem {
                kind: invar::front::SystemItemKind::Component { external: true, .. },
                ..
            }]
        )
    {
        return match invar::system::compile_manifest(program) {
            Ok(_) => {
                println!("{}", invar::diagnostics::document("ok", Vec::new()));
                0
            }
            Err(error) => report_front(error, path, true),
        };
    }
    let ir = match invar::system::compile_at(program, source_directory(path)) {
        Ok(ir) => ir,
        Err(error) => return report_front(error, path, true),
    };
    let path_name = path.to_string_lossy();
    let graph_issues = ir.graph_issues();
    if !graph_issues.is_empty() {
        let diagnostics = graph_issues
            .into_iter()
            .map(|(code, message, span)| {
                invar::diagnostics::diagnostic(code, &message, &path_name, &source, Some(span))
            })
            .collect();
        println!(
            "{}",
            invar::diagnostics::document_in_order("source_error", diagnostics)
        );
        return 1;
    }
    if matches!(action, SystemAction::Ir) {
        println!("{}", ir.canonical_json());
    } else if let SystemAction::Report(evidence) = action {
        if let Some(evidence) = evidence {
            let mut bytes = Vec::new();
            if let Err(error) = std::fs::File::open(evidence)
                .and_then(|file| file.take(1_048_577).read_to_end(&mut bytes))
            {
                return report_front(
                    FrontError::tool("FILE_READ", format!("cannot read run evidence: {error}")),
                    evidence,
                    true,
                );
            }
            return match ir.report_with_run_evidence(&bytes) {
                Ok((report, accepted)) => {
                    print!("{report}");
                    i32::from(!accepted)
                }
                Err(message) => report_front(FrontError::tool("S1406", message), evidence, true),
            };
        }
        match ir.report() {
            Ok(report) => print!("{report}"),
            Err(message) => {
                return report_front(FrontError::tool("UNSUPPORTED_FEATURE", message), path, true);
            }
        }
    } else {
        let warnings = ir
            .warnings()
            .iter()
            .map(|(message, span)| {
                let mut diagnostic = invar::diagnostics::diagnostic(
                    "S1011",
                    message,
                    &path_name,
                    &source,
                    Some(*span),
                );
                diagnostic["severity"] = serde_json::json!("warning");
                diagnostic
            })
            .chain(ir.robot_warnings().iter().map(|(message, span)| {
                let mut diagnostic = invar::diagnostics::diagnostic(
                    "S1604",
                    message,
                    &path_name,
                    &source,
                    Some(*span),
                );
                diagnostic["severity"] = serde_json::json!("warning");
                diagnostic
            }))
            .collect();
        println!("{}", invar::diagnostics::document("ok", warnings));
    }
    0
}

/// Numeric contracts of the program `invar run` executes, recorded in its
/// JSON receipt (one program per process).
static RUN_CONTRACTS: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
/// Its fold receipt (D12).
static RUN_FOLD: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();

fn print_run_json(mut document: serde_json::Value, path: &OsString) -> i32 {
    if let Some(contracts) = RUN_CONTRACTS.get() {
        document["numeric_contracts"] = contracts.clone();
    }
    if let Some(fold) = RUN_FOLD.get() {
        document["numeric_contracts"]["fold"] = fold.clone();
    }
    document["schema"] = serde_json::json!(invar::diagnostics::SCHEMA);
    let serialized = document.to_string();
    if serialized.len() > 1_000_000 {
        println!(
            "{}",
            invar::diagnostics::document(
                "tool_error",
                vec![invar::diagnostics::diagnostic(
                    "DIAGNOSTIC_DOCUMENT_LIMIT",
                    "JSON output exceeds one million bytes",
                    &path.to_string_lossy(),
                    &[],
                    None,
                )],
            )
        );
        2
    } else {
        println!("{serialized}");
        0
    }
}

fn load_checked(
    path: &OsString,
    needs_main: bool,
) -> Result<(Vec<u8>, invar::check::CheckedProgram), FrontError> {
    read_source(path).and_then(|bytes| {
        let checked = invar::front::parse(&bytes)
            .and_then(|program| invar::check::check(program, needs_main))?;
        Ok((bytes, checked))
    })
}

fn read_source(path: &OsString) -> Result<Vec<u8>, FrontError> {
    if std::path::Path::new(path).extension() != Some(std::ffi::OsStr::new("inv")) {
        return Err(FrontError::tool(
            "FILE_EXTENSION",
            "Invar source paths must end in .inv",
        ));
    }
    std::fs::read(path)
        .map_err(|error| FrontError::tool("FILE_READ", format!("cannot read source: {error}")))
}

fn explain(path: &OsString, engine: Option<&str>, artifact: Option<&OsString>) -> i32 {
    if let Some(error) = unhonored_artifact(artifact) {
        return report_eval(error, path, true);
    }
    let source = match read_source(path) {
        Ok(source) => source,
        Err(error) => return report_front(error, path, true),
    };
    let checked = match invar::front::parse(&source)
        .and_then(|program| invar::check::check(program, engine.is_some()))
    {
        Ok(checked) => checked,
        Err(error) => return report_front(error, path, true),
    };
    let lowered = match invar::mir::lower(&checked) {
        Ok(lowered) => lowered,
        Err(error) => return report_front(FrontError::tool(error.code, error.message), path, true),
    };
    match invar::schedule::regions(&source, &lowered) {
        Ok(report) => {
            let mut document = report.to_json();
            explain_intrinsics(&mut document, &lowered);
            if let Some(engine) = engine {
                let Some(selected) = invar::engine::Engine::parse(engine) else {
                    return usage();
                };
                let device_result = invar::engine::run_mir(
                    &source,
                    &lowered,
                    selected,
                    1_000_000,
                    1,
                    artifact.map(std::path::Path::new),
                );
                match device_result {
                    Ok(device) => {
                        let mut device = invar::engine::device_document(&device);
                        invar::spatial::placement::annotate_device(&lowered, &mut device);
                        document["device_execution"] = device;
                    }
                    Err(error) => return auto_command::report(error, path, true, selected),
                }
            }
            // Every `--engine auto` explain records its path (#50).
            if engine == Some("auto") {
                document["auto_path"] = serde_json::json!("established");
            }
            document["schema"] = serde_json::json!(invar::admission::SCHEMA);
            if document.get("device_execution").is_some() {
                document["device_schema"] = serde_json::json!(invar::admission::DEVICE_SCHEMA);
            }
            if document.to_string().len() > 1_000_000 {
                return report_front(
                    FrontError::tool(
                        "DIAGNOSTIC_DOCUMENT_LIMIT",
                        "explain document exceeds one million bytes",
                    ),
                    path,
                    true,
                );
            }
            println!("{document}");
            0
        }
        Err(error) => report_front(FrontError::tool(error.code, error.message), path, true),
    }
}

/// The numeric contracts and explicit intrinsic sites (sort, spatial) every
/// explain document carries.
fn explain_intrinsics(document: &mut serde_json::Value, program: &invar::mir::Program) {
    if let Some(contracts) = invar::math::contracts(program) {
        document["numeric_contracts"] = contracts;
    }
    let sorts = invar::sorting::explain(program);
    if sorts.as_array().is_some_and(|sites| !sites.is_empty()) {
        document["explicit_sorts"] = sorts;
    }
    let spatial = invar::spatial::placement::explain(program);
    if spatial.as_array().is_some_and(|sites| !sites.is_empty()) {
        document["explicit_spatial"] = spatial;
    }
}

fn report_front(error: FrontError, path: &OsString, json: bool) -> i32 {
    if json {
        println!(
            "{}",
            error.to_diagnostics_json(&path.to_string_lossy(), &diagnostic_source(path))
        );
    } else {
        eprintln!(
            "{}: {}: {}",
            path.to_string_lossy(),
            error.code,
            error.message
        );
    }
    if error.kind == ErrorKind::Source {
        1
    } else {
        2
    }
}

fn execute(
    path: &OsString,
    json: bool,
    fuel: Option<u64>,
    engine: Option<&str>,
    workers: Option<usize>,
    c_library: Option<&OsString>,
    artifact: Option<&OsString>,
) -> i32 {
    if let Some(error) = unhonored_artifact(artifact) {
        return report_eval(error, path, json);
    }
    let (source, checked) = match load_checked(path, fuel.is_some()) {
        Ok(checked) => checked,
        Err(error) => return report_front(error, path, json),
    };
    let Some(fuel) = fuel else {
        if json {
            return print_run_json(
                serde_json::json!({
                    "schema_version": 3, "status": "ok", "diagnostics": [],
                    "checks_completed": ["syntax", "names", "types", "local_ownership", "call_loans", "view_lifetimes"],
                    "checks_pending": []
                }),
                path,
            );
        } else {
            println!("Syntax, names, types, ownership, and call loans accepted.");
        }
        return 0;
    };
    if let Some(contracts) = invar::mir::lower(&checked)
        .ok()
        .and_then(|program| invar::math::contracts(&program))
    {
        let _ = RUN_CONTRACTS.set(contracts);
    }
    let foreign = !checked.program.foreign_functions.is_empty();
    if foreign != c_library.is_some()
        || (foreign && !matches!(engine, Some("reference" | "native-serial")))
    {
        return usage();
    }
    if let Some(library) = c_library {
        let library = std::path::Path::new(library);
        if !library.is_absolute() || !library.is_file() {
            return usage();
        }
    }
    if engine == Some("native-serial") {
        return execute_native(&checked, path, json, fuel, c_library);
    }
    if let Some(selected) = engine.and_then(invar::engine::Engine::parse)
        && !matches!(selected, invar::engine::Engine::Reference)
    {
        return execute_scheduled(
            &source,
            &checked,
            path,
            json,
            fuel,
            selected,
            workers.unwrap_or(1),
            artifact,
        );
    }
    match invar::eval::run(&checked, fuel) {
        Ok(outcome) => match outcome.value {
            Value::I32(value) => {
                if json {
                    return print_run_json(
                        serde_json::json!({
                            "schema_version": 3, "status": "ok", "diagnostics": [],
                            "result": {"type": "i32", "value": value}
                        }),
                        path,
                    );
                } else {
                    println!("Result: {value}");
                }
                0
            }
            _ => report_eval(
                EvalError::Tool {
                    code: "INTERNAL_MAIN",
                    message: "checked main returned non-i32".into(),
                },
                path,
                json,
            ),
        },
        Err(error) => report_eval(error, path, json),
    }
}

#[allow(clippy::too_many_arguments)] // CLI flags map one-to-one onto the shared engine call.
fn execute_scheduled(
    source: &[u8],
    checked: &invar::check::CheckedProgram,
    path: &OsString,
    json: bool,
    fuel: u64,
    engine: invar::engine::Engine,
    workers: usize,
    artifact: Option<&OsString>,
) -> i32 {
    let program = match invar::mir::lower(checked) {
        Ok(program) => program,
        Err(error) => return report_front(FrontError::tool(error.code, error.message), path, json),
    };
    match invar::engine::run_mir(
        source,
        &program,
        engine,
        fuel,
        workers,
        artifact.map(std::path::Path::new),
    ) {
        Ok(receipt) => {
            if json {
                let mut diagnostics = Vec::new();
                if let Some(reason) = receipt["fallback_reason"].as_str() {
                    let mut note = invar::diagnostics::diagnostic(
                        reason,
                        "CPU execution used the serial fallback",
                        &path.to_string_lossy(),
                        source,
                        None,
                    );
                    note["severity"] = serde_json::json!("note");
                    diagnostics.push(note);
                }
                let mut document = serde_json::json!({
                    "schema_version":3, "status":"ok", "diagnostics":diagnostics,
                    "result":{"type":"i32", "value":receipt["result"]},
                });
                if let Some(route) = receipt.get("normal_form") {
                    document["normal_form"] = route.clone();
                }
                return auto_command::receipt(document, path, engine);
            }
            println!("Result: {}", receipt["result"]);
            0
        }
        Err(error) => auto_command::report(error, path, json, engine),
    }
}

/// `--artifact` names a compiled-ahead Metal library; a build without Metal
/// cannot honor it, so it is refused before any backend is chosen.
fn unhonored_artifact(artifact: Option<&OsString>) -> Option<EvalError> {
    (artifact.is_some() && !cfg!(all(feature = "metal", target_os = "macos"))).then(|| {
        EvalError::Tool {
            code: "METAL_UNAVAILABLE",
            message: "this invar build has no Metal support, so --artifact cannot be honored"
                .into(),
        }
    })
}

fn execute_native(
    checked: &invar::check::CheckedProgram,
    path: &OsString,
    json: bool,
    fuel: u64,
    c_library: Option<&OsString>,
) -> i32 {
    let program = match invar::mir::lower(checked) {
        Ok(program) => program,
        Err(error) => return report_front(FrontError::tool(error.code, error.message), path, json),
    };
    match invar::engine::run_native_with(
        &program,
        fuel,
        &path.to_string_lossy(),
        &diagnostic_source(path),
        c_library.map(std::path::Path::new),
        native_command::debug(),
    ) {
        Ok((status, document)) => {
            if json {
                if print_run_json(document, path) == 2 {
                    return 2;
                }
            } else if status == 0 {
                println!("Result: {}", document["result"]["value"]);
            } else {
                eprintln!("{}: {document}", path.to_string_lossy());
            }
            status
        }
        Err(error) => report_eval(error, path, json),
    }
}

fn report_eval(error: EvalError, path: &OsString, json: bool) -> i32 {
    if json {
        println!(
            "{}",
            error.to_diagnostics_json(&path.to_string_lossy(), &diagnostic_source(path))
        );
    } else {
        eprintln!("{}: {error:?}", path.to_string_lossy());
    }
    match error {
        EvalError::Trap { .. } => 3,
        EvalError::EvaluationLimit { .. } => 4,
        EvalError::Tool { .. } => 2,
    }
}

fn diagnostic_source(path: &OsString) -> Vec<u8> {
    let mut source = Vec::new();
    if let Ok(file) = std::fs::File::open(path) {
        let _ = file.take(1_048_577).read_to_end(&mut source);
    }
    source
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;

    #[cfg(feature = "native")]
    #[test]
    fn issue_ai_01_machine_diagnostics_native_build_fault() {
        let error = native_command::build_error("injected linker failure".into());
        let document = error.to_diagnostics_json("fixture.inv", b"");
        assert_eq!(document["diagnostics"][0]["code"], "NATIVE_BUILD");
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/invar-diagnostics-v3.schema.json"))
                .unwrap();
        assert!(jsonschema::is_valid(&schema, &document));
    }

    #[test]
    fn issue_ai_01_machine_diagnostics_document_size_limit() {
        let oversized = serde_json::json!({"payload": "x".repeat(1_000_001)});
        assert_eq!(print_run_json(oversized, &OsString::from("x.inv")), 2);
    }
}
