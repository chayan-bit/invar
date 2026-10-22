//! DX-20: DWARF 5 from the native backend. The object's DWARF is read
//! back with gimli; debug info must leave the code itself unchanged.
#![cfg(feature = "native")]

use std::{borrow::Cow, collections::BTreeMap, path::Path};

use invar::{check::check, front::parse, mir, native};
use object::{Object as _, ObjectSection as _, ObjectSymbol as _, RelocationTarget};

const PATH: &str = "/work/fixture/debug.inv";
const SOURCE: &str = "\
struct Pair { left: i32, right: u32 }

enum Choice { Low, High(i32) }

fn identity<T: Value>(x: T) -> T {
    return x;
}

fn mix(a: i32, b: u32, c: f32, d: f64, flag: Bool, xs: read Array<i32>, pair: take Pair) -> i32 {
    let acc: i32 = a + xs[0u32] + pair.left;
    if flag and b == 8u32 and c == 1.5 and d == 2.5f64 {
        return identity(acc);
    }
    return 0;
}

fn nested(n: i32) -> i32 {
    var out: i32 = 0;
    if n > 0 {
        let inner: i32 = n * 2;
        out = inner;
    }
    return out;
}

fn walk(xs: read Array<i32>) -> i32 {
    var sum: i32 = 0;
    for i in 0u32..len(xs) {
        sum = sum + xs[i];
    }
    return sum;
}

fn pick(choice: take Choice) -> i32 {
    return match move choice { Choice::Low => 0i32, Choice::High(number) => number };
}

fn main() -> i32 {
    let xs: Array<i32> = [1, 2, 3];
    let pair: Pair = Pair { left: 4, right: 5u32 };
    let scale: f32 = 1.5;
    let wide: f64 = 2.5f64;
    let result: i32 = mix(7, 8u32, scale, wide, true, xs, move pair);
    let choice: Choice = Choice::Low();
    return result + nested(0) + walk(xs) - 6 + pick(move choice);
}
";

fn lower(source: &str) -> Result<mir::Program, String> {
    let parsed = parse(source.as_bytes()).map_err(|error| format!("{error:?}"))?;
    let checked = check(parsed, true).map_err(|error| format!("{error:?}"))?;
    mir::lower(&checked).map_err(|error| format!("{error:?}"))
}

/// The first differing line of two texts, for readable failures.
fn first_difference(left: &str, right: &str) -> Option<(String, String)> {
    let mut lines = left.lines().zip(right.lines());
    lines
        .find(|(a, b)| a != b)
        .map(|(a, b)| (a.to_owned(), b.to_owned()))
        .or_else(|| (left != right).then(|| (left.len().to_string(), right.len().to_string())))
}

fn debug_object(source: &str, path: &str) -> Result<native::Object, String> {
    let program = lower(source).unwrap();
    native::emit_mir_debug(
        &program,
        &native::DebugSource {
            path: Path::new(path),
            text: source.as_bytes(),
        },
    )
}

/// One-based line and column (characters) of the first `needle` in SOURCE.
fn position(needle: &str) -> (u64, u64) {
    let offset = SOURCE.find(needle).expect("needle in fixture");
    let prefix = &SOURCE[..offset];
    let line = prefix.matches('\n').count() as u64 + 1;
    let column = prefix.rsplit('\n').next().unwrap().chars().count() as u64 + 1;
    (line, column)
}

/// A relocatable object's section with its absolute relocations applied, so
/// DWARF offsets into other sections read correctly.
fn section(file: &object::File<'_>, name: &str) -> Vec<u8> {
    let Some(section) = file.section_by_name(name) else {
        return Vec::new();
    };
    let mut data = section.uncompressed_data().unwrap().into_owned();
    for (offset, relocation) in section.relocations() {
        let base = match relocation.target() {
            RelocationTarget::Symbol(index) => file.symbol_by_index(index).unwrap().address(),
            RelocationTarget::Section(index) => file.section_by_index(index).unwrap().address(),
            other => panic!("unexpected relocation target {other:?} in {name}"),
        };
        let size = usize::from(relocation.size() / 8);
        let field = &mut data[offset as usize..offset as usize + size];
        let mut implicit = [0u8; 8];
        if relocation.has_implicit_addend() {
            implicit[..size].copy_from_slice(field);
        }
        let value = base
            .wrapping_add(relocation.addend() as u64)
            .wrapping_add(u64::from_le_bytes(implicit));
        field.copy_from_slice(&value.to_le_bytes()[..size]);
    }
    data
}

#[derive(Debug, Default)]
struct Subprogram {
    linkage: String,
    /// (tag, name, type name) of each parameter and variable; a variable's
    /// tag says whether it sits in a lexical block.
    variables: Vec<(String, String, String)>,
}

#[derive(Debug, Default)]
struct Dwarf {
    version: u16,
    files: Vec<String>,
    rows: Vec<(u64, u64)>,
    subprograms: BTreeMap<String, Subprogram>,
    /// Structure name to member names.
    structures: BTreeMap<String, Vec<String>>,
}

type Reader<'a> = gimli::EndianSlice<'a, gimli::RunTimeEndian>;

fn string<'a>(
    dwarf: &gimli::Dwarf<Reader<'a>>,
    unit: &gimli::Unit<Reader<'a>>,
    value: gimli::AttributeValue<Reader<'a>>,
) -> String {
    dwarf
        .attr_string(unit, value)
        .unwrap()
        .to_string_lossy()
        .into_owned()
}

fn name_of<'a>(
    dwarf: &gimli::Dwarf<Reader<'a>>,
    unit: &gimli::Unit<Reader<'a>>,
    entry: &gimli::DebuggingInformationEntry<Reader<'a>>,
    attr: gimli::DwAt,
) -> String {
    entry
        .attr_value(attr)
        .map(|value| string(dwarf, unit, value))
        .unwrap_or_default()
}

fn read_dwarf(bytes: &[u8]) -> Dwarf {
    let file = object::File::parse(bytes).unwrap();
    let sections = gimli::DwarfSections::load(|id| {
        Ok::<_, gimli::Error>(Cow::<[u8]>::Owned(section(&file, id.name())))
    })
    .unwrap();
    let dwarf = sections.borrow(|data| gimli::EndianSlice::new(data, gimli::RunTimeEndian::Little));
    let mut out = Dwarf::default();
    let mut units = dwarf.units();
    while let Some(header) = units.next().unwrap() {
        out.version = header.version();
        let unit = dwarf.unit(header).unwrap();
        if let Some(program) = unit.line_program.clone() {
            let header = program.header().clone();
            for entry in header.file_names() {
                out.files.push(string(&dwarf, &unit, entry.path_name()));
            }
            let mut rows = program.rows();
            while let Some((_, row)) = rows.next_row().unwrap() {
                let column = match row.column() {
                    gimli::ColumnType::LeftEdge => 0,
                    gimli::ColumnType::Column(column) => column.get(),
                };
                out.rows
                    .push((row.line().map_or(0, |line| line.get()), column));
            }
        }
        let mut entries = unit.entries();
        let mut current: Option<(isize, String)> = None;
        let mut structure: Option<(isize, String)> = None;
        while let Some(entry) = entries.next_dfs().unwrap() {
            let depth = entry.depth();
            if current.as_ref().is_some_and(|(at, _)| depth <= *at) {
                current = None;
            }
            if structure.as_ref().is_some_and(|(at, _)| depth <= *at) {
                structure = None;
            }
            let name = name_of(&dwarf, &unit, entry, gimli::DW_AT_name);
            match entry.tag() {
                gimli::DW_TAG_subprogram => {
                    let linkage = name_of(&dwarf, &unit, entry, gimli::DW_AT_linkage_name);
                    out.subprograms.insert(
                        name.clone(),
                        Subprogram {
                            linkage,
                            variables: Vec::new(),
                        },
                    );
                    current = Some((depth, name));
                }
                gimli::DW_TAG_structure_type => {
                    out.structures.entry(name.clone()).or_default();
                    structure = Some((depth, name));
                }
                gimli::DW_TAG_member => {
                    if let Some((_, owner)) = &structure {
                        out.structures.get_mut(owner).unwrap().push(name);
                    }
                }
                tag @ (gimli::DW_TAG_formal_parameter | gimli::DW_TAG_variable) => {
                    // Directly under the subprogram, or inside lexical blocks.
                    let blocks = current.as_ref().map_or(0, |(at, _)| depth - at - 1);
                    let ty = match entry.attr_value(gimli::DW_AT_type) {
                        Some(gimli::AttributeValue::UnitRef(offset)) => {
                            let target = unit.entry(offset).unwrap();
                            name_of(&dwarf, &unit, &target, gimli::DW_AT_name)
                        }
                        _ => String::new(),
                    };
                    // Not the depth: LLVM may fold a block whose only child
                    // is another block, and whether it does depends on codegen.
                    let tag = if tag == gimli::DW_TAG_variable {
                        format!("{}variable", if blocks > 0 { "block " } else { "" })
                    } else {
                        "parameter".to_owned()
                    };
                    if let Some((_, owner)) = &current {
                        out.subprograms
                            .get_mut(owner)
                            .unwrap()
                            .variables
                            .push((tag, name, ty));
                    }
                }
                _ => {}
            }
        }
    }
    out
}

#[cfg(target_os = "linux")]
#[test]
fn issue_dx_20_debug_info_dwarf_describes_the_fixture() {
    let object = debug_object(SOURCE, PATH).unwrap();
    let dwarf = read_dwarf(&object.bytes);
    assert_eq!(dwarf.version, 5);
    // gdb takes a `.debug_names` index as covering every unit, so one here
    // would hide the host's DWARF linked beside this object.
    let file = object::File::parse(&*object.bytes).unwrap();
    assert!(file.section_by_name(".debug_names").is_none());
    assert!(
        dwarf.files.iter().any(|file| file.ends_with("debug.inv")),
        "{:?}",
        dwarf.files
    );

    // The line table maps the add in `mix` to its source line and column.
    let add = position("a + xs[0u32]");
    assert!(dwarf.rows.contains(&add), "{add:?} not in {:?}", dwarf.rows);
    let call = position("mix(7, 8u32");
    assert!(
        dwarf.rows.contains(&call),
        "{call:?} not in {:?}",
        dwarf.rows
    );

    // No linkage names: debuggers show source names, not `invar_fN`.
    assert!(
        dwarf.subprograms.values().all(|sub| sub.linkage.is_empty()),
        "{:?}",
        dwarf.subprograms
    );
    assert!(dwarf.subprograms.contains_key("main"));
    assert!(
        dwarf.subprograms.contains_key("identity<i32>"),
        "{:?}",
        dwarf.subprograms.keys()
    );

    let variables = |name: &str| {
        let mut list = dwarf.subprograms[name]
            .variables
            .iter()
            .map(|(tag, name, ty)| format!("{tag} {name}: {ty}"))
            .collect::<Vec<_>>();
        list.sort();
        list
    };
    assert_eq!(
        variables("mix"),
        [
            "parameter a: i32",
            "parameter b: u32",
            "parameter c: f32",
            "parameter d: f64",
            "parameter flag: Bool",
            "parameter pair: Pair",
            "parameter xs: Array<i32>",
            "variable acc: i32",
        ]
    );
    assert_eq!(
        variables("main"),
        [
            "variable choice: Choice",
            "variable pair: Pair",
            "variable result: i32",
            "variable scale: f32",
            "variable wide: f64",
            "variable xs: Array<i32>",
        ]
    );
    assert_eq!(variables("identity<i32>"), ["parameter x: i32"]);
    // A local bound in a nested block lives in a lexical block (DWARF emitter review).
    assert_eq!(
        variables("nested"),
        [
            "block variable inner: i32",
            "parameter n: i32",
            "variable out: i32"
        ]
    );
    // A loop variable lives in the loop body, a match payload in its arm
    // (lexical scope review).
    assert_eq!(
        variables("walk"),
        [
            "block variable i: u32",
            "parameter xs: Array<i32>",
            "variable sum: i32"
        ]
    );
    // Which arm is checked in the IR (`..._scopes_a_match_payload_to_its_arm`).
    assert_eq!(
        variables("pick"),
        ["block variable number: i32", "parameter choice: Choice"]
    );

    // Arrays and records are host handles: an 8-byte structure named by the
    // Invar type with one `invar_handle` member.
    assert_eq!(dwarf.structures["Array<i32>"], ["invar_handle"]);
    assert_eq!(dwarf.structures["Pair"], ["invar_handle"]);
}

/// Example programs the native backend accepts, sorted by path.
/// Line and column of the lexical block that scopes local `name` in the IR.
fn ir_scope(ir: &str, name: &str) -> (u64, u64) {
    let variable = ir
        .lines()
        .find(|line| line.contains(&format!("!DILocalVariable(name: \"{name}\"")))
        .expect("variable in the IR");
    let scope = variable
        .split("scope: ")
        .nth(1)
        .unwrap()
        .split(',')
        .next()
        .unwrap();
    let block = ir
        .lines()
        .find(|line| line.starts_with(&format!("{scope} = distinct !DILexicalBlock(")))
        .expect("lexical block scope");
    let field = |key: &str| {
        let value = block.split(&format!("{key}: ")).nth(1).unwrap();
        value.split([',', ')']).next().unwrap().parse().unwrap()
    };
    (field("line"), field("column"))
}

/// A match payload is scoped to its own arm, not the whole match. Checked in
/// the IR: LLVM may fold the match's block into the arm's in the DWARF.
#[test]
fn issue_dx_20_debug_info_scopes_a_match_payload_to_its_arm() {
    let object = debug_object(SOURCE, PATH).unwrap();
    assert_eq!(
        ir_scope(&object.ir, "number"),
        position("Choice::High(number)")
    );
}

fn corpus() -> Vec<(String, String)> {
    fn walk(directory: &Path, found: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, found);
            } else if path.extension().is_some_and(|ext| ext == "inv") {
                found.push(path);
            }
        }
    }
    let mut paths = Vec::new();
    walk(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("examples"),
        &mut paths,
    );
    paths.sort();
    paths
        .into_iter()
        .filter_map(|path| {
            let source = std::fs::read_to_string(&path).ok()?;
            let program = lower(&source).ok()?;
            native::emit_mir(&program).ok()?;
            Some((path.to_string_lossy().into_owned(), source))
        })
        .collect()
}

fn text_section(bytes: &[u8]) -> Vec<u8> {
    let file = object::File::parse(bytes).unwrap();
    let section = file
        .section_by_name(".text")
        .or_else(|| file.section_by_name("__text"))
        .expect("object has a text section");
    section.data().unwrap().to_vec()
}

#[test]
fn issue_dx_20_debug_info_is_code_neutral() {
    let mut programs = corpus();
    programs.push((PATH.to_owned(), SOURCE.to_owned()));
    assert!(
        programs.len() >= 31,
        "corpus has {} programs",
        programs.len()
    );
    for (path, source) in &programs {
        let program = lower(source).unwrap();
        let plain = native::emit_mir(&program).unwrap();
        let debug = debug_object(source, path).unwrap();
        assert_ne!(plain.ir, debug.ir, "{path}: no debug metadata");
        let difference = first_difference(
            &native::canonical_ir(&plain.ir, false).unwrap(),
            &native::canonical_ir(&debug.ir, true).unwrap(),
        );
        assert_eq!(
            difference, None,
            "{path}: IR differs once debug info is stripped"
        );
        assert_eq!(
            text_section(&plain.bytes),
            text_section(&debug.bytes),
            "{path}: .text differs"
        );
    }
}

#[test]
fn issue_dx_20_debug_info_refuses_relative_paths_and_bad_sources() {
    let program = lower(SOURCE).unwrap();
    let relative = native::emit_mir_debug(
        &program,
        &native::DebugSource {
            path: Path::new("debug.inv"),
            text: SOURCE.as_bytes(),
        },
    );
    assert_eq!(
        relative.unwrap_err(),
        "native debug source path must be absolute"
    );
    let invalid = native::emit_mir_debug(
        &program,
        &native::DebugSource {
            path: Path::new(PATH),
            text: b"\xff",
        },
    );
    assert_eq!(invalid.unwrap_err(), "native debug source must be UTF-8");
}

#[test]
fn issue_dx_20_debug_info_escapes_file_names() {
    let object = debug_object(SOURCE, "/work/odd \"dir\"/caf\u{e9}.inv").unwrap();
    assert!(
        object
            .ir
            .contains(r#"!DIFile(filename: "caf\C3\A9.inv", directory: "/work/odd \22dir\22")"#),
        "{}",
        object.ir
    );
}

#[path = "support/scratch.rs"]
mod scratch;

const TRAP: &str = "fn main() -> i32 {\n    let x: i32 = 2147483647;\n    return x + 1;\n}\n";
const LOOP: &str = "fn main() -> i32 {\n    var n: i32 = 0;\n    for i in 0u32..1000u32 {\n        n = n + 1;\n    }\n    return n;\n}\n";

/// Value, trap and step count are the same with and without DWARF.
#[test]
fn issue_dx_20_debug_info_keeps_behaviour() {
    for (source, fuel) in [(SOURCE, 10_000), (TRAP, 10_000), (LOOP, 50)] {
        let program = lower(source).unwrap();
        let plain = native::run_program(&program, fuel).unwrap();
        let debug = native::run_program_debug(
            &program,
            fuel,
            None,
            &native::DebugSource {
                path: Path::new(PATH),
                text: source.as_bytes(),
            },
        )
        .unwrap();
        assert_eq!(plain, debug, "{source}");
    }
}

fn invar(args: &[&str]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn issue_dx_20_debug_info_cli_build_and_run() {
    let directory = std::env::temp_dir().join(format!("invar-dx-20-{}", scratch::unique()));
    std::fs::create_dir(&directory).unwrap();
    let source = directory.join("debug.inv");
    std::fs::write(&source, SOURCE).unwrap();
    let source = source.to_str().unwrap();
    let output = directory.join("program");
    let output = output.to_str().unwrap();

    let build = invar(&[
        "build", source, "--target", "native", "--output", output, "--debug",
    ]);
    let stdout = String::from_utf8_lossy(&build.stdout);
    assert_eq!(build.status.code(), Some(0), "{build:?}");
    assert!(stdout.contains("Debug info: DWARF 5"), "{stdout}");
    let direct = std::process::Command::new(output).output().unwrap();
    assert_eq!(direct.status.code(), Some(0));
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&direct.stdout).unwrap()["result"]["value"],
        12
    );
    if cfg!(target_os = "macos") {
        assert!(Path::new(&format!("{output}.dSYM")).is_dir());
    } else {
        // The linked executable carries this compile unit's line table.
        let bytes = std::fs::read(output).unwrap();
        let file = object::File::parse(&*bytes).unwrap();
        let strings = [".debug_line_str", ".debug_str"]
            .iter()
            .filter_map(|name| file.section_by_name(name))
            .any(|section| {
                let data = section.data().unwrap();
                data.windows(9).any(|window| window == b"debug.inv")
            });
        assert!(strings, "no debug.inv in the executable's DWARF strings");
    }

    let plain = invar(&[
        "run",
        source,
        "--engine",
        "native-serial",
        "--format",
        "json",
    ]);
    let debug = invar(&["run", source, "--debug", "--format", "json"]);
    assert_eq!(debug.status.code(), Some(0), "{debug:?}");
    assert!(
        String::from_utf8_lossy(&debug.stderr).contains("--debug implies --engine native-serial")
    );
    let result = |output: &std::process::Output| {
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap()["result"].clone()
    };
    assert_eq!(result(&plain), result(&debug));
    assert_eq!(result(&debug)["value"], 12);

    for args in [
        &[
            "run",
            source,
            "--engine",
            "cpu-parallel",
            "--workers",
            "2",
            "--debug",
            "--format",
            "json",
        ][..],
        &[
            "build", source, "--target", "metal", "--output", output, "--debug", "--format", "json",
        ][..],
    ] {
        let refused = invar(args);
        assert_eq!(refused.status.code(), Some(2), "{args:?}");
        let text = String::from_utf8_lossy(&refused.stdout);
        assert!(text.contains("DEBUG_UNSUPPORTED"), "{args:?}: {text}");
    }
    std::fs::remove_dir_all(directory).unwrap();
}

/// macOS publishes `OUT.dSYM` before `OUT`, so when the symbols cannot be
/// published no executable is left behind without them (`--debug` review).
#[cfg(target_os = "macos")]
#[test]
fn issue_dx_20_debug_info_never_publishes_an_executable_without_symbols() {
    let directory = std::env::temp_dir().join(format!("invar-dx-20-{}", scratch::unique()));
    std::fs::create_dir(&directory).unwrap();
    let output = directory.join("program");
    let symbols = directory.join("program.dSYM");
    std::fs::create_dir(&symbols).unwrap();
    std::fs::write(symbols.join("keep"), b"user data").unwrap();
    let object = debug_object(SOURCE, PATH).unwrap();
    assert_eq!(
        native::link(&object, &output).unwrap_err(),
        "native debug symbols output already exists"
    );
    assert!(!output.exists());
    assert_eq!(std::fs::read(symbols.join("keep")).unwrap(), b"user data");
    std::fs::remove_dir_all(&symbols).unwrap();
    native::link(&object, &output).unwrap();
    assert!(output.is_file() && symbols.is_dir());
    std::fs::remove_dir_all(directory).unwrap();
}
