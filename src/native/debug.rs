//! DWARF 5 for native debug builds (DX-20). The emitter writes LLVM IR as text,
//! so debug info is textual metadata beside it: one compile unit and file, a
//! subprogram per lowered function, a location per IR instruction from its MIR
//! span, and a variable for every parameter and named local. LLVM lowers the
//! metadata to DWARF; the code it describes is the same as without it.

use std::{
    collections::HashMap,
    fmt::Write as _,
    path::{Path, PathBuf},
    process::Command,
};

use crate::{
    front::{Span, Type},
    mir::{Function, Operation, Program},
};

/// The source file a debug build maps MIR spans back to.
pub struct Source<'a> {
    /// Absolute path recorded as the compile unit's file.
    pub path: &'a Path,
    pub text: &'a [u8],
}

pub(super) struct Module<'a> {
    text: &'a str,
    line_starts: Vec<usize>,
    nodes: Vec<String>,
    unique: HashMap<String, usize>,
    types: HashMap<String, usize>,
    unit: usize,
    file: usize,
    /// Every `{`...`}` pair of the source.
    braces: Vec<Span>,
    /// The current function's lexical blocks: source block and scope node.
    blocks: Vec<(Span, usize)>,
    /// The current function's named locals: binding span and scope node.
    bindings: HashMap<String, (Span, usize)>,
}

/// LLVM metadata string: printable ASCII except `"` and `\`, else `\XX`.
fn escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for byte in text.bytes() {
        if (0x20..0x7f).contains(&byte) && byte != b'"' && byte != b'\\' {
            escaped.push(char::from(byte));
        } else {
            let _ = write!(escaped, "\\{byte:02X}");
        }
    }
    escaped
}

/// Source-facing name of a lowered function: generic instances `f$<i32|u32>`
/// read `f<i32, u32>`.
pub(super) fn display_name(name: &str) -> String {
    name.replace("$<", "<").replace('|', ", ")
}

impl<'a> Module<'a> {
    pub(super) fn new(source: &Source<'a>) -> Result<Self, String> {
        // Scalars occupy the low bytes of their i64 slot (see `variable`).
        if !cfg!(target_endian = "little") {
            return Err("native debug info requires a little-endian target".into());
        }
        let text = std::str::from_utf8(source.text)
            .map_err(|_| "native debug source must be UTF-8".to_owned())?;
        let path = source
            .path
            .to_str()
            .ok_or("native debug source path must be UTF-8")?;
        if !source.path.is_absolute() {
            return Err("native debug source path must be absolute".into());
        }
        let (directory, filename) = match (source.path.parent(), source.path.file_name()) {
            (Some(parent), Some(name)) => (
                parent.to_str().unwrap_or_default(),
                name.to_str().unwrap_or_default(),
            ),
            _ => ("", path),
        };
        let line_starts = std::iter::once(0)
            .chain(text.match_indices('\n').map(|(at, _)| at + 1))
            .collect();
        let mut module = Self {
            text,
            line_starts,
            braces: crate::front::brace_spans(text).unwrap_or_default(),
            blocks: Vec::new(),
            bindings: HashMap::new(),
            nodes: Vec::new(),
            unique: HashMap::new(),
            types: HashMap::new(),
            unit: 0,
            file: 0,
        };
        module.file = module.node(format!(
            "!DIFile(filename: \"{}\", directory: \"{}\")",
            escape(filename),
            escape(directory)
        ));
        // No `.debug_names`: gdb takes a name index as covering every unit
        // and would then ignore the host's DWARF linked beside this one.
        module.unit = module.distinct(format!(
            "distinct !DICompileUnit(language: DW_LANG_C11, file: !{}, producer: \"invar {}\", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)",
            module.file,
            env!("CARGO_PKG_VERSION")
        ));
        Ok(module)
    }

    fn distinct(&mut self, text: String) -> usize {
        self.nodes.push(text);
        self.nodes.len() - 1
    }

    fn node(&mut self, text: String) -> usize {
        if let Some(id) = self.unique.get(&text) {
            return *id;
        }
        let id = self.distinct(text.clone());
        self.unique.insert(text, id);
        id
    }

    /// One-based line and column (in characters, as diagnostics count them)
    /// of a byte offset; line 0 when the offset is not a source position.
    fn position(&self, offset: usize) -> (usize, usize) {
        let line = self.line_starts.partition_point(|start| *start <= offset);
        match line
            .checked_sub(1)
            .and_then(|index| self.text.get(self.line_starts[index]..offset))
        {
            Some(prefix) => (line, prefix.chars().count() + 1),
            None => (0, 0),
        }
    }

    fn ty(&mut self, ty: &Type) -> Option<usize> {
        let name = ty.source_name().to_owned();
        if let Some(id) = self.types.get(&name) {
            return Some(*id);
        }
        let base = |name: &str, size: u32, encoding: &str| {
            format!("!DIBasicType(name: \"{name}\", size: {size}, encoding: {encoding})")
        };
        let id = match ty {
            Type::Unit => return None,
            Type::I32 => self.node(base("i32", 32, "DW_ATE_signed")),
            Type::U32 => self.node(base("u32", 32, "DW_ATE_unsigned")),
            Type::F32 => self.node(base("f32", 32, "DW_ATE_float")),
            Type::F64 => self.node(base("f64", 64, "DW_ATE_float")),
            Type::Bool => self.node(base("Bool", 8, "DW_ATE_boolean")),
            _ => {
                // Arrays, records, variants and views are host-owned handles in
                // an i64 slot (`native_host.rs`), so the type is an 8-byte
                // structure named by its Invar type, holding the handle.
                let word = self.node(base("u64", 64, "DW_ATE_unsigned"));
                let member = self.node(format!(
                    "!DIDerivedType(tag: DW_TAG_member, name: \"invar_handle\", file: !{}, baseType: !{word}, size: 64)",
                    self.file
                ));
                let members = self.node(format!("!{{!{member}}}"));
                self.node(format!(
                    "!DICompositeType(tag: DW_TAG_structure_type, name: \"{}\", file: !{}, size: 64, elements: !{members})",
                    escape(&name),
                    self.file
                ))
            }
        };
        self.types.insert(name, id);
        Some(id)
    }

    /// The subprogram for `function`, attached to its definition. It has no
    /// linkage name: debuggers then show the source name (`mix`,
    /// `identity<i32>`) rather than the symbol `invar_fN`.
    pub(super) fn subprogram(&mut self, function: &Function, helper: bool) -> usize {
        let line = if helper {
            0
        } else {
            self.position(function.span.start).0
        };
        let types = std::iter::once(&function.result)
            .chain(function.params.iter().map(|param| &param.ty))
            .map(|ty| match self.ty(ty) {
                Some(id) => format!("!{id}"),
                None => "null".into(),
            })
            .collect::<Vec<_>>()
            .join(", ");
        let signature = self.node(format!("!DISubroutineType(types: !{{{types}}})"));
        let flags = if helper {
            "DIFlagPrototyped | DIFlagArtificial"
        } else {
            "DIFlagPrototyped"
        };
        let subprogram = self.distinct(format!(
            "distinct !DISubprogram(name: \"{}\", scope: !{file}, file: !{file}, line: {line}, type: !{signature}, scopeLine: {line}, flags: {flags}, spFlags: DISPFlagDefinition, unit: !{})",
            escape(&display_name(&function.name)),
            self.unit,
            file = self.file,
        ));
        self.scopes(function, subprogram, helper);
        subprogram
    }

    /// Lexical blocks for the source blocks of `function` that bind a local
    /// (`Operation::Bind`), nested as in the source, so a debugger shows each
    /// local only inside its block. The function body is the subprogram.
    fn scopes(&mut self, function: &Function, subprogram: usize, helper: bool) {
        self.blocks.clear();
        self.bindings.clear();
        if helper {
            return;
        }
        let mut binds: Vec<(String, Vec<Span>)> = Vec::new();
        for instruction in function.blocks.iter().flat_map(|block| &block.instructions) {
            if let Operation::Bind { name, .. } = &instruction.kind {
                match binds.iter_mut().find(|(known, _)| known == name) {
                    Some((_, spans)) => spans.push(instruction.span),
                    None => binds.push((name.clone(), vec![instruction.span])),
                }
            }
        }
        for (name, spans) in binds {
            // The source regions holding the binding, outermost first: the
            // function's own brace pairs (its span covers only the
            // signature) except the body, then the binding's own region.
            let inside = |pair: &Span, at: usize| {
                function.span.start <= pair.start && pair.start < at && at < pair.end
            };
            let mut regions = self
                .braces
                .iter()
                .filter(|pair| spans.iter().all(|at| inside(pair, at.start)))
                .copied()
                .collect::<Vec<_>>();
            regions.sort_by_key(|pair| pair.start);
            if !regions.is_empty() {
                regions.remove(0);
            }
            if let [at] = spans[..] {
                let keyword = self.text.get(at.start..).unwrap_or_default();
                let starts = |word: &str| {
                    keyword.strip_prefix(word).is_some_and(|rest| {
                        !rest.starts_with(|next: char| next.is_alphanumeric() || next == '_')
                    })
                };
                if starts("for") {
                    // A loop variable lives in the body: the pair that ends
                    // the `for` statement.
                    regions.extend(
                        self.braces
                            .iter()
                            .copied()
                            .filter(|pair| at.start < pair.start && pair.end == at.end),
                    );
                } else if !starts("let") && !starts("var") {
                    // A match payload lives in its arm.
                    regions.push(at);
                }
            }
            let mut scope = subprogram;
            for region in regions {
                scope = match self.blocks.iter().find(|(known, _)| *known == region) {
                    Some((_, node)) => *node,
                    None => {
                        let (line, column) = self.position(region.start);
                        let node = self.distinct(format!(
                            "distinct !DILexicalBlock(scope: !{scope}, file: !{}, line: {line}, column: {column})",
                            self.file
                        ));
                        self.blocks.push((region, node));
                        node
                    }
                };
            }
            self.bindings.insert(name, (spans[0], scope));
        }
    }

    /// The location node of `span` in the innermost lexical block holding
    /// it, else in `scope`; frame helpers' synthetic spans map to line 0.
    pub(super) fn location(&mut self, span: Span, scope: usize, helper: bool) -> usize {
        let (line, column) = if helper {
            (0, 0)
        } else {
            self.position(span.start)
        };
        let scope = self
            .blocks
            .iter()
            .filter(|(pair, _)| pair.start <= span.start && span.start < pair.end)
            .max_by_key(|(pair, _)| pair.start)
            .map_or(scope, |(_, node)| *node);
        self.node(format!(
            "!DILocation(line: {line}, column: {column}, scope: !{scope})"
        ))
    }

    /// `#dbg_declare` records for the parameters and named locals of
    /// `function`, whose i64 slots `%s{slot}` follow `declaration_order`.
    pub(super) fn declarations(
        &mut self,
        function: &Function,
        scope: usize,
        helper: bool,
    ) -> Result<String, String> {
        let at = self.location(function.span, scope, helper);
        let line = if helper {
            0
        } else {
            self.position(function.span.start).0
        };
        let mut text = String::new();
        for (slot, name) in function.declaration_order.iter().enumerate() {
            let argument = function.params.iter().position(|param| param.name == *name);
            let ty = match argument {
                Some(index) => &function.params[index].ty,
                None => {
                    &function
                        .locals
                        .get(name)
                        .ok_or("verified MIR local missing in native debug info")?
                        .ty
                }
            };
            let Some(ty) = self.ty(ty) else { continue };
            let argument = argument.map_or(String::new(), |index| format!("arg: {}, ", index + 1));
            // LLVM places a declared variable in the scope of its record's
            // location, so a block local's record is located in its block.
            let (line, scope, at) = match self.bindings.get(name).copied() {
                Some((bound, block)) => {
                    let (line, column) = self.position(bound.start);
                    let at = self.node(format!(
                        "!DILocation(line: {line}, column: {column}, scope: !{block})"
                    ));
                    (line, block, at)
                }
                None => (line, scope, at),
            };
            let variable = self.node(format!(
                "!DILocalVariable(name: \"{}\", {argument}scope: !{scope}, file: !{}, line: {line}, type: !{ty})",
                escape(name),
                self.file
            ));
            let _ = writeln!(
                text,
                "    #dbg_declare(ptr %s{slot}, !{variable}, !DIExpression(), !{at})"
            );
        }
        Ok(text)
    }

    /// Module metadata: the compile unit list, DWARF 5 flags and every node.
    pub(super) fn finish(&self) -> String {
        let mut text = format!(
            "\n!llvm.dbg.cu = !{{!{}}}\n!llvm.module.flags = !{{!{flags}, !{}}}\n!{flags} = !{{i32 7, !\"Dwarf Version\", i32 5}}\n!{} = !{{i32 2, !\"Debug Info Version\", i32 3}}\n",
            self.unit,
            self.nodes.len() + 1,
            self.nodes.len() + 1,
            flags = self.nodes.len(),
        );
        for (id, node) in self.nodes.iter().enumerate() {
            let _ = writeln!(text, "!{id} = {node}");
        }
        text
    }
}

/// Compile with DWARF for `source`, then run like
/// [`run_program_with_library`](super::run_program_with_library).
pub fn run_program_debug(
    program: &Program,
    fuel: u64,
    library: Option<&Path>,
    source: &Source,
) -> Result<(i32, serde_json::Value), String> {
    super::run_object(&super::emit_mir_debug(program, source)?, fuel, library)
}

/// `<executable>.dSYM`, where macOS debuggers look for its DWARF.
fn symbols(executable: &Path) -> PathBuf {
    let mut path = executable.as_os_str().to_owned();
    path.push(".dSYM");
    PathBuf::from(path)
}

/// The host's table of Invar value layouts for `invar_debug_render_v1`
/// (`native_debug_host.rs`), named as in the DWARF: arrays by element,
/// non-generic records by field and variants by arm. Other named types
/// (views, generic records) render untyped.
pub(super) fn host_types(program: &Program) -> String {
    let arrays = ["i32", "u32", "f32", "f64", "Bool"];
    let records = program
        .records
        .iter()
        .filter(|record| record.generic_params.is_empty());
    let names = arrays
        .iter()
        .map(|element| format!("Array<{element}>"))
        .chain(records.clone().map(|record| record.name.clone()))
        .chain(program.variants.iter().map(|variant| variant.name.clone()))
        .collect::<Vec<_>>();
    let kind = |ty: &Type| match ty {
        Type::I32 => "DebugKind::I32".to_owned(),
        Type::U32 => "DebugKind::U32".to_owned(),
        Type::F32 => "DebugKind::F32".to_owned(),
        Type::F64 => "DebugKind::F64".to_owned(),
        Type::Bool => "DebugKind::Bool".to_owned(),
        Type::Unit => "DebugKind::Unit".to_owned(),
        Type::ArrayI32 => "DebugKind::Type(0)".to_owned(),
        Type::ArrayU32 => "DebugKind::Type(1)".to_owned(),
        Type::ArrayF32 => "DebugKind::Type(2)".to_owned(),
        Type::ArrayF64 => "DebugKind::Type(3)".to_owned(),
        Type::ArrayBool => "DebugKind::Type(4)".to_owned(),
        Type::Named(name) => names
            .iter()
            .position(|known| known == name)
            .map_or("DebugKind::Raw".to_owned(), |index| {
                format!("DebugKind::Type({index})")
            }),
    };
    let layouts = ["I32", "U32", "F32", "F64", "Bool"]
        .iter()
        .map(|element| format!("DebugLayout::Array(DebugKind::{element})"))
        .chain(records.map(|record| {
            let fields = record
                .fields
                .iter()
                .map(|field| format!("({:?}, {})", field.name, kind(&field.ty)));
            format!(
                "DebugLayout::Record(&[{}])",
                fields.collect::<Vec<_>>().join(", ")
            )
        }))
        .chain(program.variants.iter().map(|variant| {
            let arms = variant.arms.iter().map(|arm| {
                let payload = arm
                    .payload
                    .as_ref()
                    .map_or("None".to_owned(), |ty| format!("Some({})", kind(ty)));
                format!("({:?}, {payload})", arm.name)
            });
            format!(
                "DebugLayout::Variant(&[{}])",
                arms.collect::<Vec<_>>().join(", ")
            )
        }));
    let entries = names
        .iter()
        .zip(layouts)
        .map(|(name, layout)| format!("({name:?}, {layout})"));
    format!(
        "const DEBUG_TYPES: &[(&str, DebugLayout)] = &[{}];",
        entries.collect::<Vec<_>>().join(", ")
    )
}

/// The debugger hooks appended to the native host, with the layout table of
/// a `--debug` build (else an empty one).
pub(super) fn host_source(types: Option<&str>) -> String {
    let empty = "const DEBUG_TYPES: &[(&str, DebugLayout)] = &[];";
    format!(
        "{}\n{}\n",
        include_str!("../native_debug_host.rs"),
        types.unwrap_or(empty)
    )
}

/// rustc flags for the host of a `--debug` build. On macOS the host's DWARF
/// must stay in its object files (`unpacked`) until `bundle` runs.
pub(super) fn host_flags(debug: bool) -> &'static [&'static str] {
    match (debug, cfg!(target_os = "macos")) {
        (false, _) => &[],
        (true, false) => &["-g"],
        (true, true) => &["-g", "-Csplit-debuginfo=unpacked"],
    }
}

/// macOS links leave DWARF in the object file, which the link directory
/// removes, so `dsymutil` copies it beside the linked `executable` first.
/// Elsewhere the linker keeps DWARF in the executable itself.
pub(super) fn bundle(executable: &Path, output: &Path) -> Result<(), String> {
    if !cfg!(target_os = "macos") {
        return Ok(());
    }
    if symbols(output).exists() {
        return Err("native debug symbols output already exists".into());
    }
    let mut command = Command::new("dsymutil");
    command.arg(executable).arg("-o").arg(symbols(executable));
    let ran = super::wait_bounded(command)
        .map_err(|failure| format!("native debug build needs dsymutil on PATH: {failure}"))?;
    if !ran.status.success() {
        return Err(format!(
            "dsymutil failed: {}",
            String::from_utf8_lossy(&ran.stderr)
        ));
    }
    Ok(())
}

/// Remove the bundle [`publish_symbols`] placed beside `output`.
pub(super) fn retract_symbols(output: &Path) {
    if cfg!(target_os = "macos") {
        let _ = std::fs::remove_dir_all(symbols(output));
    }
}

/// Move the bundle from [`bundle`] beside the not yet published `output`.
pub(super) fn publish_symbols(linked: &Path, output: &Path) -> Result<(), String> {
    if !cfg!(target_os = "macos") {
        return Ok(());
    }
    std::fs::rename(symbols(linked), symbols(output))
        .map_err(|failure| format!("native debug symbols publication failed: {failure}"))
}

/// Test hook for code neutrality (DX-20): `ir` parsed and printed again by
/// LLVM, with all debug metadata stripped when `strip` is set.
#[doc(hidden)]
pub fn canonical_ir(ir: &str, strip: bool) -> Result<String, String> {
    let print = |ir: &str| {
        let context = inkwell::context::Context::create();
        let buffer = inkwell::memory_buffer::MemoryBuffer::create_from_memory_range_copy(
            ir.as_bytes(),
            "invar-native-v1",
        );
        let module = context
            .create_module_from_ir(buffer)
            .map_err(|error| error.to_string())?;
        if strip {
            module.strip_debug_info();
        }
        Ok::<_, String>(module.print_to_string().to_string())
    };
    let printed = print(ir)?;
    if !strip {
        return Ok(printed);
    }
    // LLVM's strip keeps module flags; drop the two DWARF flags this module
    // adds (plain native IR has no module flags), then print once more.
    let flags = [
        "!\"Dwarf Version\", i32 5}",
        "!\"Debug Info Version\", i32 3}",
    ];
    let kept = printed
        .lines()
        .filter(|line| {
            !line.starts_with("!llvm.module.flags = ")
                && !flags.iter().any(|flag| line.ends_with(flag))
        })
        .collect::<Vec<_>>()
        .join("\n");
    print(&kept)
}

#[cfg(test)]
mod tests {
    use crate::front::{Span, brace_spans};

    #[test]
    fn brace_spans_pair_blocks_and_skip_comments_and_strings() {
        let source = "fn f() { /* { */ let s = \"}\"; if t { g(); } } // }";
        let spans = brace_spans(source).unwrap();
        assert_eq!(
            spans,
            [Span { start: 35, end: 43 }, Span { start: 7, end: 45 }]
        );
        assert_eq!(&source[35..43], "{ g(); }");
    }

    #[test]
    fn host_types_lay_out_arrays_records_and_variants() {
        let source = "struct Pair { left: i32, scale: f32, xs: Array<Bool> }\n\
            enum Choice { Low, High(f64), Flag(Bool) }\n\
            fn main() -> i32 { return 0; }\n";
        let parsed = crate::front::parse(source.as_bytes()).unwrap();
        let program = crate::mir::lower(&crate::check::check(parsed, true).unwrap()).unwrap();
        let table = super::host_types(&program);
        for entry in [
            "(\"Array<f32>\", DebugLayout::Array(DebugKind::F32))",
            "(\"Pair\", DebugLayout::Record(&[(\"left\", DebugKind::I32), (\"scale\", DebugKind::F32), (\"xs\", DebugKind::Type(4))]))",
            "(\"Choice\", DebugLayout::Variant(&[(\"Low\", None), (\"High\", Some(DebugKind::F64)), (\"Flag\", Some(DebugKind::Bool))]))",
        ] {
            assert!(table.contains(entry), "{entry} not in {table}");
        }
        assert!(table.starts_with("const DEBUG_TYPES: &[(&str, DebugLayout)] = &[(\"Array<i32>\""));
    }
}
