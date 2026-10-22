//! PIC scalar objects and a separate returning-failure host. Legacy executable
//! callbacks are never linked into these libraries.
use super::*;
use std::collections::HashSet;

pub(crate) fn linkage() -> &'static str {
    if cfg!(target_os = "macos") {
        "MachO-twolevel"
    } else {
        "ELF-Bsymbolic"
    }
}

pub(crate) fn configuration() -> Result<(String, String, String), String> {
    Target::initialize_native(&InitializationConfig::default())?;
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
    let machine = target
        .create_target_machine(
            &triple,
            "generic",
            "",
            OptimizationLevel::Default,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or("LLVM could not create the export target machine")?;
    let layout = machine.get_target_data().get_data_layout();
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let mut command = Command::new(rustc);
    command.arg("-vV");
    let version = wait_bounded(command)?;
    if !version.status.success() {
        return Err("export rustc version probe failed".into());
    }
    Ok((
        triple.as_str().to_string_lossy().into_owned(),
        layout.as_str().to_string_lossy().into_owned(),
        String::from_utf8(version.stdout).map_err(|e| e.to_string())?,
    ))
}

pub(crate) fn build(
    program: &mir::Program,
    selected: &[usize],
    reachable: &HashSet<usize>,
    input_identity: &[u8; 32],
    output: &Path,
) -> Result<Object, String> {
    if !cfg!(any(target_os = "linux", target_os = "macos")) {
        return Err("scalar shared exports require Linux or macOS".into());
    }
    if output.exists() {
        return Err("export library output already exists".into());
    }
    let hex = input_identity
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let namespace = format!("invar_{hex}_");
    let object = object_from_ir(llvm_ir_mode(
        program,
        Some((selected, reachable, &namespace)),
    )?)?;
    if object.ir.contains("unreachable") || object.ir.contains("alloca ") {
        return Err("scalar export emitted a terminating failure or native value frame".into());
    }
    let directory = unique_directory()?;
    let result = (|| {
        let object_path = directory.join("export.o");
        let host_path = directory.join("export_host.rs");
        fs::write(&object_path, &object.bytes).map_err(|e| e.to_string())?;
        let mut host = include_str!("export_host.rs").replace("inv_private_namespace", &namespace);
        writeln!(host, "const INPUT_IDENTITY: [u8;32] = {input_identity:?};")
            .map_err(|e| e.to_string())?;
        let signatures = selected
            .iter()
            .map(|id| {
                let function = &program.functions[*id];
                let tags = function
                    .params
                    .iter()
                    .map(|p| parameter_tag(p).map(|tag| tag.to_string()))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(", ");
                Ok(format!("(&[{tags}], {})", foreign_tag(&function.result)?))
            })
            .collect::<Result<Vec<_>, String>>()?
            .join(", ");
        writeln!(
            host,
            "const SIGNATURES: &[(&[u32], u32)] = &[{signatures}];"
        )
        .map_err(|e| e.to_string())?;
        host.push_str("unsafe extern \"C\" {\n");
        for (ordinal, id) in selected.iter().enumerate() {
            let args = (0..program.functions[*id].params.len())
                .map(|n| format!(", a{n}: i64"))
                .collect::<String>();
            writeln!(host, "#[link_name = \"{namespace}entry_{ordinal}\"] fn entry_{ordinal}(out: *mut i64{args}) -> i32;")
                .map_err(|e| e.to_string())?;
        }
        host.push_str("}\nunsafe fn call_entry(ordinal: usize, args: &[i64;8], out: *mut i64) -> i32 {\n match ordinal {\n");
        for (ordinal, id) in selected.iter().enumerate() {
            let args = (0..program.functions[*id].params.len())
                .map(|n| format!(", args[{n}]"))
                .collect::<String>();
            // All unsafe generated entry calls share the same reviewed invariants
            // documented at dispatch: exact signatures and valid output storage.
            writeln!(
                host,
                "{ordinal} => unsafe {{ entry_{ordinal}(out{args}) }},"
            )
            .map_err(|e| e.to_string())?;
        }
        host.push_str("_ => { tool(1); 1 }\n }\n}\n");
        for (ordinal, id) in selected.iter().enumerate() {
            // HA-1.2: a function with array parameters is reachable only
            // through the private export_call_arrays_v1 entry.
            if has_array_parameters(&program.functions[*id]) {
                continue;
            }
            writeln!(host, "// SAFETY: public C callers uphold pointer validity; dispatch checks all scalar wire fields.\n#[unsafe(no_mangle)] pub unsafe extern \"C\" fn inv_export_{ordinal}(args: *const Arg, argc: u32, out: *mut Output) -> i32 {{ unsafe {{ dispatch({ordinal}, args, argc, out) }} }}")
                .map_err(|e| e.to_string())?;
        }
        fs::write(&host_path, host).map_err(|e| e.to_string())?;
        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let linked_library = directory.join(format!("library.{}", std::env::consts::DLL_EXTENSION));
        let mut command = Command::new(rustc);
        command
            .arg("--edition=2024")
            .arg("--crate-type=cdylib")
            .arg("--crate-name")
            .arg(format!("invar_export_{hex}"))
            .arg("-C")
            .arg("opt-level=2")
            .arg("-C")
            .arg("panic=abort")
            .arg("-C")
            .arg("relocation-model=pic")
            .arg("-C")
            .arg(if cfg!(target_os = "macos") {
                "link-arg=-Wl,-twolevel_namespace"
            } else {
                "link-arg=-Wl,-Bsymbolic"
            })
            .arg("-C")
            .arg(format!("link-arg={}", object_path.display()))
            .arg("-o")
            .arg(&linked_library)
            .arg(&host_path);
        let linked = wait_bounded(command)?;
        if !linked.status.success() {
            return Err(format!(
                "returning export host link failed: {}",
                String::from_utf8_lossy(&linked.stderr)
            ));
        }
        let mut source = fs::File::open(linked_library).map_err(|e| e.to_string())?;
        let mut destination = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)
            .map_err(|e| e.to_string())?;
        if let Err(e) = std::io::copy(&mut source, &mut destination) {
            let _ = fs::remove_file(output);
            return Err(e.to_string());
        }
        Ok(())
    })();
    let _ = fs::remove_dir_all(directory);
    result.map(|()| object)
}

fn has_array_parameters(function: &mir::Function) -> bool {
    function
        .params
        .iter()
        .any(|p| matches!(p.ty, Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32))
}

/// Public scalar tag, or the private HA-1.2 descriptor tag of an admitted
/// `read`/`mut` `Array<i32|u32>` parameter (never a public `inv_arg_v1` tag).
pub(crate) fn parameter_tag(param: &mir::Parameter) -> Result<u32, String> {
    use crate::front::Mode;
    use invar_foreign::host_array as host;
    let element = match param.ty {
        Type::ArrayI32 => host::ELEMENT_I32,
        Type::ArrayU32 => host::ELEMENT_U32,
        Type::ArrayF32 => host::ELEMENT_F32,
        _ => return foreign_tag(&param.ty),
    };
    let mode = match param.mode {
        Some(Mode::Read) => host::MODE_READ,
        Some(Mode::Mut) => host::MODE_MUT,
        _ => return Err(error("export array parameter mode is unsupported")),
    };
    Ok(host::descriptor_tag(element, mode))
}

#[cfg(test)]
#[test]
fn issue_l1_06_f32_export_host_record_validation() {
    let directory = unique_directory().unwrap();
    let source = directory.join("record_tests.rs");
    let executable = directory.join("record_tests");
    let mut host = include_str!("export_host.rs").to_owned();
    host.push_str(
        r#"
const INPUT_IDENTITY: [u8; 32] = [0; 32];
const SIGNATURES: &[(&[u32], u32)] = &[];
unsafe fn call_entry(_: usize, _: &[i64; 8], _: *mut i64) -> i32 { 1 }
#[test]
fn f32_record_metadata() {
    let mut xs = [0u32; 4];
    let record = HostArray { element: 3, mode: 1, data: xs.as_mut_ptr().cast(), len: 4 };
    assert!(entry_for(0x131, record).is_some());
    assert!(entry_for(0x133, HostArray { mode: 3, ..record }).is_some());
    assert!(entry_for(0x133, record).is_none());
    assert!(entry_for(0x121, record).is_none());
    assert!(entry_for(0x141, record).is_none());
    for bad in [
        HostArray { element: 2, ..record }, HostArray { mode: 2, ..record },
        HostArray { len: 1_000_001, ..record }, HostArray { data: ptr::null_mut(), ..record },
        HostArray { len: 0, ..record },
        HostArray { data: ptr::without_provenance_mut(1), ..record },
        HostArray { data: ptr::without_provenance_mut(usize::MAX - 3), len: 2, ..record },
    ] { assert!(entry_for(0x131, bad).is_none()); }
    assert!(entry_for(0x131, HostArray { data: ptr::null_mut(), len: 0, ..record }).is_some());
}
"#,
    );
    fs::write(&source, host).unwrap();
    let mut compiler = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()));
    compiler
        .args(["--edition=2024", "--test"])
        .arg(&source)
        .arg("-o")
        .arg(&executable);
    let result = wait_bounded(compiler).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let result = wait_bounded(Command::new(executable)).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    fs::remove_dir_all(directory).unwrap();
}
