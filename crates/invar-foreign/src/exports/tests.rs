use super::*;
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    directory: PathBuf,
    path: PathBuf,
    spec: ArtifactSpec,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

impl Fixture {
    fn new(marker: u8, offset: i32, peer_add: bool) -> Self {
        Self::with_identity_bytes(marker, marker, offset, peer_add)
    }

    fn with_identity_bytes(marker: u8, reported: u8, offset: i32, peer_add: bool) -> Self {
        let directory = std::env::temp_dir().join(format!(
            "invar-scalar-loader-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join(if cfg!(target_os = "macos") {
            "artifact.dylib"
        } else {
            "artifact.so"
        });
        let input_identity = [marker; 32];
        let begin = String::from_utf8(control_name(&input_identity, "export_begin_v1")).unwrap();
        let end = String::from_utf8(control_name(&input_identity, "export_end_v1")).unwrap();
        let identity =
            String::from_utf8(control_name(&input_identity, "export_identity_v1")).unwrap();
        let source = format!(
            r#"
#include "invar_foreign_v1.h"
typedef struct {{
uint32_t abi_version, kind, reason, span_start, span_end, reserved;
uint64_t fuel_limit, steps_used;
}} failure_record;
static _Thread_local int active;
static _Thread_local failure_record failure;
static const uint8_t identity_bytes[32] = {{ {identity_bytes} }};
uint32_t inv_abi_version_v1(void) {{ return 1; }}
const uint8_t *{identity}(void) {{ return identity_bytes; }}
int32_t {begin}(uint64_t fuel) {{
if (!fuel || active) return 1;
active = 1;
failure = (failure_record){{ .abi_version=1, .fuel_limit=fuel }};
return 0;
}}
int32_t {end}(failure_record *out) {{
if (!active || !out) return 1;
*out = failure;
active = 0;
return 0;
}}
static int32_t invoke(const inv_arg_v1 *args, uint32_t argc,
                  inv_result_v1 *out, uint32_t tag) {{
if (!active) return 1;
if (!args || argc != 1 || !out || args[0].tag != tag || args[0].mode
    || args[0].data || args[0].len) {{
    failure.kind=4; failure.reason=1; return 1;
}}
failure.steps_used=1;
int64_t scalar = tag==1 ? args[0].scalar+{offset}
                       : (int64_t)(uint32_t)({peer_expression});
*out=(inv_result_v1){{ .abi_version=1, .tag=tag, .scalar=scalar }};
return 0;
}}
int32_t inv_export_0(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out)
{{ return invoke(args, argc, out, 1); }}
int32_t inv_export_1(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out)
{{ return invoke(args, argc, out, 2); }}
"#,
            identity_bytes = [reported; 32]
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(","),
            begin = begin.trim_end_matches('\0'),
            end = end.trim_end_matches('\0'),
            identity = identity.trim_end_matches('\0'),
            peer_expression = if peer_add {
                "(uint32_t)args[0].scalar+29u"
            } else {
                "(uint32_t)args[0].scalar*3u"
            },
        );
        let c_path = directory.join("fixture.c");
        fs::write(&c_path, source).unwrap();
        let output = Command::new("cc")
            .arg(if cfg!(target_os = "macos") {
                "-dynamiclib"
            } else {
                "-shared"
            })
            .args(["-fPIC", "-std=c11", "-Wall", "-Wextra", "-Werror"])
            .arg("-I")
            .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../include"))
            .arg(&c_path)
            .arg("-o")
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let library_sha256 = file_digest(&path).unwrap();
        let mut digest = Sha256::new();
        digest.update(input_identity);
        digest.update(library_sha256);
        let identity = digest.finalize().into();
        let descriptors = [I32, U32]
            .into_iter()
            .enumerate()
            .map(|(index, tag)| Descriptor {
                artifact_id: identity,
                symbol: format!("inv_export_{index}"),
                parameter_tags: vec![tag],
                result_tag: tag,
            })
            .collect();
        Self {
            directory,
            path,
            spec: ArtifactSpec {
                identity,
                input_identity,
                library_sha256,
                descriptors,
            },
        }
    }

    fn load(&self) -> LoadedExports {
        LoadedExports::load(&self.path, &self.spec).unwrap()
    }
}

fn values(loader: &mut LoadedExports, fixture: &Fixture, signed: i32, unsigned: u32) {
    for (index, args, expected) in [
        (0, vec![Arg::I32(5)], Output::I32(signed)),
        (1, vec![Arg::U32(7)], Output::U32(unsigned)),
    ] {
        let result = loader
            .invoke(&fixture.spec.descriptors[index], &args, 100)
            .unwrap();
        assert_eq!(result.output, Some(expected));
        assert_eq!(result.failure.kind, 0);
        assert_eq!(result.failure.fuel_limit, 100);
    }
}

#[test]
fn issue_l1_06_export_lineage_loader_simultaneous_symbols_and_stale_descriptors() {
    let a = Fixture::new(0x11, 11, false);
    let b = Fixture::new(0x22, 23, true);
    let mut loaded_a = a.load();
    let mut loaded_b = b.load();
    for _ in 0..3 {
        values(&mut loaded_a, &a, 16, 21);
        values(&mut loaded_b, &b, 28, 36);
    }
    assert_eq!(
        loaded_b.invoke(&a.spec.descriptors[0], &[Arg::I32(5)], 100),
        Err(ExportLoadError::Identity)
    );
    assert_eq!(
        loaded_a.invoke(&b.spec.descriptors[0], &[Arg::I32(5)], 100),
        Err(ExportLoadError::Identity)
    );
    let retained = a.spec.descriptors[0].clone();
    drop(loaded_a);
    assert_eq!(
        loaded_b.invoke(&retained, &[Arg::I32(5)], 100),
        Err(ExportLoadError::Identity)
    );
    values(&mut loaded_b, &b, 28, 36);
}

#[test]
fn issue_l1_06_export_lineage_loader_retains_library_after_path_replacement_and_removal() {
    let a = Fixture::new(0x33, 11, false);
    let b = Fixture::new(0x44, 23, true);
    let mut loaded_a = a.load();
    let replacement = a.directory.join("replacement");
    fs::copy(&b.path, &replacement).unwrap();
    fs::rename(&replacement, &a.path).unwrap();
    values(&mut loaded_a, &a, 16, 21);
    // A stale specification cannot verify the replaced bytes.
    assert!(matches!(
        LoadedExports::load(&a.path, &a.spec),
        Err(ExportLoadError::Identity)
    ));
    // Some system loaders retain a path-keyed cache while A is alive. A new
    // load must either verify B's private identity or reject the cached A.
    match LoadedExports::load(&a.path, &b.spec) {
        Ok(mut loaded_b) => values(&mut loaded_b, &b, 28, 36),
        Err(ExportLoadError::Identity | ExportLoadError::Boundary(Error::Symbol)) => {}
        Err(error) => panic!("unexpected replacement load error: {error:?}"),
    }
    fs::remove_file(&a.path).unwrap();
    values(&mut loaded_a, &a, 16, 21);
    let mut loaded_b = b.load();
    values(&mut loaded_b, &b, 28, 36);
}

#[test]
fn issue_l1_06_export_lineage_loader_digest_private_identity_and_inventory_rejection() {
    // The correct private symbol name alone is insufficient: its immutable
    // identity bytes must match the compiler's prelink input digest too.
    let mismatched = Fixture::with_identity_bytes(0x77, 0x78, 11, false);
    assert!(matches!(
        LoadedExports::load(&mismatched.path, &mismatched.spec),
        Err(ExportLoadError::Identity)
    ));
    let fixture = Fixture::new(0x55, 11, false);
    let mut spec = fixture.spec.clone();
    spec.library_sha256[0] ^= 1;
    assert!(matches!(
        LoadedExports::load(&fixture.path, &spec),
        Err(ExportLoadError::Identity)
    ));
    spec = fixture.spec.clone();
    spec.input_identity[0] ^= 1;
    assert!(matches!(
        LoadedExports::load(&fixture.path, &spec),
        Err(ExportLoadError::Boundary(Error::Symbol))
    ));
    spec = fixture.spec.clone();
    spec.descriptors[0].artifact_id[0] ^= 1;
    assert!(matches!(
        LoadedExports::load(&fixture.path, &spec),
        Err(ExportLoadError::Identity)
    ));
    spec = fixture.spec.clone();
    spec.descriptors[1].symbol = spec.descriptors[0].symbol.clone();
    assert!(matches!(
        LoadedExports::load(&fixture.path, &spec),
        Err(ExportLoadError::Boundary(Error::Layout))
    ));
    spec = fixture.spec.clone();
    spec.descriptors[0].symbol = "invar_entry_v1".into();
    assert!(matches!(
        LoadedExports::load(&fixture.path, &spec),
        Err(ExportLoadError::Boundary(Error::Layout))
    ));
    let mut loader = fixture.load();
    values(&mut loader, &fixture, 16, 21);
}

#[test]
fn issue_l1_06_export_lineage_loader_full_descriptor_and_arguments_before_begin() {
    let fixture = Fixture::new(0x66, 11, false);
    let mut loader = fixture.load();
    let descriptor = &fixture.spec.descriptors[0];
    let mut changed = descriptor.clone();
    changed.result_tag = U32;
    assert_eq!(
        loader.invoke(&changed, &[Arg::I32(5)], 100),
        Err(ExportLoadError::Identity)
    );
    changed = descriptor.clone();
    changed.parameter_tags[0] = U32;
    assert_eq!(
        loader.invoke(&changed, &[Arg::U32(5)], 100),
        Err(ExportLoadError::Identity)
    );
    for args in [
        vec![],
        vec![Arg::U32(5)],
        vec![Arg::Bytes(b"input", super::super::Mode::Read)],
    ] {
        assert_eq!(
            loader.invoke(descriptor, &args, 100),
            Err(ExportLoadError::Boundary(Error::Layout))
        );
    }
    assert_eq!(
        loader.invoke(descriptor, &[Arg::I32(5)], 0),
        Err(ExportLoadError::Boundary(Error::Layout))
    );
    values(&mut loader, &fixture, 16, 21);
}

#[test]
fn issue_l1_06_export_symbol_inventory_validation() {
    let mut spec = ArtifactSpec {
        identity: [1; 32],
        input_identity: [2; 32],
        library_sha256: [3; 32],
        descriptors: Vec::new(),
    };
    for tags in [vec![I32], vec![0x111], vec![I32, 0x123]] {
        spec.descriptors = (0..2)
            .map(|ordinal| Descriptor {
                artifact_id: spec.identity,
                symbol: format!("inv_export_{ordinal}"),
                parameter_tags: tags.clone(),
                result_tag: I32,
            })
            .collect();
        assert_eq!(validate_spec(&spec), Ok(()));
        spec.descriptors.swap(0, 1);
        assert_eq!(
            validate_spec(&spec),
            Err(ExportLoadError::SymbolOrdinal {
                position: 0,
                ordinal: 1
            })
        );
        spec.descriptors.swap(0, 1);
        for symbol in [
            "inv_export_",
            "inv_export_01",
            "inv_export_+1",
            "inv_export_4294967296",
            "inv_export_1\0",
        ] {
            spec.descriptors[0].symbol = symbol.into();
            assert_eq!(
                validate_spec(&spec),
                Err(ExportLoadError::Boundary(Error::Layout))
            );
        }
        spec.descriptors[0].symbol = "inv_export_4294967295".into();
        assert_eq!(
            validate_spec(&spec),
            Err(ExportLoadError::SymbolOrdinal {
                position: 0,
                ordinal: u32::MAX,
            })
        );
    }
}

#[test]
fn issue_l1_06_native_arrays_dispatch_by_symbol_not_inventory_position() {
    unsafe extern "C" fn call(
        ordinal: u32,
        _: *const InvArg,
        _: u32,
        _: *const std::ffi::c_void,
        _: u32,
        result: *mut InvResult,
    ) -> i32 {
        // SAFETY: LoadedExports supplies live result storage for this call.
        unsafe {
            *result = InvResult {
                abi_version: ABI_VERSION,
                tag: I32,
                scalar: 100 + i64::from(ordinal),
                data: std::ptr::null_mut(),
                len: 0,
                release: None,
            };
        }
        0
    }

    let fixture = Fixture::new(0x88, 0, false);
    let mut loaded = fixture.load();
    // Fault seam: loading rejects reordered inventories. Permute the retained
    // private inventory here to isolate dispatch from that separate guard.
    for descriptor in &mut loaded.spec.descriptors {
        descriptor.parameter_tags = vec![0x111];
        descriptor.result_tag = I32;
    }
    loaded.spec.descriptors.swap(0, 1);
    loaded.call_arrays = Some(call);
    #[repr(C)]
    struct Record {
        element: u32,
        mode: u32,
        data: *const i32,
        len: u64,
    }
    let input = [0i32];
    let record = Record {
        element: 1,
        mode: 1,
        data: input.as_ptr(),
        len: 1,
    };
    for descriptor in loaded.spec.descriptors.clone() {
        // SAFETY: this one read-i32 record has the frozen private layout and
        // descriptor tag; its backing input stays live throughout the call.
        let output = unsafe {
            loaded.invoke_arrays(&descriptor, &[], std::ptr::from_ref(&record).cast(), 1, 100)
        }
        .unwrap()
        .output;
        let expected = if descriptor.symbol == "inv_export_1" {
            101
        } else {
            100
        };
        assert_eq!(output, Some(Output::I32(expected)));
    }
}
