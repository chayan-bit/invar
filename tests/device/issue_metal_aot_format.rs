//! Compiled-ahead Metal container: every field a production load relies on is
//! validated as untrusted input, and a container is matched to its program
//! with typed refusals. Runs everywhere; no GPU or Metal Toolchain needed.
#![forbid(unsafe_code)]

use invar::{
    admission, check, front,
    metal_aot::{self, Error, Header, Library, Route, Toolchain},
    mir,
    plan::Execution,
};
use sha2::{Digest, Sha256};

const SOURCE: &str = "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; \
    var ys: Array<u32> = filled_u32(len(xs),0u32); \
    for i in 0u32 .. len(xs) { ys[i] = wrapping_add(xs[i],5u32); } \
    if ys[2u32] == 8u32 { return 0; } return 1; }";
const MSL_A: &str = "kernel void a_v1(device uint* o [[buffer(0)]]) { o[0] = 1u; }";
const MSL_B: &str = "kernel void b_v1(device uint* o [[buffer(0)]]) { o[0] = 2u; }";

#[test]
fn gnet_inventory_requires_the_exact_offline_d1_probe() {
    use invar::general_net::{
        compiled::Target,
        device_host::{Limits, Plan},
    };
    for (source, count) in [
        ("fn run(x:i32)->i32{return x+7;}", 1),
        ("fn run(x:f32)->f32{return x+0.5;}", 2),
    ] {
        let plan =
            Plan::from_source(source.as_bytes(), "run", Target::Metal, Limits::default()).unwrap();
        let kernels = metal_aot::gnet_kernels(plan.artifact());
        assert_eq!(kernels.len(), count);
        let (mut header, _) = header_and_images();
        header.libraries = kernels
            .iter()
            .map(|(msl, functions)| {
                let mut l = library(msl, &functions[0], &metallib(64));
                l.functions = functions.clone();
                l
            })
            .collect();
        header.verify_gnet(plan.artifact()).unwrap();
        if count == 2 {
            let mut missing = header.clone();
            missing.libraries.pop();
            assert_eq!(
                missing.verify_gnet(plan.artifact()),
                Err(Error::Program("libraries"))
            );
            let mut stale = header.clone();
            stale.libraries[1].msl_sha256[0] ^= 1;
            assert_eq!(
                stale.verify_gnet(plan.artifact()),
                Err(Error::Program("msl_sha256"))
            );
            header.libraries[1].functions[0].push('_');
            assert!(matches!(
                header.verify_gnet(plan.artifact()),
                Err(Error::EntryPoint(_))
            ));
        }
    }
}

/// A minimal macOS metallib header: magic, macOS platform byte, deployment
/// target 14.0 and the file-size field, as Apple's `metallib` writes them.
fn metallib(len: usize) -> Vec<u8> {
    let mut b = vec![0u8; len];
    b[..4].copy_from_slice(b"MTLB");
    b[4..12].copy_from_slice(&[0x01, 0x80, 0x02, 0x00, 0x07, 0x00, 0x00, 0x81]);
    b[12..14].copy_from_slice(&14u16.to_le_bytes());
    b[16..24].copy_from_slice(&(len as u64).to_le_bytes());
    b
}

fn sha(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn library(msl: &str, function: &str, image: &[u8]) -> Library {
    Library {
        msl_sha256: sha(msl.as_bytes()),
        functions: vec![function.into()],
        bytes: image.len(),
        sha256: sha(image),
    }
}

/// Two libraries, sorted by MSL digest as the build writes them.
fn header_and_images() -> (Header, Vec<Vec<u8>>) {
    let mut items = vec![
        (library(MSL_A, "a_v1", &metallib(64)), metallib(64)),
        (library(MSL_B, "b_v1", &metallib(96)), metallib(96)),
    ];
    items.sort_by_key(|a| a.0.msl_sha256);
    let (libraries, images) = items.into_iter().unzip();
    let header = Header {
        route: Route::Run,
        entry: "main".into(),
        source_sha256: sha(SOURCE.as_bytes()),
        toolchain: Toolchain {
            metal: "Apple metal version 32023.921 (metalfe-32023.921.6)".into(),
            metallib: "AIR-LLD 32023.921 (metalfe-32023.921.6)".into(),
            sdk: "27.0 (26A425)".into(),
        },
        invar: "0.1.0".into(),
        libraries,
    };
    (header, images)
}

fn container() -> Vec<u8> {
    let (header, images) = header_and_images();
    metal_aot::encode(&header, &images).unwrap()
}

/// Re-seal a container around arbitrary header text and image bytes, so a
/// test reaches the checks after the container digest.
fn seal(header: &[u8], image: &[u8]) -> Vec<u8> {
    let mut out = b"INVGMETL".to_vec();
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    out.extend_from_slice(&(image.len() as u64).to_le_bytes());
    let mut hash = Sha256::new();
    hash.update(&out);
    hash.update(header);
    hash.update(image);
    out.extend_from_slice(&hash.finalize());
    out.extend_from_slice(header);
    out.extend_from_slice(image);
    out
}

fn parts(bytes: &[u8]) -> (String, Vec<u8>) {
    let h = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    (
        String::from_utf8(bytes[56..56 + h].to_vec()).unwrap(),
        bytes[56 + h..].to_vec(),
    )
}

fn with_header(edit: impl Fn(&mut serde_json::Value)) -> Result<metal_aot::Image, Error> {
    let (text, image) = parts(&container());
    let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
    edit(&mut value);
    metal_aot::decode(&seal(&serde_json::to_vec(&value).unwrap(), &image))
}

#[test]
fn round_trip_records_program_target_and_toolchain() {
    let image = metal_aot::decode(&container()).unwrap();
    image
        .header()
        .verify(Route::Run, "main", sha(SOURCE.as_bytes()))
        .unwrap();
    let receipt = image.receipt();
    assert_eq!(receipt["schema"], metal_aot::SCHEMA);
    assert_eq!(receipt["format_version"], 1);
    assert_eq!(receipt["route"], "run-device-ir-v1");
    assert_eq!(receipt["target"]["language"], "metal3.1");
    assert_eq!(receipt["target"]["min_os"], "14.0");
    assert_eq!(
        receipt["toolchain"]["options"],
        serde_json::json!([
            "-std=metal3.1",
            "-fno-fast-math",
            "-mmacosx-version-min=14.0"
        ])
    );
    assert_eq!(receipt["toolchain"]["sdk"], "27.0 (26A425)");
    assert_eq!(receipt["runtime_compiler"], "none");
    assert_eq!(receipt["loaded_via"], "newLibraryWithData");
    assert_eq!(receipt["libraries"].as_array().unwrap().len(), 2);
    let sizes = image
        .libraries()
        .map(|(l, b)| (l.bytes, b.len()))
        .collect::<Vec<_>>();
    assert!(sizes.iter().all(|(a, b)| a == b), "{sizes:?}");
    assert_eq!(
        image.header().library(MSL_A).map(|l| l.functions.clone()),
        Some(vec!["a_v1".to_owned()])
    );
}

#[test]
fn container_framing_is_validated_before_parsing() {
    let good = container();
    let mut bad = good.clone();
    bad[0] ^= 1;
    assert_eq!(metal_aot::decode(&bad), Err(Error::Magic));
    let mut bad = good.clone();
    bad[8] = 2;
    assert_eq!(metal_aot::decode(&bad), Err(Error::Version(2)));
    let mut bad = good.clone();
    bad[12..16].copy_from_slice(&(metal_aot::MAX_HEADER_BYTES as u32 + 1).to_le_bytes());
    assert_eq!(metal_aot::decode(&bad), Err(Error::Size));
    let mut bad = good.clone();
    bad[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(metal_aot::decode(&bad), Err(Error::Size));
    let mut bad = good.clone();
    bad[16..24].copy_from_slice(&0u64.to_le_bytes());
    assert_eq!(metal_aot::decode(&bad), Err(Error::Size));
    assert_eq!(metal_aot::decode(&good[..40]), Err(Error::Truncated));
    assert_eq!(
        metal_aot::decode(&good[..good.len() - 1]),
        Err(Error::Truncated)
    );
    let mut long = good.clone();
    long.push(0);
    assert_eq!(metal_aot::decode(&long), Err(Error::Size));
    // Any flipped byte in the digest, header or library image is corrupt.
    for at in [30, 60, good.len() - 1] {
        let mut bad = good.clone();
        bad[at] ^= 0x20;
        assert_eq!(metal_aot::decode(&bad), Err(Error::Digest), "byte {at}");
    }
}

#[test]
fn header_must_be_canonical_complete_and_pinned() {
    let (text, image) = parts(&container());
    let spaced = text.replacen(':', ": ", 1);
    assert_eq!(
        metal_aot::decode(&seal(spaced.as_bytes(), &image)),
        Err(Error::Header("canonical"))
    );
    assert_eq!(
        with_header(|v| v["extra"] = 1.into()),
        Err(Error::Header("keys"))
    );
    assert_eq!(
        with_header(|v| v["schema"] = "invar-metal-library-v0".into()),
        Err(Error::Header("schema"))
    );
    assert_eq!(
        with_header(|v| v["route"] = "gnet-abi4".into()),
        Err(Error::Header("route"))
    );
    // Toolchain mismatch: any option, target or compiler other than the
    // pinned ones is refused, so fast math can never ride in.
    assert_eq!(
        with_header(|v| v["toolchain"]["options"][1] = "-ffast-math".into()),
        Err(Error::Toolchain("options"))
    );
    assert_eq!(
        with_header(|v| v["target"]["language"] = "metal3.2".into()),
        Err(Error::Toolchain("target"))
    );
    assert_eq!(
        with_header(|v| v["target"]["min_os"] = "15.0".into()),
        Err(Error::Toolchain("target"))
    );
    assert_eq!(
        with_header(|v| v["toolchain"]["compiler"] = "newLibraryWithSource".into()),
        Err(Error::Toolchain("compiler"))
    );
    assert_eq!(
        with_header(|v| v["program"]["source_sha256"] = "ABC".into()),
        Err(Error::Header("digest"))
    );
    assert_eq!(
        with_header(|v| v["libraries"] = serde_json::json!([])),
        Err(Error::Header("libraries"))
    );
    assert_eq!(
        with_header(|v| v["libraries"].as_array_mut().unwrap().reverse()),
        Err(Error::Header("library order"))
    );
    assert_eq!(
        with_header(|v| v["libraries"][0]["functions"][0] = "a(); evil".into()),
        Err(Error::Header("functions"))
    );
    assert_eq!(
        with_header(|v| v["libraries"][0]["bytes"] = (metal_aot::MAX_LIBRARY_BYTES + 1).into()),
        Err(Error::Header("bytes"))
    );
    // A header whose library sizes or digests disagree with the image.
    assert_eq!(
        with_header(|v| v["libraries"][0]["sha256"] = "0".repeat(64).into()),
        Err(Error::Digest)
    );
    let bytes = |v: &serde_json::Value| v["libraries"][0]["bytes"].as_u64().unwrap();
    assert_eq!(
        with_header(|v| v["libraries"][0]["bytes"] = (bytes(v) - 8).into()),
        Err(Error::Digest)
    );
}

#[test]
fn metallib_header_fields_are_checked() {
    assert_eq!(metal_aot::metallib(&metallib(64)), Ok(()));
    let check = |edit: &dyn Fn(&mut Vec<u8>)| {
        let mut b = metallib(64);
        edit(&mut b);
        metal_aot::metallib(&b)
    };
    assert_eq!(check(&|b| b[0] = b'X'), Err(Error::Metallib("magic")));
    assert_eq!(check(&|b| b[11] = 0x82), Err(Error::Metallib("platform")));
    assert_eq!(
        check(&|b| b[12] = 15),
        Err(Error::Metallib("deployment target"))
    );
    assert_eq!(check(&|b| b[16] = 63), Err(Error::Metallib("file size")));
    assert_eq!(
        metal_aot::metallib(b"MTLB"),
        Err(Error::Metallib("platform"))
    );
    // A library that is not a metallib never reaches Metal.
    let (header, _) = header_and_images();
    let junk = vec![vec![0u8; 64], vec![0u8; 96]];
    let mut header = header;
    for (library, bytes) in header.libraries.iter_mut().zip(&junk) {
        library.sha256 = sha(bytes);
        library.bytes = bytes.len();
    }
    assert_eq!(
        metal_aot::decode(&metal_aot::encode(&header, &junk).unwrap()),
        Err(Error::Metallib("magic"))
    );
}

#[test]
fn container_is_matched_to_its_program_route_and_entry() {
    let image = metal_aot::decode(&container()).unwrap();
    let header = image.header();
    let stale = SOURCE.replace("5u32", "6u32");
    assert_eq!(
        header.verify(Route::Run, "main", sha(stale.as_bytes())),
        Err(Error::Program("source_sha256"))
    );
    assert_eq!(
        header.verify(Route::Gnet, "main", sha(SOURCE.as_bytes())),
        Err(Error::Program("route"))
    );
    assert_eq!(
        header.verify(Route::Run, "entry", sha(SOURCE.as_bytes())),
        Err(Error::Program("entry"))
    );
    // Kernels are found by their exact MSL; one changed byte misses.
    assert!(header.library(MSL_A).is_some());
    assert!(header.library(&MSL_A.replace("1u", "3u")).is_none());
}

#[test]
fn entry_points_are_read_from_generated_msl() {
    let msl = "kernel void b_v1(device uint* o) {}\nkernel void a_v1 (device uint* o) {}\nvoid helper() {}";
    assert_eq!(metal_aot::msl_entry_points(msl), ["a_v1", "b_v1"]);
}

fn regions(source: &str) -> (Vec<u8>, mir::Program) {
    let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
    (source.as_bytes().to_vec(), mir::lower(&checked).unwrap())
}

/// One build serves every run: generated MSL is the same at every length and
/// on every Apple GPU family, for map, DAG, reduction and scan regions.
#[test]
fn generated_msl_does_not_depend_on_length_or_family() {
    let programs = [
        SOURCE,
        "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; let ys: Array<u32> = [4u32,5u32,6u32]; \
         let zs: Array<u32> = [7u32,8u32,9u32]; var a: Array<u32> = filled_u32(len(xs),0u32); \
         for i in 0u32 .. len(xs) { a[i] = wrapping_add(xs[i],ys[i]); } let first: Array<u32> = move a; \
         var b: Array<u32> = filled_u32(len(first),0u32); \
         for j in 0u32 .. len(first) { b[j] = wrapping_mul(first[j],zs[j]); } return 0; }",
        "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; let zs: Array<u32> = [7u32,8u32,4294967295u32]; \
         var ys: Array<u32> = filled_u32(len(xs),0u32); \
         for i in 0u32 .. len(xs) { ys[i] = wrapping_mul(zs[i],xs[i]); } \
         if ys[2u32] != 4294967293u32 { return 1; } return 0; }",
        "fn main() -> i32 { let xs: Array<u32> = [1u32,2u32,3u32]; var acc: u32 = 7u32; for i in 0u32 .. len(xs) { acc = wrapping_add(acc,xs[i]); } if acc == 13u32 { return 0; } return 1; }",
        "fn main() -> i32 { let aa: Array<u32> = [2u32,3u32,5u32]; let bb: Array<u32> = [7u32,11u32,13u32]; var state: u32 = 17u32; var out: Array<u32> = filled_u32(len(aa),0u32); for i in 0u32 .. len(aa) { state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); out[i] = state; } if out[2u32] == 683u32 { return 0; } return 1; }",
    ];
    let mut kinds = 0;
    for program in programs {
        let (source, lowered) = regions(program);
        let report = admission::analyze(&source, &lowered).unwrap();
        let built = metal_aot::run_kernels(&source, &lowered).unwrap();
        for region in report.regions() {
            if matches!(
                region.execution(),
                Execution::FusedConsumer { .. } | Execution::DagConsumer { .. }
            ) {
                continue;
            }
            let Some((reference, functions)) = metal_aot::region_msl(
                region,
                4096,
                invar_device::Capabilities::Metal {
                    apple_family: 9,
                    msl_major: 3,
                    msl_minor: 1,
                },
            ) else {
                continue;
            };
            kinds += 1;
            assert!(
                built
                    .iter()
                    .any(|(msl, f)| *msl == reference && f == functions),
                "the build holds the region's kernel"
            );
            for length in [1u32, 3, 255, 256, 257, 4097, 65_536, 999_999, 1_000_000] {
                for family in 7..=10 {
                    let caps = invar_device::Capabilities::Metal {
                        apple_family: family,
                        msl_major: 3,
                        msl_minor: 1,
                    };
                    let (msl, _) = metal_aot::region_msl(region, length, caps).unwrap();
                    assert_eq!(msl, reference, "length {length} family {family}");
                }
            }
        }
        assert_eq!(built[0].0, invar_device::PROBE_MSL, "the probe ships first");
    }
    assert_eq!(kinds, 5);
}

#[test]
fn files_are_written_atomically_privately_and_read_bounded() {
    let dir = std::env::temp_dir().join(format!("invar-metal-aot-format-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("program.metallibc");
    metal_aot::build::write_atomic(&path, &container()).unwrap();
    metal_aot::build::write_atomic(&path, &container()).unwrap();
    let names = std::fs::read_dir(&dir).unwrap().count();
    assert_eq!(names, 1, "no temporary file is left behind");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }
    assert!(metal_aot::build::read(&path).is_ok());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o620)).unwrap();
        assert_eq!(metal_aot::build::read(&path), Err(Error::Permissions));
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        let link = dir.join("link.metallibc");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert_eq!(metal_aot::build::read(&link), Err(Error::Permissions));
        std::fs::remove_file(&link).unwrap();
    }
    let huge = dir.join("huge.metallibc");
    let file = std::fs::File::create(&huge).unwrap();
    file.set_len(metal_aot::MAX_CONTAINER_BYTES as u64 + 1)
        .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&huge, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    assert_eq!(metal_aot::build::read(&huge), Err(Error::Size));
    std::fs::remove_dir_all(&dir).unwrap();
}

fn invar(args: &[&str], path: Option<&std::path::Path>) -> (bool, serde_json::Value) {
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_invar"));
    if let Some(path) = path {
        command.env("PATH", path);
    }
    let output = command.args(args).output().unwrap();
    let text = String::from_utf8_lossy(&output.stdout);
    (
        output.status.success(),
        serde_json::from_str(text.trim()).unwrap_or(serde_json::Value::Null),
    )
}

/// Without Apple's Metal Toolchain the build refuses with a typed error that
/// names the component and how to install it; nothing falls back.
#[test]
fn build_refuses_when_the_metal_toolchain_is_missing() {
    let dir = std::env::temp_dir().join(format!("invar-metal-aot-tool-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("map.inv");
    std::fs::write(&source, SOURCE).unwrap();
    let output = dir.join("map.metallibc");
    let args = [
        "build",
        source.to_str().unwrap(),
        "--target",
        "metal",
        "--output",
        output.to_str().unwrap(),
    ];
    // A shim `xcrun` that answers as Xcode does when the component is absent.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let bin = dir.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        let shim = bin.join("xcrun");
        std::fs::write(
            &shim,
            "#!/bin/sh\necho \"error: error: cannot execute tool 'metal' due to missing Metal Toolchain; use: xcodebuild -downloadComponent MetalToolchain\" >&2\nexit 1\n",
        )
        .unwrap();
        std::fs::set_permissions(&shim, std::fs::Permissions::from_mode(0o700)).unwrap();
        let (ok, report) = invar(&args, Some(&bin));
        assert!(!ok);
        let error = report["error"].as_str().unwrap();
        assert!(error.starts_with("MissingToolchain("), "{error}");
        assert!(error.contains("Metal Toolchain"), "{error}");
        assert!(
            error.contains("xcodebuild -downloadComponent MetalToolchain"),
            "{error}"
        );
    }
    // No `xcrun` at all.
    let empty = dir.join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let (ok, report) = invar(&args, Some(&empty));
    assert!(!ok);
    assert!(
        report["error"]
            .as_str()
            .unwrap()
            .starts_with("MissingToolchain("),
        "{report}"
    );
    assert!(!output.exists(), "a refused build publishes nothing");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn artifacts_apply_to_metal_and_auto_only() {
    let dir = std::env::temp_dir().join(format!("invar-metal-aot-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("map.inv");
    std::fs::write(&source, SOURCE).unwrap();
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "run",
            source.to_str().unwrap(),
            "--engine",
            "reference",
            "--artifact",
            "x",
        ])
        .output()
        .unwrap()
        .status;
    assert_eq!(status.code(), Some(2), "usage error");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Swapping a symlink to a valid container in for a junk regular file, while
/// reads run, must never get the container loaded: the check and the open
/// are one atomic step. Bounded at five seconds of attempts.
#[cfg(unix)]
#[test]
fn a_symlink_swapped_in_during_a_read_is_never_followed() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};
    let dir = std::env::temp_dir().join(format!("invar-metal-aot-race-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let target = dir.join("valid.metallibc");
    metal_aot::build::write_atomic(&target, &container()).unwrap();
    let junk = dir.join("junk");
    metal_aot::build::write_atomic(&junk, b"not a container").unwrap();
    let swap = dir.join("swap.metallibc");
    std::fs::hard_link(&junk, &swap).unwrap();
    let stop = AtomicBool::new(false);
    let (attempts, loaded) = std::thread::scope(|scope| {
        scope.spawn(|| {
            let (link, regular) = (dir.join("link.tmp"), dir.join("regular.tmp"));
            while !stop.load(Ordering::Relaxed) {
                std::os::unix::fs::symlink(&target, &link).unwrap();
                std::fs::rename(&link, &swap).unwrap();
                std::fs::hard_link(&junk, &regular).unwrap();
                std::fs::rename(&regular, &swap).unwrap();
            }
        });
        let begun = Instant::now();
        let mut attempts = 0u64;
        let mut loaded = false;
        while begun.elapsed() < Duration::from_secs(5) && !loaded {
            loaded = metal_aot::build::read(&swap).is_ok();
            attempts += 1;
        }
        stop.store(true, Ordering::Relaxed);
        (attempts, loaded)
    });
    assert!(
        !loaded,
        "a container reachable only through a symlink loaded after {attempts} attempts"
    );
    eprintln!("{attempts} reads, none followed the swapped symlink");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A FIFO (or any non-regular file) is refused at once, never read: a FIFO
/// without a writer would otherwise block the open forever.
#[cfg(unix)]
#[test]
fn a_fifo_is_refused_without_blocking() {
    let dir = std::env::temp_dir().join(format!("invar-metal-aot-fifo-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let fifo = dir.join("pipe.metallibc");
    let made = std::process::Command::new("mkfifo")
        .args(["-m", "600"])
        .arg(&fifo)
        .status()
        .unwrap();
    assert!(made.success());
    let (send, receive) = std::sync::mpsc::channel();
    let path = fifo.clone();
    // Detached: if the read blocks, the test fails on the timeout below.
    std::thread::spawn(move || send.send(metal_aot::build::read(&path)).unwrap());
    let result = receive
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("reading a FIFO must not block");
    assert_eq!(result, Err(Error::Permissions));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A build without Metal cannot honor `--artifact`: run and explain refuse
/// with a typed error instead of returning a CPU result.
#[cfg(not(all(feature = "metal", target_os = "macos")))]
#[test]
fn a_build_without_metal_refuses_an_artifact() {
    let dir = std::env::temp_dir().join(format!("invar-metal-aot-nometal-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("map.inv");
    std::fs::write(&source, SOURCE).unwrap();
    let source = source.to_str().unwrap();
    for engine in ["auto", "metal"] {
        let (ok, report) = invar(
            &[
                "run",
                source,
                "--engine",
                engine,
                "--artifact",
                "does-not-exist",
                "--format",
                "json",
            ],
            None,
        );
        assert!(!ok, "{engine}: {report}");
        assert_eq!(
            report["diagnostics"][0]["code"], "METAL_UNAVAILABLE",
            "{engine}: {report}"
        );
        assert!(report["result"].is_null(), "{engine}: {report}");
        let (ok, report) = invar(
            &[
                "explain",
                source,
                "--engine",
                engine,
                "--artifact",
                "does-not-exist",
                "--format",
                "json",
            ],
            None,
        );
        assert!(!ok, "{engine}: {report}");
        assert_eq!(
            report["diagnostics"][0]["code"], "METAL_UNAVAILABLE",
            "{engine}: {report}"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
