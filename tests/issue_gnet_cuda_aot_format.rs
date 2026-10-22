//! Compiled-ahead CUDA container: every field a production load relies on is
//! validated as untrusted input, and an image is matched to its program and
//! device with typed refusals. Runs everywhere; no device or NVRTC needed.
#![forbid(unsafe_code)]

use invar::general_net::{
    compiled::Target,
    device_host::{
        Limits, Plan,
        cuda_aot::{self, Error, Header},
    },
};
use sha2::{Digest, Sha256};

const SOURCE: &str = "fn twice(x: i32) -> i32 { return x + x; } fn run(a: i32, b: i32) -> i32 { let p = twice(a); let q = twice(b); return p + q; }";
const OTHER: &str = "fn twice(x: i32) -> i32 { return x * 2; } fn run(a: i32, b: i32) -> i32 { let p = twice(a); let q = twice(b); return p + q; }";

fn plan(source: &str) -> Plan {
    Plan::from_source(source.as_bytes(), "run", Target::Cuda, Limits::default()).unwrap()
}

/// A minimal CUDA ELF64 executable: header, null section, a one-byte string
/// table and a one-byte executable section, laid out like NVRTC's ABI 7
/// (sm_89) or ABI 8 (sm_120).
fn elf(sm: u32) -> Vec<u8> {
    let (osabi, abi, flags) = if sm >= 100 {
        (0x41, 8, 0x0600_0002 | (sm << 8))
    } else {
        (0x33, 7, (sm << 16) | 0x0500 | sm)
    };
    let mut b = vec![0u8; 64 + 3 * 64 + 2];
    b[..9].copy_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1, 1, osabi, abi]);
    b[16..18].copy_from_slice(&2u16.to_le_bytes());
    b[18..20].copy_from_slice(&190u16.to_le_bytes());
    b[20..24].copy_from_slice(&1u32.to_le_bytes());
    b[40..48].copy_from_slice(&64u64.to_le_bytes());
    b[48..52].copy_from_slice(&flags.to_le_bytes());
    b[52..54].copy_from_slice(&64u16.to_le_bytes());
    b[54..56].copy_from_slice(&56u16.to_le_bytes());
    b[58..60].copy_from_slice(&64u16.to_le_bytes());
    b[60..62].copy_from_slice(&3u16.to_le_bytes());
    b[62..64].copy_from_slice(&1u16.to_le_bytes());
    let strtab = 128;
    b[strtab + 4..strtab + 8].copy_from_slice(&3u32.to_le_bytes());
    b[strtab + 24..strtab + 32].copy_from_slice(&256u64.to_le_bytes());
    b[strtab + 32..strtab + 40].copy_from_slice(&1u64.to_le_bytes());
    let text = 192;
    b[text + 4..text + 8].copy_from_slice(&1u32.to_le_bytes());
    b[text + 8..text + 16].copy_from_slice(&6u64.to_le_bytes());
    b[text + 24..text + 32].copy_from_slice(&257u64.to_le_bytes());
    b[text + 32..text + 40].copy_from_slice(&1u64.to_le_bytes());
    b
}

fn container(plan: &Plan, sm: u32, cubin: &[u8]) -> Vec<u8> {
    cuda_aot::encode(&Header::of(plan, sm, (12, 9), cubin), cubin).unwrap()
}

/// Re-seal a container around arbitrary header text and image bytes, so a
/// test reaches the check after the digest.
fn seal(header: &[u8], image: &[u8]) -> Vec<u8> {
    let mut out = b"INVGCUDA".to_vec();
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

fn parts(bytes: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let h = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    (bytes[56..56 + h].to_vec(), bytes[56 + h..].to_vec())
}

/// Re-seal with a header whose image fields describe the new image.
fn reseal_image(plan: &Plan, sm: u32, image: &[u8]) -> Vec<u8> {
    let header = Header::of(plan, sm, (12, 9), image);
    let (text, _) = parts(&cuda_aot::encode(&header, image).unwrap());
    seal(&text, image)
}

#[test]
fn round_trip_records_program_target_and_toolchain() {
    let plan = plan(SOURCE);
    for sm in [89, 120] {
        let image = cuda_aot::decode(&container(&plan, sm, &elf(sm))).unwrap();
        assert_eq!(cuda_aot::cubin_sm(image.cubin()), Ok(sm));
        image.header().verify(&plan).unwrap();
        image.header().admit_device(sm, 12090).unwrap();
        let receipt = image.receipt();
        assert_eq!(receipt["schema"], cuda_aot::SCHEMA);
        assert_eq!(receipt["format_version"], 1);
        assert_eq!(receipt["target"]["sm"], sm);
        assert_eq!(receipt["target"]["min_driver"], 12090);
        assert_eq!(receipt["toolchain"]["compiler"], "nvrtc");
        assert_eq!(receipt["toolchain"]["nvrtc"], "12.9");
        assert_eq!(
            receipt["toolchain"]["options"],
            serde_json::json!([
                format!("--gpu-architecture=sm_{sm}"),
                "--std=c++17",
                "--fmad=false",
                "--ftz=false",
                "--prec-div=true",
                "--prec-sqrt=true",
            ])
        );
        assert_eq!(receipt["kernel"], cuda_aot::KERNEL);
        assert_eq!(receipt["abi"], 5);
        assert_eq!(receipt["entry"], "run");
        assert_eq!(receipt["runtime_compiler"], "none");
        assert_eq!(receipt["loaded_via"], "cuModuleLoadData");
        assert_eq!(receipt["image"]["bytes"], elf(sm).len());
        assert_eq!(
            receipt["program"]["source_sha256"].as_str().unwrap().len(),
            64
        );
    }
}

/// The header records the numeric flags the build-time compiler really
/// passes, not only the architecture and language standard.
#[cfg(feature = "cuda")]
#[test]
fn recorded_options_are_the_ones_the_gnet_compiler_passes() {
    let recorded = cuda_aot::options(120);
    assert_eq!(&recorded[..2], ["--gpu-architecture=sm_120", "--std=c++17"]);
    assert_eq!(
        recorded[2..],
        invar_cuda::cuda::GNET_STRICT_NVRTC_OPTIONS.map(String::from)
    );
}

#[test]
fn container_framing_is_validated_before_parsing() {
    let plan = plan(SOURCE);
    let good = container(&plan, 120, &elf(120));
    let mut bad = good.clone();
    bad[0] ^= 1;
    assert_eq!(cuda_aot::decode(&bad), Err(Error::Magic));
    let mut bad = good.clone();
    bad[8] = 2;
    assert_eq!(cuda_aot::decode(&bad), Err(Error::Version(2)));
    let mut bad = good.clone();
    bad[12..16].copy_from_slice(&(cuda_aot::MAX_HEADER_BYTES as u32 + 1).to_le_bytes());
    assert_eq!(cuda_aot::decode(&bad), Err(Error::Size));
    let mut bad = good.clone();
    bad[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(cuda_aot::decode(&bad), Err(Error::Size));
    let mut bad = good.clone();
    bad[16..24].copy_from_slice(&0u64.to_le_bytes());
    assert_eq!(cuda_aot::decode(&bad), Err(Error::Size));
    assert_eq!(cuda_aot::decode(&good[..40]), Err(Error::Truncated));
    assert_eq!(
        cuda_aot::decode(&good[..good.len() - 1]),
        Err(Error::Truncated)
    );
    let mut long = good.clone();
    long.push(0);
    assert_eq!(cuda_aot::decode(&long), Err(Error::Size));
    for at in [30, 60, good.len() - 1] {
        let mut bad = good.clone();
        bad[at] ^= 0x20;
        assert_eq!(cuda_aot::decode(&bad), Err(Error::Digest), "byte {at}");
    }
}

#[test]
fn header_must_be_canonical_complete_and_consistent() {
    let plan = plan(SOURCE);
    let (text, image) = parts(&container(&plan, 120, &elf(120)));
    let value: serde_json::Value = serde_json::from_slice(&text).unwrap();
    let pretty = serde_json::to_vec_pretty(&value).unwrap();
    assert_eq!(
        cuda_aot::decode(&seal(&pretty, &image)),
        Err(Error::Header("canonical"))
    );
    let edit = |f: &dyn Fn(&mut serde_json::Value)| {
        let mut v = value.clone();
        f(&mut v);
        cuda_aot::decode(&seal(&serde_json::to_vec(&v).unwrap(), &image))
    };
    assert_eq!(edit(&|v| v["extra"] = 1.into()), Err(Error::Header("keys")));
    assert_eq!(
        edit(&|v| {
            v["toolchain"]["options"] =
                serde_json::json!(["--gpu-architecture=sm_120", "--std=c++17"]);
        }),
        Err(Error::Header("toolchain"))
    );
    assert_eq!(
        edit(&|v| {
            v["toolchain"].as_object_mut().unwrap().remove("invar");
        }),
        Err(Error::Header("keys"))
    );
    assert_eq!(
        edit(&|v| v["schema"] = "invar-gnet-cuda-image-v0".into()),
        Err(Error::Header("schema"))
    );
    assert_eq!(
        edit(&|v| v["toolchain"]["options"][1] = "--use_fast_math".into()),
        Err(Error::Header("toolchain"))
    );
    assert_eq!(
        edit(&|v| v["target"]["min_driver"] = 1.into()),
        Err(Error::Header("toolchain"))
    );
    assert_eq!(
        edit(&|v| v["toolchain"]["nvrtc"] = "12".into()),
        Err(Error::Header("nvrtc"))
    );
    assert_eq!(
        edit(&|v| v["program"]["source_sha256"] = "AB".repeat(32).into()),
        Err(Error::Header("digest"))
    );
    assert_eq!(
        edit(&|v| v["image"]["bytes"] = 1.into()),
        Err(Error::Digest)
    );
    assert_eq!(
        edit(&|v| v["image"]["sha256"] = "00".repeat(32).into()),
        Err(Error::Digest)
    );
    assert_eq!(
        edit(&|v| v["target"]["sm"] = 7.into()),
        Err(Error::Header("sm"))
    );
}

#[test]
fn cubin_header_fields_and_tables_are_bounded() {
    let plan = plan(SOURCE);
    let tamper = |f: &dyn Fn(&mut Vec<u8>)| {
        let mut image = elf(120);
        f(&mut image);
        cuda_aot::decode(&reseal_image(&plan, 120, &image))
    };
    assert!(tamper(&|_| {}).is_ok());
    assert_eq!(
        tamper(&|b| b[1] = b'X'),
        Err(Error::Elf("not a little-endian ELF64"))
    );
    assert_eq!(
        tamper(&|b| b[4] = 1),
        Err(Error::Elf("not a little-endian ELF64"))
    );
    assert_eq!(
        tamper(&|b| b[18] = 62),
        Err(Error::Elf("not a CUDA executable"))
    );
    assert_eq!(
        tamper(&|b| b[16] = 1),
        Err(Error::Elf("not a CUDA executable"))
    );
    assert_eq!(
        tamper(&|b| b[8] = 9),
        Err(Error::Elf("unknown CUDA ELF ABI"))
    );
    assert_eq!(
        tamper(&|b| b[40..48].copy_from_slice(&4096u64.to_le_bytes())),
        Err(Error::Elf("table out of bounds"))
    );
    assert_eq!(
        tamper(&|b| b[58] = 40),
        Err(Error::Elf("table out of bounds"))
    );
    assert_eq!(tamper(&|b| b[62] = 3), Err(Error::Elf("section table")));
    assert_eq!(
        tamper(&|b| b[128 + 32..128 + 40].copy_from_slice(&3u64.to_le_bytes())),
        Err(Error::Elf("section out of bounds"))
    );
    assert_eq!(
        tamper(&|b| b[128 + 24..128 + 32].copy_from_slice(&u64::MAX.to_le_bytes())),
        Err(Error::Elf("section out of bounds"))
    );
    // An ELF with no executable section holds no kernel; the Windows driver
    // faults on it rather than refusing it, so decode must.
    assert_eq!(
        tamper(&|b| b[192 + 8] = 2),
        Err(Error::Elf("no executable kernel section"))
    );
    assert_eq!(
        tamper(&|b| b[192 + 4] = 3),
        Err(Error::Elf("no executable kernel section"))
    );
    assert_eq!(
        tamper(&|b| b[192 + 32] = 0),
        Err(Error::Elf("no executable kernel section"))
    );
    // A NOBITS section occupies no file bytes and may lie past the end.
    assert!(
        tamper(&|b| {
            b[128 + 4..128 + 8].copy_from_slice(&8u32.to_le_bytes());
            b[128 + 32..128 + 40].copy_from_slice(&4096u64.to_le_bytes());
        })
        .is_ok()
    );
    assert_eq!(
        tamper(&|b| {
            b[56..58].copy_from_slice(&1u16.to_le_bytes());
            b[32..40].copy_from_slice(&250u64.to_le_bytes());
        }),
        Err(Error::Elf("table out of bounds"))
    );
    // The SASS architecture in the ELF flags must agree with the header.
    let mut image = elf(120);
    image[49] = 100;
    assert_eq!(
        cuda_aot::decode(&reseal_image(&plan, 120, &image)),
        Err(Error::Elf("architecture differs from header"))
    );
}

#[test]
fn image_is_matched_to_its_program_and_device() {
    let plan = plan(SOURCE);
    let image = cuda_aot::decode(&container(&plan, 120, &elf(120))).unwrap();
    let header = image.header();
    assert_eq!(
        header.verify(&self::plan(OTHER)),
        Err(Error::Program("source_sha256"))
    );
    let mut forged = header.clone();
    forged.device_source_sha256[0] ^= 1;
    assert_eq!(
        forged.verify(&plan),
        Err(Error::Program("device_source_sha256"))
    );
    let mut forged = header.clone();
    forged.entry = "twice".into();
    assert_eq!(forged.verify(&plan), Err(Error::Program("entry")));
    let metal =
        Plan::from_source(SOURCE.as_bytes(), "run", Target::Metal, Limits::default()).unwrap();
    assert_eq!(header.verify(&metal), Err(Error::Program("target")));
    assert_eq!(
        header.admit_device(89, 13000),
        Err(Error::Sm {
            image: 120,
            device: 89
        })
    );
    assert_eq!(
        header.admit_device(120, 12080),
        Err(Error::Driver {
            required: 12090,
            found: 12080
        })
    );
    assert_eq!(header.admit_device(120, 12090), Ok(()));
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_build_and_run_refuse_without_the_cuda_feature() {
    let (status, report) = invar(&[
        "build", "x.inv", "--target", "cuda", "--entry", "run", "--sm", "120", "--output", "x.img",
    ]);
    assert!(!status);
    assert!(
        report["error"]
            .as_str()
            .unwrap()
            .contains("requires the cuda feature")
    );
}

/// A forced CUDA run is the compiled-ahead route only: without an image it
/// refuses before opening any device, and an image never applies elsewhere.
#[test]
fn forced_cuda_requires_an_image_and_images_apply_to_device_backends_only() {
    let dir = std::env::temp_dir().join(format!("invar-aot-format-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("p.inv");
    std::fs::write(&source, SOURCE).unwrap();
    let config = r#"{"fuel":10000,"transitions":100000,"trace_events":4000,"quantum":8,"rounds":4,"capacity":2}"#;
    let path = source.to_str().unwrap();
    let (ok, report) = invar(&["gnet-device", path, "run", "cuda", "[1,2]", config]);
    assert!(!ok);
    let error = report["error"].as_str().unwrap();
    if cfg!(feature = "cuda") {
        assert!(error.contains("compiled-ahead image"), "{report}");
        assert!(error.contains("INVAR_CUDA_RUNTIME_NVRTC=1"), "{report}");
        // The developer flag selects the runtime-NVRTC adapter instead.
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["gnet-device", path, "run", "cuda", "[1,2]", config])
            .env("INVAR_CUDA_RUNTIME_NVRTC", "1")
            .output()
            .unwrap();
        let flagged: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let flagged_error = flagged["error"].as_str().unwrap_or_default();
        assert!(!flagged_error.contains("compiled-ahead image"), "{flagged}");
    } else {
        assert!(error.contains("forced CUDA requires"), "{report}");
    }
    let (ok, report) = invar(&[
        "gnet-device",
        path,
        "run",
        "cpu-model-cuda",
        "[1,2]",
        config,
        "--artifact",
        path,
    ]);
    assert!(!ok);
    assert!(
        report["error"]
            .as_str()
            .unwrap()
            .contains("cuda and metal backends only")
    );
    std::fs::remove_dir_all(dir).unwrap();
}

fn invar(args: &[&str]) -> (bool, serde_json::Value) {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(args)
        .output()
        .unwrap();
    (
        output.status.success(),
        serde_json::from_slice(&output.stdout).unwrap(),
    )
}
