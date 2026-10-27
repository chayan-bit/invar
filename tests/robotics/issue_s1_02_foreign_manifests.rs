use std::{path::Path, process::Command};

use invar::{front, system};
use invar_foreign::component::{Component, ComponentError, State};

const MANIFEST: &str = include_str!("../fixtures/s1_02/manifest-valid.inv");

fn build_planar(fixture: &Path, directory: &Path) -> std::path::PathBuf {
    let source = std::fs::read(fixture.join("manifest.inv")).unwrap();
    let manifest = system::compile_manifest(front::parse_system(&source).unwrap()).unwrap();
    let header = system::foreign::header(&manifest).unwrap();
    std::fs::create_dir_all(directory).unwrap();
    std::fs::write(directory.join("invar_planar2_bridge_v1.h"), header).unwrap();
    let library = directory.join(if cfg!(target_os = "macos") {
        "libplanar.dylib"
    } else {
        "libplanar.so"
    });
    let output = Command::new("cc")
        .arg(if cfg!(target_os = "macos") {
            "-dynamiclib"
        } else {
            "-shared"
        })
        .args([
            "-fPIC",
            "-fvisibility=hidden",
            "-std=c11",
            "-Wall",
            "-Werror",
        ])
        .arg("-I")
        .arg(directory)
        .arg(fixture.join("bridge.c"))
        .arg(fixture.join("planar2.c"))
        .arg("-lm")
        .arg("-o")
        .arg(&library)
        .output()
        .unwrap();
    crate::child_deadline::assert_success(&output);
    library
}

fn keep_passing_evidence(path: &Path) {
    if let Some(directory) = std::env::var_os("INVAR_S1_02_EVIDENCE_DIR") {
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory).unwrap();
        let destination = directory.join(path.file_name().unwrap());
        let source = std::fs::read(path).unwrap();
        if destination.exists() {
            assert_eq!(std::fs::read(destination).unwrap(), source);
        } else {
            std::fs::write(destination, source).unwrap();
        }
    }
}

#[test]
fn issue_s1_02_foreign_manifests_typed_config_and_tolerance_are_checked() {
    let program = front::parse_system(MANIFEST.as_bytes()).unwrap();
    let manifest = system::compile_manifest(program).unwrap();
    assert_eq!(manifest["abi"]["prefix"], "invar_fk_bridge_v1");
    assert_eq!(manifest["config"]["layout"], "inv_fk_config_v1");
    assert_eq!(manifest["golden"][0]["tolerances"][0]["unit"], "m");
    assert_eq!(manifest["golden"][0]["tolerances"][1]["unit"], "rad");

    let unknown = MANIFEST.replace("inv_fk_config_v1", "unknown_layout");
    let error =
        system::compile_manifest(front::parse_system(unknown.as_bytes()).unwrap()).unwrap_err();
    assert_eq!(error.code, "S1400");

    let negative = MANIFEST.replace("abs 0.000000000001 m", "abs -1 m");
    let error =
        system::compile_manifest(front::parse_system(negative.as_bytes()).unwrap()).unwrap_err();
    assert_eq!(error.code, "S1400");
}

#[test]
fn issue_s1_02_foreign_manifests_public_source_code_is_registered() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/s1_02/manifest-unknown-config.inv");
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["check", path.to_str().unwrap(), "--format", "json"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["diagnostics"][0]["code"], "S1400");
}

#[test]
fn issue_s1_02_foreign_manifests_header_is_deterministic_and_c11_layout_checked() {
    let manifest =
        system::compile_manifest(front::parse_system(MANIFEST.as_bytes()).unwrap()).unwrap();
    let header = system::foreign::header(&manifest).unwrap();
    assert_eq!(header, system::foreign::header(&manifest).unwrap());
    assert!(header.contains("invar_fk_bridge_v1_configure"));
    assert!(header.contains("model_xml_path_len) == 8"));
    assert!(header.contains("quaternion_xyzw) == 24"));
    let mut forged = manifest.clone();
    let q = forged["ports"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|port| port["name"] == "q")
        .unwrap();
    q["layout"] = serde_json::json!("inv_pose_f64");
    assert!(system::foreign::header(&forged).is_err());

    let directory = std::env::temp_dir().join(format!("invar-s1-02-header-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join("fk.h"), header).unwrap();
    std::fs::write(
        directory.join("layout.c"),
        "#include \"fk.h\"\nint main(void) { return 0; }\n",
    )
    .unwrap();
    let output = Command::new("cc")
        .args(["-std=c11", "-Wall", "-Werror", "-fsyntax-only"])
        .arg(directory.join("layout.c"))
        .current_dir(&directory)
        .output()
        .unwrap();
    crate::child_deadline::assert_success(&output);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn issue_s1_02_foreign_manifests_versioned_system_ir_round_trips() {
    let manifest = MANIFEST.replace(
        "layout c \"inv_joint7_f64\";",
        "layout c \"inv_joint7_f64\" optional;",
    );
    let source = format!(
        "{manifest}\nproduct Cell {{ distribution internal; commercial false; robot menagerie::franka_emika_panda; capacity cpu_cores = 2, memory = 1 GiB; }}\nsystem cell for Cell {{ fk: fk; }}\n"
    );
    let ir = system::compile(front::parse_system(source.as_bytes()).unwrap()).unwrap();
    let document: serde_json::Value = serde_json::from_str(&ir.canonical_json()).unwrap();
    assert_eq!(document["schema"], "invar-system-v12");
    let schema: serde_json::Value = serde_json::from_str(system::SCHEMA_JSON).unwrap();
    assert!(jsonschema::is_valid(&schema, &document));
    let restored = system::SystemIr::from_json(ir.canonical_json().as_bytes()).unwrap();
    assert_eq!(restored.canonical_json(), ir.canonical_json());
}

#[test]
fn issue_s1_02_foreign_manifests_hash_tracks_declared_transitive_inputs() {
    let root = std::env::temp_dir().join(format!("invar-s1-02-digest-{}", std::process::id()));
    let component = root.join("component");
    std::fs::create_dir_all(component.join("shim")).unwrap();
    std::fs::create_dir_all(component.join("golden")).unwrap();
    for (name, bytes) in [
        ("common.h", b"header-v1".as_slice()),
        ("child.py", b"child-v1".as_slice()),
        ("model.xml", b"model-v1".as_slice()),
    ] {
        std::fs::write(root.join(name), bytes).unwrap();
    }
    std::fs::write(component.join("shim/fk_bridge.c"), b"bridge-v1").unwrap();
    std::fs::write(component.join("golden/panda_fk.json"), b"golden-v1").unwrap();
    let source = MANIFEST.replace(
        "    shim \"shim/fk_bridge.c\";",
        "    input \"../common.h\";\n    input \"../child.py\";\n    input \"../model.xml\";\n    shim \"shim/fk_bridge.c\";",
    );
    let path = component.join("manifest.inv");
    std::fs::write(&path, &source).unwrap();
    let manifest =
        system::compile_manifest(front::parse_system(source.as_bytes()).unwrap()).unwrap();
    let header = system::foreign::header(&manifest).unwrap();
    let (original, records) =
        system::foreign::manifest_hash(&root, &path, &manifest, &header).unwrap();
    assert!(records.iter().all(|(_, name, _)| !name.starts_with('/')));
    for (name, original_bytes) in [
        ("common.h", b"header-v1".as_slice()),
        ("child.py", b"child-v1".as_slice()),
        ("model.xml", b"model-v1".as_slice()),
    ] {
        std::fs::write(root.join(name), format!("{name}-changed")).unwrap();
        let (changed, _) =
            system::foreign::manifest_hash(&root, &path, &manifest, &header).unwrap();
        assert_ne!(
            changed, original,
            "{name} was omitted from evidence identity"
        );
        std::fs::write(root.join(name), original_bytes).unwrap();
    }
    std::fs::write(
        component.join("shim/fk_bridge.c"),
        b"#include \"../../common.h\"\n",
    )
    .unwrap();
    assert!(system::foreign::manifest_hash(&root, &path, &manifest, &header).is_ok());
    let undeclared = source.replace("    input \"../common.h\";\n", "");
    let undeclared_manifest =
        system::compile_manifest(front::parse_system(undeclared.as_bytes()).unwrap()).unwrap();
    assert!(system::foreign::manifest_hash(&root, &path, &undeclared_manifest, &header).is_err());
    std::fs::write(component.join("shim/fk_bridge.c"), b"bridge-v1").unwrap();
    let duplicate = source.replace(
        "    input \"../common.h\";",
        "    input \"../component/shim/fk_bridge.c\";",
    );
    let duplicate_manifest =
        system::compile_manifest(front::parse_system(duplicate.as_bytes()).unwrap()).unwrap();
    assert!(system::foreign::manifest_hash(&root, &path, &duplicate_manifest, &header).is_err());
    let repeated = source.replace(
        "    input \"../common.h\";",
        "    input \"../common.h\";\n    input \"../common.h\";",
    );
    assert_eq!(
        system::compile_manifest(front::parse_system(repeated.as_bytes()).unwrap())
            .unwrap_err()
            .code,
        "S1400"
    );
    let outside = std::env::temp_dir().join(format!("invar-s1-02-outside-{}", std::process::id()));
    std::fs::write(&outside, b"outside").unwrap();
    let escaped = source.replace(
        "../common.h",
        &format!("../../{}", outside.file_name().unwrap().to_str().unwrap()),
    );
    let escaped_manifest =
        system::compile_manifest(front::parse_system(escaped.as_bytes()).unwrap()).unwrap();
    assert!(system::foreign::manifest_hash(&root, &path, &escaped_manifest, &header).is_err());
    std::fs::remove_file(outside).unwrap();
    std::fs::remove_file(root.join("common.h")).unwrap();
    assert!(system::foreign::manifest_hash(&root, &path, &manifest, &header).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn issue_s1_02_foreign_manifests_public_generator_is_deterministic_and_preserves_outputs() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = root.join("tests/fixtures/s1_02/manifest-valid.inv");
    let base = std::env::temp_dir().join(format!("invar-s1-02-generate-{}", std::process::id()));
    std::fs::create_dir(&base).unwrap();
    let first = base.join("first");
    let second = base.join("second");
    for output in [&first, &second] {
        let result = Command::new(env!("CARGO_BIN_EXE_invar"))
            .arg("component")
            .arg("generate")
            .arg(&manifest)
            .arg("--output")
            .arg(output)
            .output()
            .unwrap();
        assert_eq!(
            result.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    for name in ["invar_fk_bridge_v1.h", "invar_fk_bridge_v1.manifest.json"] {
        assert_eq!(
            std::fs::read(first.join(name)).unwrap(),
            std::fs::read(second.join(name)).unwrap()
        );
    }
    let before = std::fs::read(first.join("invar_fk_bridge_v1.h")).unwrap();
    let rejected = Command::new(env!("CARGO_BIN_EXE_invar"))
        .arg("component")
        .arg("generate")
        .arg(&manifest)
        .arg("--output")
        .arg(&first)
        .output()
        .unwrap();
    assert_eq!(rejected.status.code(), Some(2));
    assert_eq!(
        std::fs::read(first.join("invar_fk_bridge_v1.h")).unwrap(),
        before
    );
    std::fs::remove_dir_all(base).unwrap();
}

#[test]
fn issue_s1_02_foreign_manifests_robotics_inputs_are_closed_and_pinned() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (name, expected) in [
        ("pinocchio_fk", "models/franka_emika_panda/panda.xml"),
        ("process_fixture", "process/fixture_server.py"),
    ] {
        let path = root.join(format!(
            "examples/cells/reach_place/components/{name}/manifest.inv"
        ));
        let source = std::fs::read(&path).unwrap();
        let manifest = system::compile_manifest(front::parse_system(&source).unwrap()).unwrap();
        let header = system::foreign::header(&manifest).unwrap();
        let (hash, inputs) =
            system::foreign::manifest_hash(root, &path, &manifest, &header).unwrap();
        assert_eq!(hash.len(), 64);
        assert!(
            inputs.iter().any(|(_, path, _)| path.ends_with(expected)),
            "{name}"
        );
        assert!(inputs.iter().any(|(role, _, _)| role == "generated-header"));
        assert!(inputs.iter().any(|(role, _, _)| role == "shim"));
    }
}

#[test]
fn issue_s1_02_foreign_manifests_planar2_generated_adapter_matches_independent_goldens() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("tests/fixtures/foreign-components/planar2");
    let source = std::fs::read(fixture.join("manifest.inv")).unwrap();
    let manifest = system::compile_manifest(front::parse_system(&source).unwrap()).unwrap();
    let header = system::foreign::header(&manifest).unwrap();
    let (digest, inputs) =
        system::foreign::manifest_hash(root, &fixture.join("manifest.inv"), &manifest, &header)
            .unwrap();
    assert_eq!(digest.len(), 64);
    assert!(
        inputs
            .iter()
            .any(|(role, name, _)| role == "shim" && name.ends_with("planar2/bridge.c"))
    );
    let directory = std::env::temp_dir().join(format!("invar-s1-02-planar-{}", std::process::id()));
    let library = build_planar(&fixture, &directory);
    let mut component = Component::open(&library, "invar_planar2_bridge_v1", 2).unwrap();
    assert_eq!(component.version(), "invar-component-v1/planar2-fixture-1");
    assert_eq!(component.step(&[0.0, 0.0]), Err(ComponentError::Lifecycle));
    component.init(None).unwrap();
    assert_eq!(component.step(&[0.0, 0.0]), Err(ComponentError::Lifecycle));
    component.configure_planar2(0.4, 0.3).unwrap();
    assert_eq!(component.state(), State::Configured);
    assert_eq!(component.step(&[f64::NAN, 0.0]), Err(ComponentError::Input));
    let goldens: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture.join("planar2_goldens.json")).unwrap())
            .unwrap();
    let vectors = goldens["vectors"].as_array().unwrap();
    assert_eq!(vectors.len(), 7);
    for vector in vectors {
        let q = vector["q_rad"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>();
        let pose = component.step(&q).unwrap();
        let yaw = vector["yaw_rad"].as_f64().unwrap();
        assert!((pose.translation_m[0] - vector["x_m"].as_f64().unwrap()).abs() <= 1e-12);
        assert!((pose.translation_m[1] - vector["y_m"].as_f64().unwrap()).abs() <= 1e-12);
        assert_eq!(pose.translation_m[2], 0.0);
        assert!((pose.quaternion_xyzw[2] - (yaw / 2.0).sin()).abs() <= 1e-12);
        assert!((pose.quaternion_xyzw[3] - (yaw / 2.0).cos()).abs() <= 1e-12);
        let norm: f64 = pose.quaternion_xyzw.iter().map(|x| x * x).sum();
        assert!((norm.sqrt() - 1.0).abs() <= 1e-12);
    }
    component.shutdown().unwrap();
    assert_eq!(component.step(&[0.0, 0.0]), Err(ComponentError::Lifecycle));
    drop(component);
    let evidence_dir = directory.join("evidence");
    let result = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["component", "test"])
        .arg(fixture.join("manifest.inv"))
        .arg("--library")
        .arg(&library)
        .arg("--output")
        .arg(&evidence_dir)
        .output()
        .unwrap();
    crate::child_deadline::assert_success(&result);
    let outcome: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    let evidence = std::path::PathBuf::from(outcome["evidence"].as_str().unwrap());
    system::foreign::verify_evidence(&evidence, &digest).unwrap();
    keep_passing_evidence(&evidence);
    assert!(system::foreign::verify_evidence(&evidence, &"0".repeat(64)).is_err());
    assert!(system::foreign::verify_evidence(&evidence_dir.join("absent.json"), &digest).is_err());
    let mut incomplete: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&evidence).unwrap()).unwrap();
    incomplete
        .as_object_mut()
        .unwrap()
        .remove("library_version");
    let incomplete_path = evidence_dir.join("incomplete.json");
    std::fs::write(&incomplete_path, serde_json::to_vec(&incomplete).unwrap()).unwrap();
    assert!(system::foreign::verify_evidence(&incomplete_path, &digest).is_err());
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn issue_s1_02_foreign_manifests_external_library_failures() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = root.join("tests/fixtures/foreign-components/planar2/manifest.inv");
    let directory =
        std::env::temp_dir().join(format!("invar-s1-02-bad-library-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let missing = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["component", "test"])
        .arg(&manifest)
        .arg("--library")
        .arg(directory.join("absent.so"))
        .arg("--output")
        .arg(directory.join("evidence"))
        .output()
        .unwrap();
    assert_eq!(missing.status.code(), Some(2));
    let error: serde_json::Value = serde_json::from_slice(&missing.stderr).unwrap();
    assert_eq!(error["code"], "S1401");
    let empty = directory.join("empty.c");
    std::fs::write(&empty, "int no_component_symbols = 1;\n").unwrap();
    let library = directory.join(if cfg!(target_os = "macos") {
        "empty.dylib"
    } else {
        "empty.so"
    });
    let compiled = Command::new("cc")
        .arg(if cfg!(target_os = "macos") {
            "-dynamiclib"
        } else {
            "-shared"
        })
        .arg("-fPIC")
        .arg(&empty)
        .arg("-o")
        .arg(&library)
        .output()
        .unwrap();
    crate::child_deadline::assert_success(&compiled);
    let result = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["component", "test"])
        .arg(&manifest)
        .arg("--library")
        .arg(&library)
        .arg("--output")
        .arg(directory.join("evidence"))
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    let error: serde_json::Value = serde_json::from_slice(&result.stderr).unwrap();
    assert_eq!(error["code"], "S1401");
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn issue_s1_02_foreign_manifests_external_golden_failure() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("tests/fixtures/foreign-components/planar2");
    let copied =
        std::env::temp_dir().join(format!("invar-s1-02-golden-failure-{}", std::process::id()));
    std::fs::create_dir_all(&copied).unwrap();
    for name in [
        "manifest.inv",
        "bridge.c",
        "planar2.c",
        "planar2.h",
        "generate_goldens.py",
        "planar2_goldens.json",
    ] {
        std::fs::copy(fixture.join(name), copied.join(name)).unwrap();
    }
    let golden_path = copied.join("planar2_goldens.json");
    let mut golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&golden_path).unwrap()).unwrap();
    golden["vectors"][0]["x_m"] = serde_json::json!(0.8);
    std::fs::write(&golden_path, serde_json::to_vec(&golden).unwrap()).unwrap();
    let build = std::env::temp_dir().join(format!("invar-s1-02-golden-lib-{}", std::process::id()));
    let library = build_planar(&copied, &build);
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["component", "test"])
        .arg(copied.join("manifest.inv"))
        .arg("--library")
        .arg(&library)
        .arg("--output")
        .arg(build.join("evidence"))
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let error: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(error["code"], "S1403");
    assert!(!build.join("evidence").exists());
    std::fs::remove_dir_all(copied).unwrap();
    std::fs::remove_dir_all(build).unwrap();
}

#[cfg(target_os = "linux")]
fn build_process() -> (std::path::PathBuf, std::path::PathBuf) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("examples/cells/reach_place/components/process_fixture");
    let source = std::fs::read(fixture.join("manifest.inv")).unwrap();
    let manifest = system::compile_manifest(front::parse_system(&source).unwrap()).unwrap();
    let directory =
        std::env::temp_dir().join(format!("invar-s1-02-process-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join("invar_process_bridge_v1.h"),
        system::foreign::header(&manifest).unwrap(),
    )
    .unwrap();
    let library = directory.join("libinvar_process_bridge_v1.so");
    let result = Command::new("cc")
        .args([
            "-shared", "-fPIC", "-std=c11", "-Wall", "-Wextra", "-Werror",
        ])
        .arg("-I")
        .arg(&directory)
        .arg(fixture.join("bridge.c"))
        .arg(fixture.join("process_reference.c"))
        .arg(format!(
            "-Wl,--version-script={}",
            fixture.join("bridge_exports.map").display()
        ))
        .args(["-lm", "-o"])
        .arg(&library)
        .output()
        .unwrap();
    crate::child_deadline::assert_success(&result);
    (directory, library)
}

#[cfg(target_os = "linux")]
fn process_command(library: &Path, directory: &Path, mode: &str) -> std::process::Output {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = root.join("examples/cells/reach_place/components/process_fixture/manifest.inv");
    let child = root.join("tests/fixtures/foreign-components/process/fixture_server.py");
    Command::new(env!("CARGO_BIN_EXE_invar"))
        .args(["component", "test"])
        .arg(&manifest)
        .arg("--library")
        .arg(library)
        .arg("--child")
        .arg(child)
        .arg("--model")
        .arg(&manifest)
        .arg("--output")
        .arg(directory.join("evidence"))
        .env("INVAR_PROCESS_FIXTURE_MODE", mode)
        .output()
        .unwrap()
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires pinned ROS 2 Lyrical Linux environment"]
fn issue_s1_02_foreign_manifests_env_process_start_step_shutdown() {
    let (directory, library) = build_process();
    let output = process_command(&library, &directory, "normal");
    crate::child_deadline::assert_success(&output);
    let record: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(record["golden_vectors"], 3);
    let evidence = std::path::PathBuf::from(record["evidence"].as_str().unwrap());
    system::foreign::verify_evidence(&evidence, record["manifest_hash"].as_str().unwrap()).unwrap();
    keep_passing_evidence(&evidence);
    std::fs::remove_dir_all(directory).unwrap();
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires pinned ROS 2 Lyrical Linux environment"]
fn issue_s1_02_foreign_manifests_env_process_crash_is_typed() {
    let (directory, library) = build_process();
    let output = process_command(&library, &directory, "crash_step");
    assert_eq!(output.status.code(), Some(2));
    let record: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(record["code"], "S1406");
    assert!(!directory.join("evidence").exists());
    std::fs::remove_dir_all(directory).unwrap();
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires pinned ROS 2 Lyrical Linux environment"]
fn issue_s1_02_foreign_manifests_env_process_malformed_is_typed() {
    let (directory, library) = build_process();
    let output = process_command(&library, &directory, "bad_magic_step");
    assert_eq!(output.status.code(), Some(2));
    let record: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(record["code"], "S1407");
    assert!(!directory.join("evidence").exists());
    std::fs::remove_dir_all(directory).unwrap();
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires ROS 2 Lyrical Pinocchio 4.1.0 and pinned Panda/UR5e MJCF"]
fn issue_s1_02_foreign_manifests_env_pinocchio_independent_goldens() {
    use sha2::{Digest, Sha256};

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("examples/cells/reach_place/components/pinocchio_fk");
    let arms = [
        (
            "panda",
            "manifest.inv",
            "tests/fixtures/foreign-components/models/franka_emika_panda/panda.xml",
            "96ad67da03710f17f798c9478fd9e9efdf24a3bf8359f05e456dd9fb158ea273",
            "pinocchio_fk_generated",
            4,
        ),
        (
            "ur5e",
            "manifest-ur5e.inv",
            "examples/cells/assets/menagerie/universal_robots_ur5e/ur5e.xml",
            "ffd0a7b336d3c2d27c35a15cc0d4aaa95196769aca190ac8d6522d3318c345a0",
            "pinocchio_fk_ur5e_generated",
            4,
        ),
    ];
    assert_eq!(
        std::fs::read(fixture.join("golden/panda_fk.json")).unwrap(),
        std::fs::read(root.join("tests/fixtures/foreign-components/pinocchio_goldens.json"))
            .unwrap()
    );
    let directory =
        std::env::temp_dir().join(format!("invar-s1-02-pinocchio-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    for (arm, manifest_file, _, _, _, _) in arms {
        let source = std::fs::read(fixture.join(manifest_file)).unwrap();
        let manifest = system::compile_manifest(front::parse_system(&source).unwrap()).unwrap();
        let generated = directory.join(format!("generated-{arm}"));
        std::fs::create_dir(&generated).unwrap();
        std::fs::write(
            generated.join("invar_pinocchio_bridge_v1.h"),
            system::foreign::header(&manifest).unwrap(),
        )
        .unwrap();
    }
    let build = directory.join("build");
    let configured = Command::new("cmake")
        .arg("-S")
        .arg(root.join("tests/fixtures/foreign-components"))
        .arg("-B")
        .arg(&build)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg(format!(
            "-DINVAR_GENERATED_INCLUDE_DIR={}",
            directory.join("generated-panda").display()
        ))
        .arg(format!(
            "-DINVAR_GENERATED_UR5E_INCLUDE_DIR={}",
            directory.join("generated-ur5e").display()
        ))
        .output()
        .unwrap();
    crate::child_deadline::assert_success(&configured);
    let built = Command::new("cmake")
        .arg("--build")
        .arg(&build)
        .args([
            "--target",
            "pinocchio_fk_generated",
            "pinocchio_fk_ur5e_generated",
            "--parallel",
            "2",
        ])
        .output()
        .unwrap();
    crate::child_deadline::assert_success(&built);
    for (arm, manifest_file, model_file, expected, target, vectors) in arms {
        let model = root.join(model_file);
        let actual = Sha256::digest(std::fs::read(&model).unwrap())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(actual, expected, "{arm} model must match oracle provenance");
        let library = build.join(format!("lib{target}.so"));
        let tested = Command::new(env!("CARGO_BIN_EXE_invar"))
            .args(["component", "test"])
            .arg(fixture.join(manifest_file))
            .arg("--library")
            .arg(&library)
            .arg("--model")
            .arg(&model)
            .arg("--output")
            .arg(directory.join("evidence"))
            .output()
            .unwrap();
        assert!(
            tested.status.success(),
            "{arm}: {}",
            String::from_utf8_lossy(&tested.stderr)
        );
        let record: serde_json::Value = serde_json::from_slice(&tested.stdout).unwrap();
        assert_eq!(record["golden_vectors"], vectors);
        let evidence = std::path::PathBuf::from(record["evidence"].as_str().unwrap());
        system::foreign::verify_evidence(&evidence, record["manifest_hash"].as_str().unwrap())
            .unwrap();
        keep_passing_evidence(&evidence);
    }
    std::fs::remove_dir_all(directory).unwrap();
}
