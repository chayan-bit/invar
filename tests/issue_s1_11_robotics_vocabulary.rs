use std::io::Write;
use std::process::{Command, Stdio};

fn joint_input<'a>(
    names: &'a [String],
    positions: &'a [f64],
    velocities: &'a [f64],
    efforts: &'a [f64],
) -> invar::robotics::RosJointState<'a> {
    invar::robotics::RosJointState {
        names,
        positions,
        velocities,
        efforts,
    }
}

#[test]
fn issue_s1_11_robotics_vocabulary_generic_record_types_check() {
    let source = br#"
        frame world;
        module robotics {
            pub struct Sample<T: Value, F: Frame, U: Unit, const N: u32> {
                scalar: T,
                offset: Quantity<T,U>,
                points: Vector3<T,U,F>
            }
        }
        import robotics::Sample;
        fn main() -> i32 {
            let x: Sample<i32,world,m,6> = Sample<i32,world,m,6> {
                scalar: 1i32,
                offset: quantity::<m>(2i32),
                points: vector3::<m,world>(1i32,2i32,3i32)
            };
            return x.scalar;
        }
    "#;
    let parsed = invar::front::parse(source).unwrap();
    invar::check::check(parsed, true).unwrap();
}

#[test]
fn issue_s1_11_robotics_vocabulary_standard_module_checks() {
    let source = include_str!("../lib/std/robotics/vocabulary.inv");
    let parsed = invar::front::parse(source.as_bytes()).unwrap();
    assert_eq!(parsed.records.len(), 16);
    let program = invar::front::parse(b"fn main() -> i32 { return 0; }").unwrap();
    invar::check::check(program, true).unwrap();
}

#[test]
fn issue_s1_11_robotics_vocabulary_all_wire_types_check_only() {
    let source = include_str!("fixtures/s1_11/all-port-types.inv");
    invar::check::check_port_types(invar::front::parse(source.as_bytes()).unwrap()).unwrap();
    let error =
        invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).unwrap_err();
    assert_eq!(error.code, "UNSUPPORTED_FEATURE");
    let custom_payload = b"struct Payload { value: i32 } fn port(x: read std::robotics::Stamped<Payload>) -> i32 { return 0; }";
    invar::check::check_port_types(invar::front::parse(custom_payload).unwrap()).unwrap();
}

#[test]
fn issue_s1_11_robotics_vocabulary_pose_composition_checks_frames() {
    assert_eq!(
        invar::robotics::compose_pose_type(
            "std::robotics::Pose<f64,world,base>",
            "std::robotics::Pose<f64,base,tool>",
        ),
        Ok("std::robotics::Pose<f64,world,tool>".to_owned())
    );
    assert_eq!(
        invar::robotics::compose_pose_type(
            "std::robotics::Pose<f64,world,base>",
            "std::robotics::Pose<f64,camera,tool>",
        ),
        Err("FRAME_MISMATCH")
    );
}

#[test]
fn issue_s1_11_robotics_vocabulary_pixel_is_nominal() {
    let good = include_str!("fixtures/s1_11/camera-px.inv");
    let parsed = invar::front::parse(good.as_bytes()).unwrap();
    invar::check::check_port_types(parsed).unwrap();
    let bad = include_str!("fixtures/s1_11/camera-wrong-focal-unit.inv");
    let error =
        invar::check::check(invar::front::parse(bad.as_bytes()).unwrap(), true).unwrap_err();
    assert_eq!(error.code, "UNIT_MISMATCH");
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "check",
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/s1_11/camera-wrong-focal-unit.inv"
            ),
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["diagnostics"][0]["code"], "UNIT_MISMATCH");
    let dimensionless = bad.replace("quantity::<m>(500)", "quantity::<one>(500)");
    assert_eq!(
        invar::check::check(invar::front::parse(dimensionless.as_bytes()).unwrap(), true)
            .unwrap_err()
            .code,
        "UNIT_MISMATCH"
    );
    for source in [
        "fn main() -> i32 { let x: Quantity<i32,m> = to::<m>(quantity::<px>(1)); return 0; }",
        "fn id<T: Value>(x: T) -> T { return x; } fn main() -> i32 { let x: Quantity<i32,m> = id(quantity::<px>(1)); return 0; }",
        "fn main() -> i32 { let x: Quantity<i32,px> = quantity::<px>(1) * quantity::<m>(1); return 0; }",
    ] {
        assert!(
            invar::check::check(invar::front::parse(source.as_bytes()).unwrap(), true).is_err()
        );
    }
}

#[test]
fn issue_s1_11_robotics_vocabulary_ros_mapping_golden() {
    let mapping = invar::robotics::ros_mapping().unwrap();
    assert_eq!(mapping["schema"], "invar-robotics-ros-mapping-v1");
    let records = mapping["records"].as_array().unwrap();
    let actual: Vec<(&str, &str)> = records
        .iter()
        .map(|entry| {
            (
                entry["invar"].as_str().unwrap(),
                entry["ros"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        actual,
        [
            ("Point", "geometry_msgs/msg/Point"),
            ("Vector3", "geometry_msgs/msg/Vector3"),
            ("Quaternion", "geometry_msgs/msg/Quaternion"),
            ("Pose", "geometry_msgs/msg/Pose"),
            ("Twist", "geometry_msgs/msg/Twist"),
            ("Wrench", "geometry_msgs/msg/Wrench"),
            ("JointPositions", "sensor_msgs/msg/JointState"),
            ("JointVelocities", "sensor_msgs/msg/JointState"),
            ("JointEfforts", "sensor_msgs/msg/JointState"),
            ("JointState", "sensor_msgs/msg/JointState"),
            ("JointWaypoint", "trajectory_msgs/msg/JointTrajectoryPoint"),
            ("JointTrajectory", "trajectory_msgs/msg/JointTrajectory"),
            ("Distortion", "sensor_msgs/msg/CameraInfo"),
            ("CameraIntrinsics", "sensor_msgs/msg/CameraInfo"),
            ("CameraExtrinsics", "geometry_msgs/msg/TransformStamped"),
            ("Stamped", "std_msgs/msg/Header"),
        ]
    );
    assert_eq!(
        mapping["pose_wrappers"],
        serde_json::json!([
            "geometry_msgs/msg/PoseStamped",
            "geometry_msgs/msg/TransformStamped"
        ])
    );
    assert_eq!(mapping["frame_id"], "declared_frame_name");
    assert_eq!(
        mapping["joint_state_conversion_boundary"],
        "invar::robotics::from_ros_joint_state"
    );
    assert_eq!(
        mapping["unit_scale_rules"]["mm"],
        serde_json::json!(["m", "1", "1000", "0"])
    );
    assert_eq!(
        mapping["unit_scale_rules"]["deg"],
        serde_json::json!(["rad", "1", "180", "1"])
    );
    assert_eq!(
        mapping["unit_scale_rules"]["px"],
        serde_json::json!(["px", "1", "1", "0"])
    );
    assert_eq!(mapping["quaternion_property"], "declared_unit_norm");
    assert_eq!(mapping["pose_equivalent_type"], "Transform<Child,Parent>");
    assert_eq!(
        mapping["joint_units"],
        "rad_or_m_by_declared_robot_joint_kind"
    );
}

#[test]
fn issue_s1_11_robotics_vocabulary_c_layouts_match_compiler() {
    use invar::robotics::Scalar;
    assert!(invar::robotics::c_header(Scalar::F64, 0).is_err());
    assert!(invar::robotics::c_header(Scalar::F64, 65).is_err());
    for (scalar, width) in [
        (Scalar::I32, 4),
        (Scalar::U32, 4),
        (Scalar::F32, 4),
        (Scalar::F64, 8),
    ] {
        let header = invar::robotics::c_header(scalar, 7).unwrap();
        let alignment = width.max(8);
        let waypoint = if width == 4 { 64 } else { 120 };
        let trajectory = if width == 4 { 4104 } else { 7688 };
        let distortion = if width == 4 { 24 } else { 48 };
        let camera_offset = 8 + 4 * width;
        let camera = camera_offset + distortion;
        let positions_offset = if width == 4 { 28 } else { 32 };
        let checks = format!(
            r#"
            #define CHECK(T, SIZE, ALIGN) _Static_assert(sizeof(T) == (SIZE) && _Alignof(T) == (ALIGN), #T)
            #define OFFSET(T, F, O) _Static_assert(__builtin_offsetof(T, F) == (O), #T "." #F)
            CHECK(invar_point, {point}, {width});
            CHECK(invar_vector3, {point}, {width});
            CHECK(invar_quaternion, {quat}, {width});
            CHECK(invar_pose, {pose}, {width}); OFFSET(invar_pose, rotation, {point});
            CHECK(invar_twist, {twist}, {width}); OFFSET(invar_twist, angular, {point});
            CHECK(invar_wrench, {twist}, {width}); OFFSET(invar_wrench, torque, {point});
            CHECK(invar_joint_positions, {joint}, {width});
            CHECK(invar_joint_velocities, {joint}, {width});
            CHECK(invar_joint_efforts, {joint}, {width});
            CHECK(invar_joint_state, {state}, {width}); OFFSET(invar_joint_state, name_ids, 0); OFFSET(invar_joint_state, positions, {positions_offset}); OFFSET(invar_joint_state, efforts, {efforts_offset});
            CHECK(invar_joint_waypoint, {waypoint}, {alignment}); OFFSET(invar_joint_waypoint, time_from_start_ns, {time_offset});
            CHECK(invar_joint_trajectory, {trajectory}, {alignment}); OFFSET(invar_joint_trajectory, count, {count_offset});
            CHECK(invar_distortion, {distortion}, {width}); OFFSET(invar_distortion, k1, {distortion_offset});
            CHECK(invar_camera_intrinsics, {camera}, {width}); OFFSET(invar_camera_intrinsics, distortion, {camera_offset});
            CHECK(invar_camera_extrinsics, {pose}, {width});
            CHECK(invar_stamped_pose, {stamped}, {alignment}); OFFSET(invar_stamped_pose, frame_id, 8); OFFSET(invar_stamped_pose, value, {stamped_value_offset});
        "#,
            point = 3 * width,
            quat = 4 * width,
            pose = 7 * width,
            twist = 6 * width,
            joint = 7 * width,
            state = positions_offset + 21 * width,
            positions_offset = positions_offset,
            efforts_offset = positions_offset + 14 * width,
            time_offset = 14 * width,
            waypoint = waypoint,
            trajectory = trajectory,
            count_offset = 64 * waypoint,
            distortion = distortion,
            distortion_offset = if width == 4 { 4 } else { 8 },
            camera = camera,
            camera_offset = camera_offset,
            stamped = if width == 4 { 40 } else { 72 },
            stamped_value_offset = if width == 4 { 12 } else { 16 },
        );
        let mut child = Command::new("cc")
            .args(["-std=c11", "-x", "c", "-fsyntax-only", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("C compiler required for S1-11 layout gate");
        child
            .stdin
            .take()
            .unwrap()
            .write_all(format!("{header}\n{checks}").as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{scalar:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn issue_s1_11_robotics_vocabulary_joint_state_external_boundary() {
    use invar::robotics::{RosJointState, from_ros_joint_state};
    let expected = ["shoulder", "elbow"];
    let ordered = vec!["shoulder".to_owned(), "elbow".to_owned()];
    let positions = [1.0, 2.0];
    let velocities = [0.1, 0.2];
    let efforts = [3.0, 4.0];
    let wire = from_ros_joint_state(
        &expected,
        joint_input(&ordered, &positions, &velocities, &efforts),
    )
    .unwrap();
    assert_eq!(wire.name_ids, [0, 1]);
    assert_eq!(wire.positions, positions);
    assert_eq!(wire.velocities, velocities);
    assert_eq!(wire.efforts, efforts);
    let missing = vec!["shoulder".to_owned()];
    let unknown = vec!["shoulder".to_owned(), "wrist".to_owned()];
    let duplicate = vec!["shoulder".to_owned(), "shoulder".to_owned()];
    let reordered = vec!["elbow".to_owned(), "shoulder".to_owned()];
    for names in [&missing, &unknown, &duplicate, &reordered] {
        let error = from_ros_joint_state(
            &expected,
            joint_input(names, &positions, &velocities, &efforts),
        )
        .unwrap_err();
        assert_eq!(error.code(), "S1701");
        let document = error.diagnostic();
        assert_eq!(
            document["diagnostics"][0]["suggested_fixes"],
            serde_json::json!([])
        );
        assert!(jsonschema::is_valid(
            &serde_json::from_str(include_str!("../schema/invar-diagnostics-v1.schema.json"))
                .unwrap(),
            &document
        ));
    }
    let bad_length = RosJointState {
        names: &ordered,
        positions: &positions[..1],
        velocities: &velocities,
        efforts: &efforts,
    };
    assert_eq!(
        from_ros_joint_state(&expected, bad_length)
            .unwrap_err()
            .code(),
        "S1701"
    );
    let nonfinite = [f64::NAN, 2.0];
    let bad_value = RosJointState {
        names: &ordered,
        positions: &nonfinite,
        velocities: &velocities,
        efforts: &efforts,
    };
    assert_eq!(
        from_ros_joint_state(&expected, bad_value)
            .unwrap_err()
            .code(),
        "S1701"
    );
}

#[test]
fn issue_s1_11_robotics_vocabulary_camera_distortion_ir_roundtrip() {
    use invar::robotics::{CameraIntrinsics, Distortion};
    let base = CameraIntrinsics {
        scalar: "f64".into(),
        width_px: "640".into(),
        height_px: "480".into(),
        fx_px: "500.25".into(),
        fy_px: "500.25".into(),
        cx_px: "320".into(),
        cy_px: "240".into(),
        distortion: Distortion::None,
    };
    for distortion in [
        Distortion::None,
        Distortion::PlumbBob([
            "0.1".into(),
            "-0.02".into(),
            "0".into(),
            "0".into(),
            "0.003".into(),
        ]),
        Distortion::Equidistant(["0.1".into(), "-0.02".into(), "0.003".into(), "0.004".into()]),
    ] {
        let camera = CameraIntrinsics {
            distortion,
            ..base.clone()
        };
        let ir = camera.to_ir().unwrap();
        assert_eq!(CameraIntrinsics::from_ir(&ir).unwrap(), camera);
        assert_eq!(CameraIntrinsics::from_ir(&ir).unwrap().to_ir().unwrap(), ir);
    }
    let mut bad = base.to_ir().unwrap();
    bad["distortion"]["model"] = serde_json::json!("equidistant");
    assert!(CameraIntrinsics::from_ir(&bad).is_err());
    bad["distortion"]["coefficients"] = serde_json::json!(["0", "0", "0", "0", "0", "0"]);
    assert!(CameraIntrinsics::from_ir(&bad).is_err());
    let mut noncanonical = base.to_ir().unwrap();
    noncanonical["fx_px"] = serde_json::json!("500.250");
    assert!(CameraIntrinsics::from_ir(&noncanonical).is_err());
    let mut invalid_width = base.to_ir().unwrap();
    invalid_width["width_px"] = serde_json::json!("0");
    assert!(CameraIntrinsics::from_ir(&invalid_width).is_err());
}

#[test]
fn issue_s1_11_robotics_vocabulary_rejects_mismatched_units_and_joint_counts() {
    let twist = include_str!("fixtures/s1_11/twist-wrong-angular-unit.inv");
    let error =
        invar::check::check(invar::front::parse(twist.as_bytes()).unwrap(), true).unwrap_err();
    assert_eq!(error.code, "TYPE_MISMATCH");
    let output = Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "check",
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/s1_11/twist-wrong-angular-unit.inv"
            ),
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["diagnostics"][0]["code"], "TYPE_MISMATCH");
    let joints = include_str!("fixtures/s1_11/joint-count-mismatch.inv");
    let error = invar::check::check_port_types(invar::front::parse(joints.as_bytes()).unwrap())
        .unwrap_err();
    assert_eq!(error.code, "TYPE_MISMATCH");
}
