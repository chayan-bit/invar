//! Shipped standard robotics vocabulary and its ROS conversion data.

pub mod planner;

pub const ROS_MAPPING_JSON: &str = include_str!("../lib/std/robotics/ros-mapping-v1.json");
const C_HEADER: &str = include_str!("../lib/std/robotics/layout-header-v1.h.in");

pub(crate) fn generic_parts(name: &str) -> Option<(&str, Vec<&str>)> {
    let (head, rest) = name.split_once('<')?;
    let content = rest.strip_suffix('>')?;
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut args = Vec::new();
    for (index, byte) in content.bytes().enumerate() {
        match byte {
            b'<' => depth += 1,
            b'>' => depth = depth.checked_sub(1)?,
            b',' if depth == 0 => {
                args.push(&content[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return None;
    }
    args.push(&content[start..]);
    Some((head, args))
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    value
        .checked_add(alignment.checked_sub(1)?)?
        .checked_div(alignment)?
        .checked_mul(alignment)
}

pub(crate) fn is_planner_record(name: &str) -> bool {
    matches!(
        name,
        "std::robotics::PlannerRequest"
            | "std::robotics::PlannerWaypoint"
            | "std::robotics::PlannerSegment"
            | "std::robotics::PlannerTrajectory"
    )
}

/// Joint count of an exact admitted planner-v1 wire type; frames are checked by
/// the normal generic-record/frame checker, never inferred from a robot name.
pub fn planner_joint_count(name: &str) -> Option<u32> {
    let (head, args) = generic_parts(name)?;
    if !is_planner_record(head)
        || args.first().copied()? != "f64"
        || args.len()
            != if head == "std::robotics::PlannerRequest" {
                5
            } else {
                2
            }
    {
        return None;
    }
    args.get(1)?
        .parse::<u32>()
        .ok()
        .filter(|n| matches!(n, 6 | 7))
}

fn c_layout(ty: &str, depth: usize) -> Option<(u64, u64)> {
    if depth > 8 {
        return None;
    }
    match ty {
        "i32" | "u32" | "f32" => return Some((4, 4)),
        "i64" | "f64" => return Some((8, 8)),
        _ => {}
    }
    let (head, args) = generic_parts(ty)?;
    if is_planner_record(head) {
        let joints = u64::from(planner_joint_count(ty)?);
        let waypoint = 8 + 16 * joints;
        let size = match head {
            "std::robotics::PlannerRequest" => align_up(16 * joints + 3 * 56 + 4, 8)?,
            "std::robotics::PlannerWaypoint" => waypoint,
            "std::robotics::PlannerSegment" => 64 * waypoint,
            "std::robotics::PlannerTrajectory" => 16 + 256 * waypoint,
            _ => return None,
        };
        return Some((size, 8));
    }
    let scalar = || c_layout(args.first()?, depth + 1);
    let array = |count: u64| {
        let (size, align) = scalar()?;
        Some((size.checked_mul(count)?, align))
    };
    let count = || {
        args.last()?
            .parse::<u64>()
            .ok()
            .filter(|n| (1..=64).contains(n))
    };
    match head {
        "Quantity" => scalar(),
        "FixedArray" => array(count()?),
        "std::robotics::Point" | "std::robotics::Vector3" => array(3),
        "std::robotics::Quaternion" => array(4),
        "std::robotics::Pose" => array(7),
        "std::robotics::Twist" | "std::robotics::Wrench" => array(6),
        "std::robotics::JointPositions"
        | "std::robotics::JointVelocities"
        | "std::robotics::JointEfforts" => array(count()?),
        "std::robotics::JointState" => {
            let n = count()?;
            let (size, align) = scalar()?;
            let names = align_up(n.checked_mul(4)?, align)?;
            Some((
                names.checked_add(n.checked_mul(size)?.checked_mul(3)?)?,
                align.max(4),
            ))
        }
        "std::robotics::JointWaypoint" => {
            let n = count()?;
            let (size, align) = scalar()?;
            let values = align_up(n.checked_mul(size)?.checked_mul(2)?, 8)?;
            Some((values.checked_add(8)?, align.max(8)))
        }
        "std::robotics::JointTrajectory" => {
            let n = count()?;
            let scalar_name = args.first()?;
            let waypoint = format!("std::robotics::JointWaypoint<{scalar_name},{n}>");
            let (size, align) = c_layout(&waypoint, depth + 1)?;
            Some((
                align_up(size.checked_mul(64)?.checked_add(4)?, align.max(4))?,
                align.max(4),
            ))
        }
        "std::robotics::Distortion" => {
            let (size, align) = scalar()?;
            Some((
                align_up(
                    align_up(4, align)?.checked_add(size.checked_mul(5)?)?,
                    align.max(4),
                )?,
                align.max(4),
            ))
        }
        "std::robotics::CameraIntrinsics" => {
            let (size, align) = scalar()?;
            let distortion_name = format!("std::robotics::Distortion<{}>", args.first()?);
            let (distortion, _) = c_layout(&distortion_name, depth + 1)?;
            let prefix = align_up(8, align)?.checked_add(size.checked_mul(4)?)?;
            Some((
                align_up(prefix.checked_add(distortion)?, align.max(4))?,
                align.max(4),
            ))
        }
        "std::robotics::CameraExtrinsics" => array(7),
        "std::robotics::Stamped" => {
            let (size, align) = scalar()?;
            let offset = align_up(12, align)?;
            Some((
                align_up(offset.checked_add(size)?, align.max(8))?,
                align.max(8),
            ))
        }
        _ => None,
    }
}

/// C ABI byte size for the fixed robotics layouts in layout-header-v1.h.in.
pub fn c_layout_size(ty: &str) -> Option<u64> {
    c_layout(ty, 0).map(|(size, _)| size)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar {
    I32,
    U32,
    F32,
    F64,
}

pub fn c_header(scalar: Scalar, joint_count: u32) -> Result<String, &'static str> {
    if !(1..=64).contains(&joint_count) {
        return Err("joint count must be 1..64");
    }
    let c_scalar = match scalar {
        Scalar::I32 => "int32_t",
        Scalar::U32 => "uint32_t",
        Scalar::F32 => "float",
        Scalar::F64 => "double",
    };
    Ok(C_HEADER
        .replace("@SCALAR@", c_scalar)
        .replace("@JOINTS@", &joint_count.to_string()))
}

pub fn ros_mapping() -> Result<serde_json::Value, serde_json::Error> {
    serde_json::from_str(ROS_MAPPING_JSON)
}

fn pose_parts(name: &str) -> Option<[&str; 3]> {
    let inside = name
        .strip_prefix("std::robotics::Pose<")?
        .strip_suffix('>')?;
    let mut depth = 0usize;
    let mut splits = Vec::new();
    for (index, byte) in inside.bytes().enumerate() {
        match byte {
            b'<' => depth = depth.checked_add(1)?,
            b'>' => depth = depth.checked_sub(1)?,
            b',' if depth == 0 => splits.push(index),
            _ => {}
        }
    }
    if depth != 0 || splits.len() != 2 {
        return None;
    }
    let parts = [
        &inside[..splits[0]],
        &inside[splits[0] + 1..splits[1]],
        &inside[splits[1] + 1..],
    ];
    parts.iter().all(|part| !part.is_empty()).then_some(parts)
}

/// Type-level pose composition for checked system port declarations.
pub fn compose_pose_type(left: &str, right: &str) -> Result<String, &'static str> {
    let [left_scalar, parent, middle] = pose_parts(left).ok_or("TYPE_MISMATCH")?;
    let [right_scalar, source, child] = pose_parts(right).ok_or("TYPE_MISMATCH")?;
    if left_scalar != right_scalar || middle != source {
        return Err("FRAME_MISMATCH");
    }
    Ok(format!(
        "std::robotics::Pose<{left_scalar},{parent},{child}>"
    ))
}

pub struct RosJointState<'a> {
    pub names: &'a [String],
    pub positions: &'a [f64],
    pub velocities: &'a [f64],
    pub efforts: &'a [f64],
}

#[derive(Debug, PartialEq)]
pub struct JointStateWire {
    /// Indices into the declared robot-model joint-name table.
    pub name_ids: Vec<u32>,
    pub positions: Vec<f64>,
    pub velocities: Vec<f64>,
    pub efforts: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JointStateConversionError(pub &'static str);

impl JointStateConversionError {
    pub fn code(self) -> &'static str {
        "S1701"
    }

    pub fn diagnostic(self) -> serde_json::Value {
        crate::diagnostics::document(
            "tool_error",
            vec![crate::diagnostics::diagnostic(
                self.code(),
                self.0,
                "<ros-joint-state>",
                b"",
                None,
            )],
        )
    }
}

/// Validate a caller-supplied ROS JointState before copying any values.
pub fn from_ros_joint_state(
    expected_names: &[&str],
    input: RosJointState<'_>,
) -> Result<JointStateWire, JointStateConversionError> {
    let count = expected_names.len();
    if !(1..=64).contains(&count) || expected_names.iter().any(|name| name.is_empty()) {
        return Err(JointStateConversionError("invalid declared joint names"));
    }
    let mut unique = std::collections::HashSet::new();
    if expected_names.iter().any(|name| !unique.insert(*name)) {
        return Err(JointStateConversionError("duplicate declared joint name"));
    }
    if input.names.len() != count
        || input.positions.len() != count
        || input.velocities.len() != count
        || input.efforts.len() != count
    {
        return Err(JointStateConversionError(
            "joint state field count does not match declaration",
        ));
    }
    if input
        .names
        .iter()
        .zip(expected_names)
        .any(|(actual, expected)| actual != expected)
    {
        return Err(JointStateConversionError(
            "joint state names differ from declared order",
        ));
    }
    if input
        .positions
        .iter()
        .chain(input.velocities)
        .chain(input.efforts)
        .any(|value| !value.is_finite())
    {
        return Err(JointStateConversionError(
            "joint state contains a non-finite value",
        ));
    }
    Ok(JointStateWire {
        name_ids: (0..count as u32).collect(),
        positions: input.positions.to_vec(),
        velocities: input.velocities.to_vec(),
        efforts: input.efforts.to_vec(),
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Distortion {
    None,
    PlumbBob([String; 5]),
    Equidistant([String; 4]),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CameraIntrinsics {
    pub scalar: String,
    pub width_px: String,
    pub height_px: String,
    pub fx_px: String,
    pub fy_px: String,
    pub cx_px: String,
    pub cy_px: String,
    pub distortion: Distortion,
}

fn canonical_decimal(value: &str) -> bool {
    if value.is_empty() || value.len() > 64 || value == "-0" {
        return false;
    }
    let unsigned = value.strip_prefix('-').unwrap_or(value);
    let (whole, fraction) = unsigned
        .split_once('.')
        .map_or((unsigned, None), |(whole, fraction)| {
            (whole, Some(fraction))
        });
    if whole.is_empty()
        || (whole.len() > 1 && whole.starts_with('0'))
        || !whole.bytes().all(|byte| byte.is_ascii_digit())
    {
        return false;
    }
    match fraction {
        Some(fraction) => {
            !fraction.is_empty()
                && fraction.as_bytes().last().is_some_and(|byte| *byte != b'0')
                && fraction.bytes().all(|byte| byte.is_ascii_digit())
        }
        None => true,
    }
}

impl CameraIntrinsics {
    pub fn to_ir(&self) -> Result<serde_json::Value, &'static str> {
        if !matches!(self.scalar.as_str(), "i32" | "u32" | "f32" | "f64")
            || [
                &self.width_px,
                &self.height_px,
                &self.fx_px,
                &self.fy_px,
                &self.cx_px,
                &self.cy_px,
            ]
            .iter()
            .any(|value| !canonical_decimal(value))
        {
            return Err("invalid camera scalar or decimal");
        }
        if self
            .width_px
            .parse::<u32>()
            .ok()
            .filter(|value| *value > 0)
            .is_none()
            || self
                .height_px
                .parse::<u32>()
                .ok()
                .filter(|value| *value > 0)
                .is_none()
        {
            return Err("camera width and height must be positive u32 pixels");
        }
        let (model, coefficients): (&str, Vec<&str>) = match &self.distortion {
            Distortion::None => ("none", Vec::new()),
            Distortion::PlumbBob(values) => {
                ("plumb_bob", values.iter().map(String::as_str).collect())
            }
            Distortion::Equidistant(values) => {
                ("equidistant", values.iter().map(String::as_str).collect())
            }
        };
        if coefficients.iter().any(|value| !canonical_decimal(value)) {
            return Err("invalid distortion decimal");
        }
        Ok(serde_json::json!({
            "schema": "invar-camera-intrinsics-v1",
            "scalar": self.scalar,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "fx_px": self.fx_px,
            "fy_px": self.fy_px,
            "cx_px": self.cx_px,
            "cy_px": self.cy_px,
            "distortion": { "model": model, "coefficients": coefficients }
        }))
    }

    pub fn from_ir(value: &serde_json::Value) -> Result<Self, &'static str> {
        let object = value.as_object().ok_or("camera IR must be an object")?;
        if object.len() != 9 || value["schema"] != "invar-camera-intrinsics-v1" {
            return Err("invalid camera IR schema or fields");
        }
        let get = |name: &str| {
            value[name]
                .as_str()
                .map(str::to_owned)
                .ok_or("invalid camera IR field")
        };
        let distortion = value["distortion"]
            .as_object()
            .ok_or("invalid distortion IR")?;
        if distortion.len() != 2 {
            return Err("invalid distortion IR fields");
        }
        let coefficients = distortion["coefficients"]
            .as_array()
            .ok_or("invalid distortion coefficients")?;
        if coefficients.len() > 5 {
            return Err("distortion coefficient limit exceeded");
        }
        let values = coefficients
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_owned)
                    .ok_or("invalid distortion coefficient")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let distortion = match (distortion["model"].as_str(), values.len()) {
            (Some("none"), 0) => Distortion::None,
            (Some("plumb_bob"), 5) => {
                Distortion::PlumbBob(values.try_into().map_err(|_| "invalid distortion count")?)
            }
            (Some("equidistant"), 4) => {
                Distortion::Equidistant(values.try_into().map_err(|_| "invalid distortion count")?)
            }
            _ => return Err("invalid distortion model or count"),
        };
        let camera = Self {
            scalar: get("scalar")?,
            width_px: get("width_px")?,
            height_px: get("height_px")?,
            fx_px: get("fx_px")?,
            fy_px: get("fy_px")?,
            cx_px: get("cx_px")?,
            cy_px: get("cy_px")?,
            distortion,
        };
        if camera.to_ir()? != *value {
            return Err("noncanonical camera IR");
        }
        Ok(camera)
    }
}
