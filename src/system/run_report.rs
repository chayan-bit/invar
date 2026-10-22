//! Reports observations without turning partial runtime evidence into acceptance.

use std::collections::BTreeSet;
use std::fmt::Write;

use serde_json::Value;

use super::SystemIr;

fn hash(value: &Value) -> Result<&str, &'static str> {
    value
        .as_str()
        .filter(|s| {
            s.len() == 64
                && s.bytes()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        })
        .ok_or("invalid evidence SHA-256")
}

fn number(value: &Value) -> Result<f64, &'static str> {
    value
        .as_f64()
        .filter(|x| x.is_finite())
        .ok_or("missing finite run metric")
}

fn pair(value: &Value) -> Result<(f64, f64), &'static str> {
    let a = value
        .as_array()
        .filter(|a| a.len() == 2)
        .ok_or("invalid pose error")?;
    let pair = (number(&a[0])?, number(&a[1])?);
    if pair.0 < 0.0 || pair.1 < 0.0 {
        return Err("negative pose error");
    }
    Ok(pair)
}

fn label(value: &Value) -> Result<&str, &'static str> {
    value
        .as_str()
        .filter(|s| {
            !s.is_empty()
                && s.len() <= 128
                && s.bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
        })
        .ok_or("invalid evidence label")
}

fn hash_state(value: &Value, count: usize) -> Result<String, &'static str> {
    let bits = value
        .as_array()
        .filter(|a| a.len() == count)
        .ok_or("invalid final state bit vector")?;
    let mut bytes = Vec::new();
    for field in bits {
        let bits = field.as_u64().ok_or("invalid final state bits")?;
        if !f64::from_bits(bits).is_finite() {
            return Err("nonfinite final state");
        }
        bytes.extend_from_slice(&bits.to_le_bytes());
    }
    Ok(super::robot::hash_bytes(&bytes))
}

impl SystemIr {
    /// Append bounded, source-bound MCAP observations to the ordinary graph report.
    /// The Boolean is acceptance, independent of successful report generation.
    pub fn report_with_run_evidence(&self, bytes: &[u8]) -> Result<(String, bool), &'static str> {
        let mut output = self.report()?;
        if bytes.len() > 1_048_576 {
            return Err("run evidence size limit exceeded");
        }
        let evidence: Value =
            serde_json::from_slice(bytes).map_err(|_| "invalid run evidence JSON")?;
        let source_hash = super::robot::hash_bytes(self.canonical_json().as_bytes());
        if evidence["schema"] != "invar-s106-run-evidence-v1"
            || hash(&evidence["source_ir_sha256"])? != source_hash
        {
            return Err("run evidence does not match checked system IR");
        }
        let arm = label(&evidence["arm"])?;
        if !matches!(arm, "panda" | "ur5e") {
            return Err("unsupported evidence arm");
        }
        let robot = if arm == "panda" {
            "menagerie::franka_emika_panda"
        } else {
            "menagerie::universal_robots_ur5e"
        };
        if !self.document["products"]
            .as_array()
            .is_some_and(|products| products.iter().any(|p| p["robot"] == robot))
        {
            return Err("run evidence arm differs from checked product");
        }
        let joints = if arm == "panda" { 7 } else { 6 };
        for key in [
            "binding_sha256",
            "scene_sha256",
            "model_sha256",
            "build_log_sha256",
        ] {
            hash(&evidence[key])?;
        }
        let build_summary = evidence["build_packages_finished"]
            .as_u64()
            .ok_or("missing build result")?;
        let build_ok = build_summary == 6;
        writeln!(output, "\n## Recorded S1-06 run evidence: {arm}\n").unwrap();
        writeln!(output, "Source IR SHA-256: `{source_hash}`.\n").unwrap();
        writeln!(
            output,
            "Generated ROS build: {build_summary} packages finished; log SHA-256 `{}`.\n",
            hash(&evidence["build_log_sha256"])?
        )
        .unwrap();
        output.push_str("Foreign-component contract evidence was checked for the current manifests by source compilation.\n\n");
        for component in self.document["extern_components"]
            .as_array()
            .into_iter()
            .flatten()
        {
            writeln!(output, "- Component: `{}`", label(&component["name"])?).unwrap();
        }
        for key in ["pinocchio", "mujoco", "numpy"] {
            writeln!(output, "- {key}: {}", label(&evidence["versions"][key])?).unwrap();
        }
        output.push_str("\n| Run | Seed | Launch | Complete (s) | Tool (mm / deg) | FK cross-check (m) | Cube (mm / deg) | Home (rad) | Recorded checks |\n|---|---|---|---|---|---|---|---|---|\n");
        let runs = evidence["runs"]
            .as_array()
            .filter(|a| (2..=16).contains(&a.len()))
            .ok_or("expected 2 to 16 runs")?;
        let mut identities = BTreeSet::new();
        let mut states = BTreeSet::new();
        let mut full_states = BTreeSet::new();
        let mut full_state_count = 0;
        let mut seeds = BTreeSet::new();
        let mut all_pass = build_ok && runs.len() >= 3;
        let mut provenance = String::from("\n### Recording identities\n\n");
        let mut waypoint_rows = String::from(
            "\n### Independent recorded waypoint checks\n\n| Run | Move | Pinocchio tool error (mm / deg) | Pinocchio vs recorded MuJoCo (m) | Pinocchio vs forwarded MuJoCo (m) |\n|---|---|---|---|---|\n",
        );
        for run in runs {
            let name = label(&run["run"])?;
            let seed = run["seed"].as_u64().ok_or("missing run seed")?;
            let launch_exit = run["launch_exit"]
                .as_u64()
                .filter(|n| *n <= 255)
                .ok_or("invalid launch exit")?;
            let snapshots = run["snapshots"]
                .as_u64()
                .filter(|n| (7..=200_000).contains(n))
                .ok_or("invalid snapshot count")?;
            if run["topic_counts"]["/invar/cell_snapshot"].as_u64() != Some(snapshots)
                || run["topic_counts"]["/sequencer/request"] != 7
                || run["topic_counts"]["/sequencer/approved"] != 7
                || run["topic_counts"]["/sequencer/phase"] != 24
            {
                return Err("incomplete recorded task topics");
            }
            seeds.insert(seed);
            let mcap = hash(&run["mcap_sha256"])?;
            if !identities.insert(mcap)
                || run["mcap_bytes"]
                    .as_u64()
                    .filter(|n| *n > 0 && *n <= 128 * 1024 * 1024)
                    .is_none()
            {
                return Err("duplicate or unbounded MCAP");
            }
            hash(&run["launch_log_sha256"])?;
            let complete = run["complete_sim_ns"]
                .as_u64()
                .ok_or("missing completion time")?;
            let waypoints = run["waypoints"]
                .as_array()
                .filter(|a| a.len() == 7)
                .ok_or("expected seven recorded waypoints")?;
            let mut position: f64 = 0.0;
            let mut orientation: f64 = 0.0;
            let mut fk: f64 = 0.0;
            let mut previous = 0;
            for (waypoint, name) in waypoints.iter().zip([
                "pregrasp",
                "descend",
                "lift",
                "transfer",
                "place_descend",
                "retreat",
                "home",
            ]) {
                if waypoint["move"] != name {
                    return Err("waypoint order mismatch");
                }
                let ns = waypoint["simulation_ns"]
                    .as_u64()
                    .ok_or("missing waypoint time")?;
                if ns <= previous || ns > complete {
                    return Err("invalid waypoint time order");
                }
                previous = ns;
                let (p, r) = pair(&waypoint["tool_error_m_rad"])?;
                position = position.max(p);
                orientation = orientation.max(r);
                for key in ["pinocchio_vs_recorded_m_rad", "pinocchio_vs_mujoco_m_rad"] {
                    let (p, _) = pair(&waypoint[key])?;
                    fk = fk.max(p);
                }
                writeln!(
                    waypoint_rows,
                    "| {} | {name} | {:.12} / {:.12} | {:e} | {:e} |",
                    label(&run["run"])?,
                    p * 1e3,
                    r.to_degrees(),
                    pair(&waypoint["pinocchio_vs_recorded_m_rad"])?.0,
                    pair(&waypoint["pinocchio_vs_mujoco_m_rad"])?.0
                )
                .unwrap();
            }
            let (cube_p, cube_r) = pair(&run["final_cube_error_m_rad"])?;
            let home = number(&run["home_error_rad"])?;
            if home < 0.0 {
                return Err("negative joint error");
            }
            let contacts = run["forbidden_robot_contacts"]
                .as_u64()
                .ok_or("missing contact count")?;
            let limit = number(&run["max_joint_or_velocity_limit_excess"])?;
            let state_hash = hash_state(
                &run["final_recorded_state_bits"],
                if joints == 7 { 47 } else { 39 },
            )?;
            if hash(&run["final_recorded_state_sha256"])? != state_hash {
                return Err("final state digest mismatch");
            }
            states.insert(state_hash);
            if let Some(bits) = run.get("mujoco_integration_state_bits") {
                let size = evidence["mujoco_integration_state_size"]
                    .as_u64()
                    .filter(|n| (1..=4096).contains(n))
                    .ok_or("invalid full integration state size")?;
                if evidence["mujoco_state_spec"] != "mjSTATE_INTEGRATION" {
                    return Err("unsupported full integration state specification");
                }
                let full_hash = hash_state(bits, size as usize)?;
                let time = bits
                    .get(0)
                    .and_then(Value::as_u64)
                    .map(f64::from_bits)
                    .ok_or("missing integration state time")?;
                if !(0.0..18446744073.0).contains(&time) || (time * 1e9) as u64 != complete {
                    return Err("integration state completion time mismatch");
                }
                writeln!(
                    provenance,
                    "- {name}: full integration state `{full_hash}`."
                )
                .unwrap();
                full_states.insert(full_hash);
                full_state_count += 1;
            }
            let passed = launch_exit == 0
                && complete <= 60_000_000_000
                && position < 0.002
                && orientation < std::f64::consts::PI / 180.0
                && fk <= 1e-6
                && cube_p <= 0.005
                && cube_r <= std::f64::consts::PI / 90.0
                && home <= 1e-6
                && contacts == 0
                && limit <= 0.0
                && run["final_attached"] == false;
            all_pass &= passed;
            writeln!(output, "| {name} | {seed} | {} | {:.9} | {:.12} / {:.12} | {fk:e} | {:.12} / {:.12} | {home:e} | {} |",
                run["launch_exit"], complete as f64 / 1e9, position * 1e3, orientation.to_degrees(), cube_p * 1e3, cube_r.to_degrees(), if passed {"pass"} else {"fail"}).unwrap();
            writeln!(
                provenance,
                "- {name}: MCAP `{mcap}`, {} bytes; final recorded state `{}`; launch log `{}`.",
                run["mcap_bytes"],
                hash(&run["final_recorded_state_sha256"])?,
                hash(&run["launch_log_sha256"])?
            )
            .unwrap();
        }
        let same = states.len() == 1 && seeds.len() == 1;
        // CellSnapshot omits cube velocity, warmstart and other integration state.
        // Equal snapshots cannot establish full-state MuJoCo determinism.
        writeln!(
            output,
            "\nRecorded completion-state subset: {} (simulation time excluded).\n",
            if same {
                "bit-exact for one seed"
            } else {
                "NOT bit-exact for one seed"
            }
        )
        .unwrap();
        if full_state_count == 0 {
            output.push_str("Full MuJoCo integration state is absent.\n\n");
        } else {
            writeln!(output, "Full MuJoCo integration state: {full_state_count}/{} runs, {} unique state hashes.\n", runs.len(), full_states.len()).unwrap();
        }
        // v0.1 repeats use the declared per-run tool and joint completion gates.
        // Exact integration-state replay requires step-scheduled inputs (v0.2).
        let accepted = all_pass && seeds.len() == 1;
        writeln!(
            output,
            "Recorded task/FK/limit/contact checks (tool <2 mm / <1 degree; home <=1e-6 rad): {}.\n\nS1-06 acceptance: {}.\n",
            if all_pass { "pass" } else { "fail" },
            if accepted {
                "pass"
            } else {
                "INCOMPLETE. Three same-seed runs within the declared gates are required"
            }
        )
        .unwrap();
        output.push_str("These are offline measurements of archived MCAPs, not new runs or a fresh qualification of the report revision.\n");
        output.push_str(&waypoint_rows);
        output.push_str(&provenance);
        if let Some(failures) = evidence.get("runtime_failures") {
            let failures = failures
                .as_array()
                .filter(|a| a.len() <= 16)
                .ok_or("runtime failure record limit")?;
            if !failures.is_empty() {
                output.push_str("\n### Recorded runtime failures\n\n");
            }
            for failure in failures {
                let name = label(&failure["run"])?;
                let code = label(&failure["code"])?;
                let exit = failure["launch_exit"]
                    .as_u64()
                    .filter(|n| *n > 0 && *n <= 255)
                    .ok_or("runtime failure needs a nonzero exit")?;
                if code != "S1406"
                    || failure["mcap_bytes"]
                        .as_u64()
                        .filter(|n| *n > 0 && *n <= 128 * 1024 * 1024)
                        .is_none()
                {
                    return Err("invalid planner failure evidence");
                }
                writeln!(output, "- {name}: planner killed; typed {code}, launch exit {exit}; finalized MCAP `{}` ({} bytes), launch log `{}`. This is a failure-path check, not a successful task run.", hash(&failure["mcap_sha256"])?, failure["mcap_bytes"], hash(&failure["launch_log_sha256"])?).unwrap();
            }
        }
        Ok((output, accepted))
    }
}
