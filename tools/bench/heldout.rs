//! Reviewed later: candidate oracle programs and separate performance recipes.
use super::{corpus, hash::sha256};
use invar::{check, contraction_source, eval, front, frontier_source, mir, mir_eval};
use serde_json::{Value, json};

pub const FAMILIES: [&str; 10] = [
    "map",
    "reduction",
    "scan",
    "compaction",
    "transpose",
    "stencil",
    "matrix",
    "segmented",
    "gather_scatter",
    "frontier",
];
pub const EXTRA_PER_FAMILY: usize = 32;
const FUEL: u64 = 20_000_000;

/// Forms that became available after the all-family held-out generators (pair
/// reduction, segmented map, Tensor Core source contract). They use their own PRNG stream so
/// the 560 earlier programs keep their bytes and order.
pub const EXTRA_FORM_COUNT: usize = 16;
const FORM_STREAM: u64 = 0x666f_726d_7332_3032;

fn u32_array(xs: &[u32]) -> String {
    format!(
        "[{}]",
        xs.iter()
            .map(|x| format!("{x}u32"))
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn f32_array(xs: &[f32]) -> String {
    format!(
        "[{}]",
        xs.iter()
            .map(|x| format!("{x:.1}"))
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn checks(expected: &str, ty: &str) -> String {
    // Finite nonzero equality is bit exact. Reciprocals distinguish +0/-0,
    // which ordinary source equality deliberately treats as equal.
    let zero = if ty == "f32" {
        "if wanted[q]==0.0 {if 1.0/answer[q]!=1.0/wanted[q]{return 1;}}"
    } else {
        ""
    };
    format!(
        "let wanted: Array<{ty}> = {expected}; if len(answer) != len(wanted) {{ return 1; }} for q in 0u32..len(wanted) {{ if answer[q] != wanted[q] {{ return 1; }} {zero} }} return 0;"
    )
}

fn finish(leaf: &str, setup: &str, expected: &[u32]) -> String {
    format!(
        "{leaf} fn main()->i32 {{ {setup} {} }}",
        checks(&u32_array(expected), "u32")
    )
}
fn finish_float(leaf: &str, setup: &str, expected: &[f32]) -> String {
    format!(
        "{leaf} fn main()->i32 {{ {setup} {} }}",
        checks(&f32_array(expected), "f32")
    )
}
fn shuffle(xs: &mut [u32], state: &mut u64) {
    for i in (1..xs.len()).rev() {
        xs.swap(i, corpus::next(state) as usize % (i + 1));
    }
}
fn shape_proposals() -> Value {
    json!({
        "status":"proposal-pending-review",
        "scope":"CPU oracle invocations and performance recipes are separate; no large source invocation or GPU performance acceptance is claimed",
        "map":{"exact_elements":[0,1,31,32,33,255,256,257,4093],"powers_of_two_exponents":[10,12,14,16,18,20,22,24,26,28],"budget_rule":"retain every requested shape; resource refusal is explicit, never remove a failed case"},
        "reduction":{"empty":0,"singleton":1,"odd":33,"ragged_block":257,"multi_block":4093,"pair_law":"ordered affine composition, wrapping u32"},
        "scan":{"boundary_elements":[0,1,31,32,33,255,256,257,4093],"long_sequence":65537,"nonidentity_seed":7},
        "compaction":{"exact_selectivity_percent":[0,1,50,99,100],"elements":10000},
        "transpose":{"exact_dimensions":[[31,33],[127,257],[1024,1024]],"larger_bounded_cases":[[2048,1025]],"strided_read_view":{"dimensions":[31,33],"stride":2}},
        "stencil":{"tiny_domains":[[0,0],[1,1],[1,3]],"halo_boundaries":[[2,2],[3,5]],"odd_widths":[[31,33]],"wide_tall":[[3,257],[257,3]],"larger_square":[[1024,1024]],"ping_pong_steps":[1,2,3,16]},
        "matrix":{"exact_square_sizes_where_budget_permits":[256,1024,4096],"rectangular":[[31,33,17]],"one_sized_dimensions":[[1,1,1],[1,33,17],[31,1,17],[31,33,1]],"zero_reduction_extent":[[31,0,17]],"odd_dimensions":[[31,33,17]],"policies":["strict ordered binary32","explicit Tensor Core tc-f16-f32-v1 (D5 as amended 2026-10-14)"]},
        "segmented":{"empty_segments":[0,0,0],"uniform_lengths":[4,4,4,4],"skewed_lengths":[0,1,2,13],"one_dominant_segment":[0,0,15,1],"ragged_totals":[3,5,9]},
        "gather_scatter":{"elements":[0,1,31,33,257,4093],"index_classes":["sorted","random","duplicates","out_of_bounds rejection"],"conflict_policy":"injective scatter; wrapping modular sum histogram"},
        "frontier":{"chains":[1,5,16],"stars":[5,17],"balanced_small_graphs":[7,15],"skewed_small_graphs":[8,17],"bounded_sparse_larger_cases":[257],"rounds":"vertex count","observations":"reachability set and per-level masks; serial pointer chasing is a separate negative control"}
    })
}

#[path = "heldout/generators.rs"]
mod generators;
use generators::{
    BOUNDARIES, compaction, frontier, indexed, map_boundary, matrix, pair_reduction,
    reduction_boundary, scan_boundary, segmented, segmented_map, stencil, tc_matrix, transpose,
};

#[path = "heldout/validate.rs"]
mod validate;
// Re-exports keep the original public surface; each includer uses a different subset.
#[allow(unused_imports)]
pub use validate::{stored_oracle, validate_source, validate_tc};

fn negative_controls() -> Result<Value, String> {
    let mut controls = Vec::new();
    for mode in 0..3 {
        let store = match mode {
            0 => "out[i]=a[ix[i]];",
            1 => "out[ix[i]]=a[i];",
            _ => "out[ix[i]]=wrapping_add(out[ix[i]],a[i]);",
        };
        let leaf = format!(
            "fn work(a:read Array<u32>,ix:read Array<u32>{})->Array<u32>{{var out=filled_u32({},0u32);for i in 0u32..len(ix){{{store}}}return out;}}",
            if mode == 0 { "" } else { ",n:u32" },
            if mode == 0 { "len(ix)" } else { "n" }
        );
        let source = format!(
            "{leaf} fn main()->i32{{let a:Array<u32> =[1u32];let ix:Array<u32> =[2u32];let answer=work(a,ix{});return 0;}}",
            if mode == 0 { "" } else { ",1u32" }
        );
        let checked = check::check(
            front::parse(source.as_bytes()).map_err(|e| format!("{e:?}"))?,
            true,
        )
        .map_err(|e| format!("{e:?}"))?;
        let refused = if mode == 2 {
            let c = invar::histogram_source::Contract::from_checked(
                source.as_bytes(),
                &checked,
                "work",
            )
            .map_err(|e| format!("{e:?}"))?;
            matches!(
                c.candidate(&[1], &[2], 1),
                Err(invar::histogram_source::Error::Bounds { iteration: 0 })
            )
        } else {
            let c =
                invar::indexed_source::Contract::from_checked(source.as_bytes(), &checked, "work")
                    .map_err(|e| format!("{e:?}"))?;
            matches!(
                c.candidate(&[1], &[2], if mode == 0 { None } else { Some(1) }),
                Err(invar::indexed_source::Error::Bounds { iteration: 0 })
            )
        };
        let reference = eval::observe(&checked, FUEL);
        let m = mir::lower(&checked).map_err(|e| format!("{e:?}"))?;
        let lowered = mir_eval::observe(&m, FUEL);
        if !refused
            || !matches!(
                reference.result,
                Err(eval::EvalError::Trap {
                    category: eval::TrapCategory::IndexOutOfBounds,
                    ..
                })
            )
            || reference.result != lowered.result
            || reference.steps_used != lowered.steps_used
            || reference.cleanup_trace != lowered.cleanup_trace
        {
            return Err("bounds negative control failed".into());
        }
        controls.push(json!({"id":format!("indexed-bounds-{mode}"),"source":source,"source_sha256":sha256(source.as_bytes()),"expected":"INDEX_OUT_OF_BOUNDS","oracle_status":"passed","recognizer_status":"recognized","candidate_status":"bounds-refused"}));
    }
    let source = "fn main()->i32{let links:Array<u32> =[1u32,2u32,3u32,3u32];var index=0u32;for step in 0u32..3u32{index=links[index];}if index!=3u32{return 1;}return 0;}";
    let checked = check::check(
        front::parse(source.as_bytes()).map_err(|e| format!("{e:?}"))?,
        true,
    )
    .map_err(|e| format!("{e:?}"))?;
    let m = mir::lower(&checked).map_err(|e| format!("{e:?}"))?;
    let a = invar::admission::analyze(source.as_bytes(), &m).map_err(|e| format!("{e:?}"))?;
    if eval::observe(&checked, FUEL).result != Ok(eval::Value::I32(0))
        || a.regions().is_empty()
        || !a
            .regions()
            .iter()
            .all(|r| r.plan_kind() == invar::plan::PlanKind::Serial)
    {
        return Err("pointer-chasing negative control was not serial".into());
    }
    controls.push(json!({"id":"frontier-pointer-chasing","source":source,"source_sha256":sha256(source.as_bytes()),"oracle_status":"passed","candidate_status":"serial-fallback","first_rejected_obligations":a.regions().iter().map(|r|r.first_rejected_obligation()).collect::<Vec<_>>()}));
    Ok(json!(controls))
}

pub fn generate(seed: u64) -> Result<Value, String> {
    if seed == 0 {
        return Err("xorshift64 seed must be nonzero".into());
    }
    let legacy = if seed == corpus::SEED {
        corpus::generate()?
    } else {
        corpus::generate_seeded(seed)?
    };
    let mut programs = legacy["programs"]
        .as_array()
        .ok_or("legacy programs missing")?
        .clone();
    for p in &mut programs {
        p["oracle_shape"] = json!({"elements":3});
    }
    let mut state = seed;
    for family in &FAMILIES[..3] {
        for ordinal in 0..EXTRA_PER_FAMILY {
            let n = BOUNDARIES[ordinal % BOUNDARIES.len()];
            let source = match *family {
                "map" => map_boundary(&legacy["programs"][ordinal], n)?,
                "reduction" => reduction_boundary(&mut state, n, ordinal % 2 == 1),
                _ => scan_boundary(&mut state, n, ordinal % 3),
            };
            validate_source(family, &source)
                .map_err(|e| format!("{family}-boundary-{ordinal:03}: {e}"))?;
            programs.push(json!({"id":format!("{family}-boundary-{ordinal:03}"),"family":family,"source":source,"source_sha256":sha256(source.as_bytes()),"oracle_shape":{"elements":n},"oracle_status":"passed","recognizer_status":"recognized","stages":if *family=="map" {legacy["programs"][ordinal]["stages"].clone()}else{json!(1)}}));
        }
    }
    for family in &FAMILIES[3..] {
        for ordinal in 0..EXTRA_PER_FAMILY {
            let (mut source, shape) = match *family {
                "compaction" => compaction(&mut state, ordinal),
                "transpose" => transpose(&mut state, ordinal),
                "stencil" => stencil(&mut state, ordinal),
                "matrix" => matrix(&mut state, ordinal),
                "segmented" => segmented(&mut state, ordinal),
                "gather_scatter" => indexed(&mut state, ordinal),
                "frontier" => frontier(&mut state, ordinal),
                _ => unreachable!(),
            }
            .map_err(|e| format!("{family}-{ordinal:03}: {e}"))?;
            // Names vary without changing semantics or teaching recognizers case IDs.
            let function = format!("kernel_{ordinal}");
            source = source.replace("work", &function);
            validate_source(family, &source).map_err(|e| format!("{family}-{ordinal:03}: {e}"))?;
            programs.push(json!({"id":format!("{family}-{ordinal:03}"),"family":family,"split":if ordinal%4==0{"held_out"}else{"tuning"},"entry":function,"source":source,"source_sha256":sha256(source.as_bytes()),"oracle_shape":shape,"oracle_status":"passed","recognizer_status":"recognized"}));
        }
    }
    let mut form_state = seed ^ FORM_STREAM;
    for (family, form) in [
        ("reduction", "pair"),
        ("segmented", "map"),
        ("matrix", "tc"),
    ] {
        for ordinal in 0..EXTRA_FORM_COUNT {
            let id = format!("{family}-{form}-{ordinal:03}");
            let function = format!("kernel_{form}_{ordinal}");
            let (source, shape) = match form {
                "pair" => pair_reduction(&mut form_state, ordinal),
                "map" => segmented_map(&mut form_state, ordinal),
                _ => {
                    let (tc, twin, mut shape) = tc_matrix(&mut form_state, ordinal);
                    let (tc, twin) = (
                        tc.replace("work", &function),
                        twin.replace("work", &function),
                    );
                    validate_tc(&tc, &twin, &function, &shape).map_err(|e| format!("{id}: {e}"))?;
                    shape["strict_twin_sha256"] = json!(sha256(twin.as_bytes()));
                    (tc, shape)
                }
            };
            let source = source.replace("work", &function);
            if form != "tc" {
                validate_source(family, &source).map_err(|e| format!("{id}: {e}"))?;
            }
            programs.push(json!({"id":id,"family":family,"form":form,"entry":function,"source":source,"source_sha256":sha256(source.as_bytes()),"oracle_shape":shape}));
        }
    }
    // Split depends only on family ordinals and its own PRNG, never results.
    // Pin subtype representatives so parity of an ordinal cannot hide a law
    // or graph topology from the held-out set (the legacy modulo split did).
    let mut split_state = seed ^ 0x6865_6c64_6f75_7431;
    for family in FAMILIES {
        let indices: Vec<usize> = programs
            .iter()
            .enumerate()
            .filter(|(_, p)| p["family"] == family)
            .map(|(i, _)| i)
            .collect();
        // Each later form keeps two representatives (pair: identity and live
        // seed; segmented map: with and without the segment operand; tc).
        let base = EXTRA_PER_FAMILY;
        let required: &[usize] = match family {
            "map" => &[0, 15, 63, 79],
            "reduction" => &[0, 1, 80 + base, 81 + base],
            "scan" => &[0, 1, 2],
            "segmented" => &[0, 1, 2, 3, 4, base, base + 1],
            "compaction" => &[0, 1, 2, 3, 4],
            "matrix" => &[0, 1, 2, 3, base, base + 1],
            "gather_scatter" => &[0, 1, 2],
            _ => &[0, 1, 2, 3],
        };
        // The split unit is a distinct program: identical sources up to the
        // generated entry name (degenerate shapes such as an empty transpose,
        // and a few legacy scans, repeat) always share one split, otherwise a
        // held-out program would also be a tuning program. One quarter of the
        // distinct programs per family is held out.
        let classes: Vec<String> = indices
            .iter()
            .map(|&i| canonical_program(programs[i]["source"].as_str().unwrap_or("")))
            .collect();
        let mut distinct: Vec<&String> = Vec::new();
        for c in &classes {
            if !distinct.contains(&c) {
                distinct.push(c);
            }
        }
        let mut held_classes: Vec<&String> = Vec::new();
        for &o in required {
            if !held_classes.contains(&&classes[o]) {
                held_classes.push(&classes[o]);
            }
        }
        let mut remaining: Vec<u32> = (0..distinct.len() as u32)
            .filter(|&i| !held_classes.contains(&distinct[i as usize]))
            .collect();
        shuffle(&mut remaining, &mut split_state);
        let quota = (distinct.len() / 4).saturating_sub(held_classes.len());
        held_classes.extend(remaining.iter().take(quota).map(|&i| distinct[i as usize]));
        let held: Vec<usize> = (0..indices.len())
            .filter(|&o| held_classes.contains(&&classes[o]))
            .collect();
        // Each distinct program is counted once: its first member in ordinal
        // order carries the split, and later identical members are recorded as
        // duplicates of it, outside both the held-out and the tuning sets, so
        // no degenerate workload is weighted more than once.
        let mut representative: Vec<(&String, String)> = Vec::new();
        for (ordinal, index) in indices.into_iter().enumerate() {
            let class = &classes[ordinal];
            if let Some((_, id)) = representative.iter().find(|(c, _)| *c == class) {
                programs[index]["split"] = json!("duplicate");
                programs[index]["duplicate_of"] = json!(id);
            } else {
                representative.push((
                    class,
                    programs[index]["id"].as_str().unwrap_or("").to_owned(),
                ));
                programs[index]["split"] = json!(if held.contains(&ordinal) {
                    "held_out"
                } else {
                    "tuning"
                });
            }
            programs[index]["oracle_status"] = json!("passed");
            programs[index]["recognizer_status"] = json!("recognized");
        }
    }
    let mut counts = json!({});
    for family in FAMILIES {
        let count = programs.iter().filter(|p| p["family"] == family).count();
        let distinct = |split: Option<&str>| {
            programs
                .iter()
                .filter(|p| p["family"] == family && split.is_none_or(|s| p["split"] == s))
                .map(|p| canonical_program(p["source"].as_str().unwrap_or("")))
                .collect::<std::collections::BTreeSet<_>>()
                .len()
        };
        counts[family] = json!({"generated":count,"valid":count,"recognized":count,"distinct":distinct(None),"held_out_distinct":distinct(Some("held_out")),"pending":0,"ignored":0,"unavailable_form_count":if family=="transpose"{1}else{0},"held_out":programs.iter().filter(|p|p["family"]==family&&p["split"]=="held_out").count()});
    }
    let corpus = json!({"programs":programs});
    let held = corpus::held_out_hash(&corpus)?;
    let negatives = negative_controls()?;
    Ok(
        json!({"schema_version":1,"status":"proposal-pending-review","generator":"xorshift64-v2-all-families-proposal","seed":seed,"held_out_sha256":held,"hash_scope":"compact serde_json serialization of full held_out program entries in corpus order; proposal metadata is excluded","split_policy":{"held_out_ratio_proposal":"1/4 (25%); numeric ratio is not specified by M5-01","rule":"one quarter of the distinct programs per family (identical sources up to the generated entry name form one unit); the first member of a unit carries its split and later members are duplicates outside both sets; predeclared subtype representatives plus a seeded shuffle of the remaining units; independent of oracle results"},"program_counts":{"total":programs.len(),"held_out":programs.iter().filter(|p|p["split"]=="held_out").count(),"tuning":programs.iter().filter(|p|p["split"]=="tuning").count(),"duplicate":programs.iter().filter(|p|p["split"]=="duplicate").count()},"family_counts":counts,"count_scope":"eligible bounded CPU oracle programs; unavailable forms are counted separately","unavailable_forms":[{"family":"transpose","form":"strided read view","status":"unavailable","reason":"the merged transpose source recognizer (M4-02) admits direct row-major transpose only"}],"available_forms":[{"family":"reduction","form":"ordered affine pair composition","source":"pair_reduction::extract plus Execution::PairReduction admission"},{"family":"segmented","form":"per-segment map","source":"segmented_device::MapContract"},{"family":"matrix","form":"explicit Tensor Core tc-f16-f32-v1","source":"contraction_source tc_mul_add; strict twin reference-evaluated, exact-sum reference checked independently"}],"shape_proposals":shape_proposals(),"pending_acceptance":["Review and bind proposed sizes for unsized shape classes","After M4-02..05 merge, rerun source recognizers and CPU oracles on the merged revision","Large performance recipe instantiation, resource admission and numeric adversarial coverage remain unexecuted","Strided-view transpose recognition remains pending family support (M4-02); not approximated by direct transpose","Freeze through reviewed manifest version bump before M4-06; this command cannot freeze"],"gpu_performance":"not_run","negative_controls":negatives,"programs":programs}),
    )
}

/// Store the candidate held-out set; tuning sources remain reproducible from
/// `generate` and do not need a second committed copy.
pub fn generate_proposal(seed: u64) -> Result<Value, String> {
    let mut candidate = generate(seed)?;
    candidate["programs"]
        .as_array_mut()
        .ok_or("missing generated programs")?
        .retain(|p| p["split"] == "held_out");
    candidate["programs_scope"] =
        json!("held_out_only; all tuning programs are regenerated by the same seeded generator");
    Ok(candidate)
}

#[path = "heldout/index.rs"]
mod index;
// Re-exports keep the original public surface; each includer uses a different subset.
#[allow(unused_imports)]
pub use index::{HeldOutIndex, canonical_functions, canonical_program, tuning_programs};
