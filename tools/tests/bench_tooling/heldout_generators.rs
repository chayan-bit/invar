//! Issue #7 (M5-01).
#![forbid(unsafe_code)]
use crate::{corpus, hash, heldout};
use invar::{check, front};
use serde_json::{Value, json};
use std::{fs, process::Command};

#[test]
fn determinism_coverage_and_proposal_contract() {
    let a = heldout::generate(corpus::SEED).unwrap();
    let b = heldout::generate(corpus::SEED).unwrap();
    assert_eq!(
        serde_json::to_vec_pretty(&a).unwrap(),
        serde_json::to_vec_pretty(&b).unwrap()
    );
    assert_eq!(a["status"], "proposal-pending-review");
    assert!(a.get("freeze_status").is_none());
    assert!(a["program_counts"]["total"].as_u64().unwrap() >= 200);
    // Programs that are identical up to the generated entry name share a split:
    // no held-out program may also appear on the tuning side.
    let side = |split: &str| {
        a["programs"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p["split"] == split)
            .map(|p| heldout::canonical_program(p["source"].as_str().unwrap()))
            .collect::<std::collections::BTreeSet<_>>()
    };
    let (held_out, tuning) = (side("held_out"), side("tuning"));
    assert!(held_out.is_disjoint(&tuning));
    // Each distinct program is weighted once: no split holds two identical
    // programs, and every later identical member is a recorded duplicate of the
    // representative that carries the split.
    let programs = a["programs"].as_array().unwrap();
    let counts = &a["program_counts"];
    assert_eq!(held_out.len() as u64, counts["held_out"].as_u64().unwrap());
    assert_eq!(tuning.len() as u64, counts["tuning"].as_u64().unwrap());
    assert!(counts["duplicate"].as_u64().unwrap() > 0);
    assert_eq!(
        counts["held_out"].as_u64().unwrap()
            + counts["tuning"].as_u64().unwrap()
            + counts["duplicate"].as_u64().unwrap(),
        counts["total"].as_u64().unwrap()
    );
    for p in programs.iter().filter(|p| p["split"] == "duplicate") {
        let of = programs
            .iter()
            .find(|q| q["id"] == p["duplicate_of"])
            .unwrap();
        assert!(of["split"] == "held_out" || of["split"] == "tuning");
        assert_eq!(of["family"], p["family"]);
        assert_eq!(
            heldout::canonical_program(of["source"].as_str().unwrap()),
            heldout::canonical_program(p["source"].as_str().unwrap())
        );
    }
    for family in heldout::FAMILIES {
        let c = &a["family_counts"][family];
        let (distinct, held) = (
            c["distinct"].as_u64().unwrap(),
            c["held_out_distinct"].as_u64().unwrap(),
        );
        assert_eq!(held, distinct / 4, "{family}");
        assert_eq!(c["held_out"], c["held_out_distinct"], "{family}");
        assert!(distinct >= 2 && distinct <= c["generated"].as_u64().unwrap());
    }
    let stored = heldout::generate_proposal(corpus::SEED).unwrap();
    assert_eq!(
        stored["programs"].as_array().unwrap().len(),
        a["program_counts"]["held_out"].as_u64().unwrap() as usize
    );
    assert!(
        stored["programs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|p| p["split"] == "held_out")
    );
    assert_eq!(stored["held_out_sha256"], a["held_out_sha256"]);
    assert!(serde_json::to_vec_pretty(&stored).unwrap().len() < 500 * 1024);
    let programs = a["programs"].as_array().unwrap();
    let mut ids = std::collections::BTreeSet::new();
    for p in programs {
        assert!(ids.insert(p["id"].as_str().unwrap()));
        assert_eq!(
            p["source_sha256"],
            hash::sha256(p["source"].as_str().unwrap().as_bytes())
        );
    }
    for family in heldout::FAMILIES {
        let c = &a["family_counts"][family];
        assert!(c["generated"].as_u64().unwrap() > 0);
        assert_eq!(c["generated"], c["valid"]);
        assert_eq!(c["generated"], c["recognized"]);
        assert_eq!(c["ignored"], 0);
        assert_eq!(c["pending"], 0);
        assert!(c["held_out"].as_u64().unwrap() > 0);
    }
    for family in ["map", "reduction", "scan"] {
        for n in [0, 1, 31, 32, 33, 255, 256, 257, 4093] {
            assert!(
                programs
                    .iter()
                    .any(|p| p["family"] == family && p["oracle_shape"]["elements"] == n),
                "{family} n={n}"
            );
        }
    }
    assert_eq!(a["negative_controls"].as_array().unwrap().len(), 4);
    let shapes = &a["shape_proposals"];
    assert_eq!(
        shapes["transpose"]["exact_dimensions"],
        json!([[31, 33], [127, 257], [1024, 1024]])
    );
    assert_eq!(
        shapes["matrix"]["exact_square_sizes_where_budget_permits"],
        json!([256, 1024, 4096])
    );
    assert_eq!(
        shapes["compaction"]["exact_selectivity_percent"],
        json!([0, 1, 50, 99, 100])
    );
    assert_eq!(
        shapes["map"]["exact_elements"],
        json!([0, 1, 31, 32, 33, 255, 256, 257, 4093])
    );
    assert_eq!(a["held_out_sha256"], corpus::held_out_hash(&a).unwrap());
    assert_eq!(
        corpus::generate().unwrap(),
        serde_json::from_slice::<Value>(include_bytes!("../../bench/corpus-v1.json")).unwrap()
    );
}

#[test]
fn seeds_and_real_oracle_failures_are_not_hidden() {
    assert!(heldout::generate(0).is_err());
    let a = heldout::generate(corpus::SEED).unwrap();
    let b = heldout::generate(corpus::SEED + 1).unwrap();
    assert_ne!(a["held_out_sha256"], b["held_out_sha256"]);
    let program = a["programs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["family"] == "transpose" && p["oracle_shape"]["dimensions"] == json!([1, 1]))
        .unwrap();
    let s = program["source"].as_str().unwrap();
    assert!(heldout::validate_source("transpose", &s.replace("return 0;", "return 1;")).is_err());
    assert!(
        heldout::validate_source(
            "transpose",
            &s.replace(
                "b[c*rows+r]=a[r*cols+c]",
                "b[c*rows+r]=wrapping_add(a[r*cols+c],1u32)"
            )
        )
        .is_err()
    );
}

#[test]
fn cli_refuses_overwrite_and_every_frozen_or_altered_proposal() {
    let dir = std::env::temp_dir().join(format!("invar-heldout-test-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let path = dir.join("proposal.json");
    let run = |mode: &str| {
        Command::new(env!("CARGO_BIN_EXE_invar-heldout"))
            .arg(mode)
            .arg(&path)
            .output()
            .unwrap()
    };
    assert!(run("--write-proposal").status.success());
    let original = fs::read(&path).unwrap();
    assert!(run("--check-proposal").status.success());
    assert!(!run("--write-proposal").status.success());
    assert_eq!(original, fs::read(&path).unwrap());
    for mutation in ["status", "freeze_status", "seed", "source", "split", "hash"] {
        let mut value: Value = serde_json::from_slice(&original).unwrap();
        match mutation {
            "status" => value["status"] = json!("frozen"),
            "freeze_status" => value["freeze_status"] = json!("frozen"),
            "seed" => value["seed"] = json!(1),
            "source" => value["programs"][0]["source"] = json!("fn main()->i32{return 0;}"),
            "split" => value["programs"][0]["split"] = json!("other"),
            _ => value["held_out_sha256"] = json!("0".repeat(64)),
        };
        fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        assert!(!run("--check-proposal").status.success(), "{mutation}");
    }
    assert!(!run("--freeze").status.success());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn heldout_exclusion_tuning_paths_never_see_held_out_programs_or_inputs() {
    let all = heldout::generate(corpus::SEED).unwrap();
    let programs = all["programs"].as_array().unwrap();
    let held: Vec<&Value> = programs
        .iter()
        .filter(|p| p["split"] == "held_out")
        .collect();
    let (index, tuning) = heldout::tuning_programs(corpus::SEED).unwrap();
    let exact: std::collections::BTreeSet<&str> =
        held.iter().map(|p| p["source"].as_str().unwrap()).collect();
    assert_eq!(index.len(), exact.len());
    let duplicates = programs
        .iter()
        .filter(|p| p["split"] == "duplicate")
        .count();
    assert_eq!(tuning.len() + held.len() + duplicates, programs.len());
    // A recorded duplicate of a held-out program is refused like the program itself.
    for p in programs.iter().filter(|p| p["split"] == "duplicate") {
        let of = programs
            .iter()
            .find(|q| q["id"] == p["duplicate_of"])
            .unwrap();
        if of["split"] == "held_out" {
            assert!(
                index.admit(p["source"].as_str().unwrap()).is_err(),
                "{}",
                p["id"]
            );
        }
    }
    let held_ids: std::collections::BTreeSet<&str> =
        held.iter().map(|p| p["id"].as_str().unwrap()).collect();
    assert!(
        tuning
            .iter()
            .all(|p| p["split"] == "tuning" && !held_ids.contains(p["id"].as_str().unwrap()))
    );
    let mut data_checked = 0;
    for p in &held {
        let source = p["source"].as_str().unwrap();
        assert!(index.admit(source).is_err(), "{}", p["id"]);
        // Renamed entry and reformatted copies are the same held-out program.
        let renamed = match p["entry"].as_str() {
            Some(entry) => source.replace(entry, "kernel_tuned"),
            None => source.replace("main", "main "),
        };
        assert!(index.admit(&renamed).is_err(), "renamed {}", p["id"]);
        assert!(index.admit(&source.replace(';', " ; ")).is_err());
        // Arbitrary renames, a new wrapping `main`, or extra helpers do not
        // make it a new program.
        let mut renamed_all = source.replace("main", "held_entry");
        if let Some(entry) = p["entry"].as_str() {
            renamed_all = renamed_all.replace(entry, "anything_goes");
        }
        let wrapped = format!(
            "fn helper(x: u32) -> u32 {{ return x; }} {renamed_all} fn main() -> i32 {{ return held_entry(); }}"
        );
        assert!(index.admit(&wrapped).is_err(), "wrapped {}", p["id"]);
        // Its data inputs alone, inside an otherwise new program, are refused too.
        if let Some(start) = source.find("let aa: Array<u32> = [") {
            let array = &source[start + 21..];
            let array = &array[..=array.find(']').unwrap()];
            if array.matches(',').count() >= 7 {
                let probe = format!("fn main()->i32{{let tuned: Array<u32> = {array}; return 0;}}");
                assert!(index.admit(&probe).is_err(), "inputs of {}", p["id"]);
                data_checked += 1;
            }
        }
    }
    assert!(data_checked > 0);
    // A fresh program with fresh data is admitted.
    assert!(index.admit("fn main()->i32{let xs: Array<u32> = [9u32,8u32,7u32,6u32,5u32,4u32,3u32,2u32,1u32]; return 0;}").is_ok());
}

#[test]
fn heldout_forms_added_after_106_are_recognized_and_held_out() {
    let all = heldout::generate(corpus::SEED).unwrap();
    let programs = all["programs"].as_array().unwrap();
    for (family, form) in [
        ("reduction", "pair"),
        ("segmented", "map"),
        ("matrix", "tc"),
    ] {
        let ours: Vec<&Value> = programs
            .iter()
            .filter(|p| p["family"] == family && p["form"] == form)
            .collect();
        assert_eq!(ours.len(), heldout::EXTRA_FORM_COUNT, "{family}/{form}");
        assert!(ours.iter().filter(|p| p["split"] == "held_out").count() >= 2);
    }
    let unavailable = all["unavailable_forms"].as_array().unwrap();
    assert_eq!(unavailable.len(), 1);
    assert_eq!(unavailable[0]["form"], "strided read view");
    // An identity-seeded and a live-seeded pair reduction are both held out.
    let pairs: Vec<&Value> = programs
        .iter()
        .filter(|p| p["form"] == "pair" && p["split"] == "held_out")
        .collect();
    assert!(
        pairs
            .iter()
            .any(|p| p["oracle_shape"]["seed"] == json!([1, 0]))
    );
    assert!(
        pairs
            .iter()
            .any(|p| p["oracle_shape"]["seed"] != json!([1, 0]))
    );
    // A tc_mul_add source is ordinary checked source on main; a wrong oracle fails.
    // The stored tc program runs its own oracle, not only its strict twin.
    for tc in programs.iter().filter(|p| p["form"] == "tc") {
        let source = tc["source"].as_str().unwrap();
        assert!(check::check(front::parse(source.as_bytes()).unwrap(), false).is_ok());
        heldout::stored_oracle(source).unwrap_or_else(|e| panic!("{}: {e}", tc["id"]));
        let at = source.rfind("return 0;").unwrap();
        let wrong = format!("{}return 1;{}", &source[..at], &source[at + 9..]);
        assert!(heldout::stored_oracle(&wrong).is_err(), "{}", tc["id"]);
    }
    let pair = pairs[0]["source"].as_str().unwrap();
    assert!(
        heldout::validate_source("reduction", &pair.replace("return 0;", "return 1;")).is_err()
    );
}
