#![forbid(unsafe_code)]
#[path = "../../bench/corpus.rs"]
mod corpus;
#[path = "../../bench/hash.rs"]
mod hash;
#[path = "../../bench/heldout.rs"]
#[allow(dead_code)] // The exclusion index serves tuning paths and the tests, not this writer.
mod heldout;
use std::{
    fs::OpenOptions,
    io::{Read, Write},
    path::PathBuf,
};
fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mode = args
        .next()
        .ok_or("usage: invar-heldout --write-proposal NEW_PATH | --check-proposal PATH")?;
    let path = PathBuf::from(args.next().ok_or("missing proposal path")?);
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    if !["--write-proposal", "--check-proposal"].contains(&mode.as_str()) {
        return Err("unknown mode; freeze is unavailable".into());
    }
    let observed = if mode == "--check-proposal" {
        let mut data = Vec::new();
        std::fs::File::open(&path)
            .map_err(|e| e.to_string())?
            .take(4 * 1024 * 1024 + 1)
            .read_to_end(&mut data)
            .map_err(|e| e.to_string())?;
        if data.len() > 4 * 1024 * 1024 {
            return Err("proposal exceeds 4 MiB limit".into());
        }
        let value: serde_json::Value =
            serde_json::from_slice(&data).map_err(|e| format!("proposal JSON: {e}"))?;
        if value["status"] != "proposal-pending-review"
            || value.get("freeze_status").is_some()
            || value["seed"].as_u64() != Some(corpus::SEED)
        {
            return Err("proposal status or seed differs; frozen proposals are refused".into());
        }
        let programs = value["programs"].as_array().ok_or("missing programs")?;
        for p in programs {
            let source = p["source"].as_str().ok_or("missing source")?;
            if p["source_sha256"] != hash::sha256(source.as_bytes())
                || !["held_out", "tuning"].contains(&p["split"].as_str().unwrap_or(""))
            {
                return Err("source hash or split differs from proposal contract".into());
            }
        }
        if value["held_out_sha256"] != corpus::held_out_hash(&value)? {
            return Err("candidate held-out hash differs".into());
        }
        Some(data)
    } else {
        if path.exists() {
            return Err("proposal path exists; preserve the prior artifact".into());
        }
        None
    };
    let proposal = heldout::generate_proposal(corpus::SEED)?;
    let mut bytes = serde_json::to_vec_pretty(&proposal).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    if bytes.len() > 4 * 1024 * 1024 {
        return Err("generated proposal exceeds 4 MiB limit".into());
    }
    if mode == "--write-proposal" {
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| e.to_string())?;
        f.write_all(&bytes).map_err(|e| e.to_string())?;
    } else if observed.as_deref() != Some(bytes.as_slice()) {
        return Err("proposal differs from seeded generator; status, source, split, shape or hash was changed".into());
    }
    println!("status: {}", proposal["status"].as_str().unwrap());
    println!("programs: {}", proposal["program_counts"]);
    for family in heldout::FAMILIES {
        println!("{family}: {}", proposal["family_counts"][family]);
    }
    println!(
        "candidate held-out SHA-256: {}",
        proposal["held_out_sha256"].as_str().unwrap()
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
