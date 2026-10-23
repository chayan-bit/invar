//! Validate exported benchmark rows before comparing their medians.
use serde_json::Value;
use std::{collections::BTreeMap, error::Error, fs, io, path::Path};
fn valid_load_receipts(row: &Value, samples: usize) -> bool {
    ["load_1m_before", "load_1m_after"].iter().all(|field| {
        row[*field].as_array().is_some_and(|loads| {
            loads.len() == samples + 1
                && loads.iter().all(|v| {
                    v.as_f64()
                        .is_some_and(|v| v.is_finite() && (0.0..8.0).contains(&v))
                })
        })
    })
}
fn main() -> Result<(), Box<dyn Error>> {
    let directory = std::env::args()
        .nth(1)
        .ok_or("usage: sort-report DIRECTORY")?;
    let mut rows = BTreeMap::new();
    for file in [
        "cpu.jsonl",
        "cuda.jsonl",
        "cub.jsonl",
        "metal.jsonl",
        "arrowmetal.jsonl",
    ] {
        let path = Path::new(&directory).join(file);
        if fs::metadata(&path)?.len() > 1 << 20 {
            return Err(io::Error::other("oversized benchmark input").into());
        }
        for line in fs::read_to_string(path)?.lines() {
            let v: Value = serde_json::from_str(line)?;
            let power = v["power"].as_u64().ok_or("power")?;
            let backend = v["backend"].as_str().ok_or("backend")?.to_owned();
            let times = v["timed_ms"].as_array().ok_or("timed_ms")?;
            if !valid_load_receipts(&v, times.len()) {
                return Err(
                    "missing or inadmissible host load receipt; every sample must remain under 8"
                        .into(),
                );
            }
            if v["calibration_id"] != "sort.sort_by_key.u32" {
                return Err("invalid sort calibration family".into());
            }
            if v["schema"] != "invar-sort-benchmark-v2"
                || !(16..=26).contains(&power)
                || v["elements"] != 1u64 << power
                || v["seed"] != (0x5eeda11u64 ^ power)
                || v["samples"] != times.len()
                || times.len() < 5
                || v["warmups"] != 1
                || times
                    .iter()
                    .any(|t| !t.as_f64().is_some_and(|t| t.is_finite() && t > 0.0))
            {
                return Err(io::Error::other("invalid benchmark row").into());
            }
            let mut sorted: Vec<_> = times.iter().map(|t| t.as_f64().unwrap()).collect();
            sorted.sort_by(f64::total_cmp);
            if (v["median_ms"].as_f64().ok_or("median_ms")? - sorted[sorted.len() / 2]).abs() > 1e-5
            {
                return Err(io::Error::other("median disagrees with samples").into());
            }
            if rows.insert((power, backend), v).is_some() {
                return Err("duplicate row".into());
            }
        }
    }
    let names = [
        "rust-sort_by_key",
        "rayon-par_sort",
        "invar-cpu-1",
        "invar-cpu-4",
        "cuda",
        "nvcc-cub-12.9.27",
        "metal",
        "standalone-arrowmetal",
    ];
    if rows.len() != names.len() * 11 {
        return Err("incomplete benchmark matrix".into());
    }
    for power in 16..=26 {
        let mut hash = None;
        for name in names {
            let row = rows
                .get(&(power, name.to_owned()))
                .ok_or("missing backend/size")?;
            let current = row["output_sha256"].as_str().ok_or("output_sha256")?;
            if current.len() != 64 || !current.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("invalid digest".into());
            }
            if let Some(hash) = hash {
                if hash != current {
                    return Err("backend output mismatch".into());
                }
            } else {
                hash = Some(current);
            }
        }
    }
    let schema: Value = serde_json::from_str(include_str!("../../schema/explain-v5.schema.json"))?;
    let parent = Path::new(&directory).parent().ok_or("benchmark parent")?;
    let identity = format!(
        "fnv1a64:{:016x}",
        invar_device::source_hash(include_str!("../../examples/sort.inv"))
    );
    for (engine, implementation) in [
        ("cpu", "cpu-lsd-radix-4x8"),
        ("cuda", "cuda-lsd-radix-8x4"),
        ("metal", "metal-lsd-radix-8x4"),
    ] {
        let path = parent.join(format!("{engine}-demo.json"));
        if fs::metadata(&path)?.len() > 1 << 20 {
            return Err("oversized demo input".into());
        }
        let v: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
        if !jsonschema::is_valid(&schema, &v)
            || v["device_execution"]["result"] != 7
            || v["device_execution"]["choices"].as_array().is_none_or(|c| {
                c.len() != 4 || c.iter().any(|c| c["implementation"] != implementation)
            })
        {
            return Err("demo schema, result or implementation mismatch".into());
        }
        if v["source_identity"] != identity {
            return Err("demo source identity differs from examples/sort.inv".into());
        }
    }
    println!("All 88 rows agree on full key/original-index SHA-256 at every size.\n");
    println!("Median milliseconds; five measured samples after one warmup.\n");
    println!(
        "| power | Rust | Rayon4 | Invar1 | Invar4 | CUDA | CUB | Metal | ArrowMetal |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    );
    for power in 16..=26 {
        print!("| 2^{power}");
        for name in names {
            print!(
                " | {:.3}",
                rows[&(power, name.to_owned())]["median_ms"]
                    .as_f64()
                    .unwrap()
            );
        }
        println!(" |");
    }
    println!(
        "\nInvar/baseline time ratio (lower is better). CPU and each GPU baseline share their respective host.\n"
    );
    println!(
        "| power | CPU1/Rust | CPU4/Rayon4 | CUDA/CUB | Metal/ArrowMetal |\n| --- | --- | --- | --- | --- |"
    );
    for power in 16..=26 {
        let ms = |name: &str| {
            rows[&(power, name.to_owned())]["median_ms"]
                .as_f64()
                .unwrap()
        };
        println!(
            "| 2^{power} | {:.2} | {:.2} | {:.2} | {:.2} |",
            ms(names[2]) / ms(names[0]),
            ms(names[3]) / ms(names[1]),
            ms(names[4]) / ms(names[5]),
            ms(names[6]) / ms(names[7])
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reject_missing_contended_or_misaligned_load_receipts() {
        let valid = serde_json::json!({"load_1m_before":[0.0,7.99],"load_1m_after":[7.5,2.0]});
        assert!(valid_load_receipts(&valid, 1));
        assert!(!valid_load_receipts(&valid, 2));
        assert!(!valid_load_receipts(&serde_json::json!({}), 1));
        for load in [
            serde_json::json!(8.0),
            serde_json::json!(-1.0),
            Value::Null,
            serde_json::json!("7.0"),
        ] {
            for field in ["load_1m_before", "load_1m_after"] {
                let mut bad = valid.clone();
                bad[field][1] = load.clone();
                assert!(!valid_load_receipts(&bad, 1), "{bad}");
            }
        }
    }
}
