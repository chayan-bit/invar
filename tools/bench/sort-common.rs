use sha2::{Digest, Sha256};
use std::hint::black_box;
pub fn input(n: usize, seed: u32) -> Vec<u32> {
    let mut x = seed;
    (0..n)
        .map(|i| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            if i % 8 == 0 { x & 255 } else { x }
        })
        .collect()
}
pub fn verify(input: &[u32], keys: &[u32], ids: &[u32]) -> String {
    assert_eq!(keys.len(), input.len());
    assert_eq!(ids.len(), keys.len());
    let mut seen = vec![false; input.len()];
    let mut hash = Sha256::new();
    for (i, (&key, &id)) in keys.iter().zip(ids).enumerate() {
        let id = id as usize;
        assert!(id < input.len() && !seen[id]);
        seen[id] = true;
        assert_eq!(key, input[id]);
        if i > 0 {
            assert!(keys[i - 1] < key || keys[i - 1] == key && (ids[i - 1] as usize) < id);
        }
        hash.update(key.to_le_bytes());
        hash.update((id as u32).to_le_bytes());
    }
    hash.finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
pub fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
// Read the actual execution host, including physical Metal benchmark runs.
fn load_1m() -> f64 {
    load_text()
        .trim_start_matches(['{', ' '])
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .expect("one-minute load")
}
#[cfg(target_os = "linux")]
fn load_text() -> String {
    std::fs::read_to_string("/proc/loadavg").expect("host load receipt")
}
#[cfg(target_os = "macos")]
fn load_text() -> String {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .expect("host load receipt");
    assert!(output.status.success());
    String::from_utf8(output.stdout).expect("host load text")
}
/// Other hosts (Windows) have no load average: the bench builds but refuses to time.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn load_text() -> String {
    panic!("sort timings require a host load receipt (Linux or macOS)")
}
pub fn row(
    backend: &str,
    power: u32,
    samples: usize,
    mut run: impl FnMut(&[u32], &[u32]) -> (Vec<u32>, Vec<u32>, f64, f64),
) -> serde_json::Value {
    let seed = 0x5eeda11 ^ power;
    let keys = input(1usize << power, seed);
    let ids: Vec<u32> = (0..keys.len() as u32).collect();
    let mut timed = Vec::new();
    let mut total = Vec::new();
    let mut digest = String::new();
    let mut load_before = Vec::new();
    let mut load_after = Vec::new();
    for sample in 0..=samples {
        let before = load_1m();
        assert!(
            before.is_finite() && before < 8.0,
            "benchmark withheld: host load {before} must be under 8"
        );
        let (sorted, payload, algorithm_ms, wall_ms) = run(&keys, &ids);
        let after = load_1m();
        assert!(
            after.is_finite() && after < 8.0,
            "benchmark withheld: host load {after} reached 8 during sample"
        );
        load_before.push(before);
        load_after.push(after);
        let current = verify(&keys, &sorted, &payload);
        if sample == 0 {
            digest = current;
        } else {
            assert_eq!(digest, current);
            timed.push(algorithm_ms);
            total.push(wall_ms);
        }
        black_box((&sorted, &payload));
    }
    serde_json::json!({"schema":"invar-sort-benchmark-v2","backend":backend,"power":power,"elements":keys.len(),"seed":seed,
        "samples":samples,"warmups":1,"timed_ms":timed,"median_ms":median(&timed),"wall_ms":total,"wall_median_ms":median(&total),
        "calibration_id":"sort.sort_by_key.u32","load_1m_before":load_before,"load_1m_after":load_after,
        "load_sampling":"before and after each warmup/measured call","host":String::from_utf8(std::process::Command::new("hostname").output().expect("execution host receipt").stdout).expect("execution hostname").trim(),
        "keys":"u32","payload":"original-u32-index","distribution":"xorshift32-with-one-eighth-duplicate-small-keys","output_sha256":digest,
        "timed_boundary":if backend=="cuda"||backend=="metal"||backend=="standalone-arrowmetal" {"device events: all radix kernels, excluding upload, download, allocation and compilation"} else {"sort call: includes algorithm scratch allocation; excludes input clone and verification"}})
}
