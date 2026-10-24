#[path = "../support/spatial_oracle.rs"]
#[allow(dead_code)]
mod oracle;
use invar::numeric_contract::F32Bits as F;
use sha2::{Digest, Sha256};

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
fn real_pcl_radius_demo_matches_independent_block_oracle() {
    let bytes = include_bytes!("../../test-support/data/spatial/pcl4096-f32-soa.bin");
    let lineage: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../test-support/data/spatial/pcl4096-lineage.json"
    ))
    .unwrap();
    assert_eq!(bytes.len(), 3 * 4096 * 4);
    assert_eq!(
        sha256(bytes),
        lineage["sample_soa_sha256"].as_str().unwrap()
    );
    let points: [Vec<F>; 3] = std::array::from_fn(|axis| {
        bytes[axis * 4096 * 4..(axis + 1) * 4096 * 4]
            .chunks_exact(4)
            .map(|v| F::from_bits(u32::from_le_bytes(v.try_into().unwrap())))
            .collect()
    });
    let queries: [Vec<F>; 3] =
        std::array::from_fn(|a| (0..16).map(|q| points[a][q * 4096 / 16]).collect());
    let q = oracle::Cloud::new(&queries[0], &queries[1], &queries[2]).unwrap();
    let mut counts = vec![0u32; 16];
    for start in (0..4096).step_by(1024) {
        let cloud = oracle::Cloud::new(
            &points[0][start..start + 1024],
            &points[1][start..start + 1024],
            &points[2][start..start + 1024],
        )
        .unwrap();
        let (block_counts, _) =
            oracle::radius_neighbors(cloud, q, F::from_bits(0x3d4ccccd), 0).unwrap();
        for (count, block) in counts.iter_mut().zip(block_counts) {
            *count += block;
        }
    }
    let total: u32 = counts.iter().sum();
    assert!(counts.iter().all(|n| *n > 0)); // each real query includes itself
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_invar"))
        .args([
            "run",
            "demo/spatial-point-cloud.inv",
            "--engine",
            "reference",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output);
    assert!(output.stderr.is_empty());
    let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["status"], "ok");
    assert_eq!(document["result"]["value"], total);
    if let Some(path) = std::env::var_os("SPATIAL_DEMO_RECEIPT") {
        let source = std::fs::read("demo/spatial-point-cloud.inv").unwrap();
        let receipt = serde_json::json!({
            "source_sha256": sha256(&source),
            "sample_soa_sha256": lineage["sample_soa_sha256"],
            "point_count":4096,"query_count":16,"counts":counts,"total":total,
            "radius_f32_bits":"3d4ccccd","stride":32,
            "oracle":"four disjoint1024-point blocks; sum every true count",
            "cli":document,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
    }
}
