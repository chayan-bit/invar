//! Explicit stable u32 key/u32 payload timings. Run the built binary directly
//! under the host's heavy/GPU lock; no timing includes input cloning or hashes.
use invar::sorting::{KeyKind, sort_words};
use rayon::prelude::*;
use std::time::Instant;
#[path = "sort-common.rs"]
mod common;
use common::row;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let backend = args.get(1).map(String::as_str).unwrap_or("cpu");
    let maximum = args.get(2).map(|s| s.parse::<u32>().unwrap()).unwrap_or(26);
    let samples = args
        .get(3)
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(5);
    assert!((16..=26).contains(&maximum) && (3..=21).contains(&samples));
    if backend == "cpu" {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for power in 16..=maximum {
            for name in [
                "rust-sort_by_key",
                "rayon-par_sort",
                "invar-cpu-1",
                "invar-cpu-4",
            ] {
                let result = row(name, power, samples, |keys, ids| {
                    if name.starts_with("invar") {
                        let mut keys = keys.to_vec();
                        let mut ids = ids.to_vec();
                        let start = Instant::now();
                        sort_words(
                            &mut keys,
                            Some(&mut ids),
                            KeyKind::U32,
                            None,
                            if name.ends_with('1') { 1 } else { 4 },
                        )
                        .unwrap();
                        let ms = start.elapsed().as_secs_f64() * 1000.0;
                        (keys, ids, ms, ms)
                    } else {
                        let mut pairs: Vec<_> =
                            keys.iter().copied().zip(ids.iter().copied()).collect();
                        let start = Instant::now();
                        if name.starts_with("rayon") {
                            pool.install(|| pairs.par_sort_by_key(|v| v.0));
                        } else {
                            pairs.sort_by_key(|v| v.0);
                        }
                        let ms = start.elapsed().as_secs_f64() * 1000.0;
                        let (keys, ids) = pairs.into_iter().unzip();
                        (keys, ids, ms, ms)
                    }
                });
                println!("{result}");
            }
        }
        return;
    }
    #[cfg(feature = "cuda")]
    if backend == "cuda" {
        let mut device = invar_cuda::cuda::CudaDevice::open().unwrap();
        let compiler = std::env::var_os("INVAR_NVRTC_COMPILER")
            .map(std::path::PathBuf::from)
            .expect("INVAR_NVRTC_COMPILER");
        for power in 16..=maximum {
            println!(
                "{}",
                row(backend, power, samples, |keys, ids| {
                    let p = invar_device::sorting::Plan::new(
                        keys.len() as u32,
                        invar_device::sorting::Kind::U32,
                    )
                    .unwrap();
                    let start = Instant::now();
                    let result = device.run_sort(&p, keys, ids, &compiler).unwrap();
                    let wall = start.elapsed().as_secs_f64() * 1000.0;
                    (result.keys, result.payload, result.kernel_ms, wall)
                })
            );
        }
        return;
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if backend == "metal" {
        let mut device = invar_metal::MetalDevice::open().unwrap();
        for power in 16..=maximum {
            println!(
                "{}",
                row(backend, power, samples, |keys, ids| {
                    let p = invar_device::sorting::Plan::new(
                        keys.len() as u32,
                        invar_device::sorting::Kind::U32,
                    )
                    .unwrap();
                    let start = Instant::now();
                    let result = device.run_sort(&p, keys, ids).unwrap();
                    let wall = start.elapsed().as_secs_f64() * 1000.0;
                    (result.keys, result.payload, result.kernel_ms, wall)
                })
            );
        }
        return;
    }
    panic!("unavailable benchmark backend: {backend}");
}
