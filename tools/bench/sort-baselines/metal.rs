//! Independent ArrowMetal radix baseline: Rust host, pinned upstream MSL.
#[cfg(target_os = "macos")]
#[path = "../sort-common.rs"]
mod common;
#[cfg(target_os = "macos")]
fn main() {
    use objc2_foundation::NSString;
    use objc2_metal::*;
    use std::{
        ptr,
        time::{Duration, Instant},
    };
    let args: Vec<String> = std::env::args().collect();
    let maximum: u32 = args.get(1).map(|v| v.parse().unwrap()).unwrap_or(26);
    let samples: usize = args.get(2).map(|v| v.parse().unwrap()).unwrap_or(5);
    assert!((16..=26).contains(&maximum) && (3..=21).contains(&samples));
    let device = MTLCreateSystemDefaultDevice().expect("Metal device");
    let queue = device.newCommandQueue().unwrap();
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version3_1);
    #[allow(deprecated)]
    options.setFastMathEnabled(false);
    let library = device
        .newLibraryWithSource_options_error(
            &NSString::from_str(include_str!("arrowmetal.metal")),
            Some(&options),
        )
        .unwrap();
    let pipelines: Vec<_> = ["radix_histogram", "radix_scan", "radix_scatter"]
        .iter()
        .map(|name| {
            let f = library
                .newFunctionWithName(&NSString::from_str(name))
                .unwrap();
            let p = device
                .newComputePipelineStateWithFunction_error(&f)
                .unwrap();
            assert!(p.threadExecutionWidth() == 32 && p.maxTotalThreadsPerThreadgroup() >= 256);
            p
        })
        .collect();
    for power in 16..=maximum {
        println!(
            "{}",
            common::row("standalone-arrowmetal", power, samples, |keys, ids| {
                let n = keys.len();
                // Upstream blockPlan, unchanged for dense u32 keys and payloads.
                let mut e = 4096usize.max(n.div_ceil(128).div_ceil(256) * 256);
                while e > 256 && n.div_ceil(e) < 64 {
                    e >>= 1;
                }
                let blocks = n.div_ceil(e);
                let sizes = [
                    n,
                    n,
                    n,
                    n,
                    256 * blocks,
                    blocks,
                    blocks,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ];
                let wall = Instant::now();
                let buffers: Vec<_> = sizes
                    .iter()
                    .map(|&size| {
                        device
                            .newBufferWithLength_options(
                                size * 4,
                                MTLResourceOptions::StorageModeShared,
                            )
                            .unwrap()
                    })
                    .collect();
                // SAFETY: all buffers are private shared storage of their declared
                // sizes, initialized before submission and retained until terminal.
                unsafe {
                    ptr::copy_nonoverlapping(
                        keys.as_ptr(),
                        buffers[0].contents().as_ptr().cast(),
                        n,
                    );
                    ptr::copy_nonoverlapping(
                        ids.as_ptr(),
                        buffers[1].contents().as_ptr().cast(),
                        n,
                    );
                    for (slot, value) in [
                        (7, n),
                        (8, e),
                        (9, blocks),
                        (10, 256 * blocks),
                        (11, 0),
                        (12, 8),
                        (13, 16),
                        (14, 24),
                    ] {
                        *buffers[slot].contents().as_ptr().cast::<u32>() = value as u32;
                    }
                }
                let command = queue.commandBuffer().unwrap();
                for pass in 0..4 {
                    let (ka, va, kb, vb) = if pass % 2 == 0 {
                        (0, 1, 2, 3)
                    } else {
                        (2, 3, 0, 1)
                    };
                    for (kernel, slots, groups) in [
                        (0, vec![ka, 7, 11 + pass, 8, 9, 4, 5, 6], blocks),
                        (1, vec![4, 10], 1),
                        (2, vec![ka, va, 7, 11 + pass, 8, 9, 4, kb, vb], blocks),
                    ] {
                        let enc = command.computeCommandEncoder().unwrap();
                        enc.setComputePipelineState(&pipelines[kernel]);
                        for (index, slot) in slots.into_iter().enumerate() {
                            // SAFETY: pinned shader signatures and matching buffer
                            // sizes; all buffers outlive this command.
                            unsafe {
                                enc.setBuffer_offset_atIndex(Some(&buffers[slot]), 0, index);
                            }
                        }
                        enc.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: groups,
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: 256,
                                height: 1,
                                depth: 1,
                            },
                        );
                        enc.endEncoding();
                    }
                }
                command.commit();
                let deadline = Instant::now();
                loop {
                    match command.status() {
                        MTLCommandBufferStatus::Completed => break,
                        MTLCommandBufferStatus::Error => {
                            panic!("Metal baseline: {:?}", command.error())
                        }
                        _ => {
                            // On timeout retain all in-flight resources. The outer
                            // bounded process controls the run's ultimate lifetime.
                            if deadline.elapsed() > Duration::from_secs(30) {
                                std::mem::forget(buffers);
                                panic!("Metal baseline timed out");
                            }
                            std::thread::sleep(Duration::from_micros(50));
                        }
                    }
                }
                let ms = (command.GPUEndTime() - command.GPUStartTime()) * 1000.0;
                let read = |slot: usize| {
                    let mut out = vec![0u32; n];
                    // SAFETY: completion observed, retained initialized output.
                    unsafe {
                        ptr::copy_nonoverlapping(
                            buffers[slot].contents().as_ptr().cast(),
                            out.as_mut_ptr(),
                            n,
                        );
                    }
                    out
                };
                (read(0), read(1), ms, wall.elapsed().as_secs_f64() * 1000.0)
            })
        );
    }
}
#[cfg(not(target_os = "macos"))]
fn main() {
    panic!("physical Metal required");
}
