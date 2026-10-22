// Physical D5 diagnostic against the unchanged P4a handwritten C1 baseline.
// Inputs and expected results are prepared by the CPU net oracle, not here.
use super::*;
use std::{fs, path::PathBuf};

#[test]
#[ignore = "explicit oracle-bound D5 inputs and exclusive foreground Metal slot"]
fn measure_d5_against_handwritten() {
    let root = PathBuf::from(std::env::var_os("INVAR_P5B_D5").expect("prepared D5 directory"));
    let cases = fs::read_to_string(root.join("cases.tsv")).unwrap();
    let mut device = MetalDevice::open_runtime_source().unwrap();
    println!("P5B_D5_DEVICE {}", device.device.name());
    for row in cases.lines() {
        let columns = row.split('\t').collect::<Vec<_>>();
        assert_eq!(columns.len(), 6);
        let id = columns[0];
        let a: i32 = columns[1].parse().unwrap();
        let b: i32 = columns[2].parse().unwrap();
        let expected: i32 = columns[3].parse().unwrap();
        assert!(matches!((a, b), (32, 3) | (256, 8))); // inside the frozen C1 proof domain
        let bytes = columns[5]
            .split(',')
            .map(|s| s.parse::<usize>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(bytes.len(), 12);
        let source = fs::read_to_string(root.join(format!("{id}.msl"))).unwrap();
        let baseline = fs::read_to_string(root.join(format!("{id}-baseline.msl"))).unwrap();
        let (generated, _) = device.compile(&source, KERNEL).unwrap();
        let (handwritten, _) = device.compile(&baseline, "C1").unwrap();
        let buffers = bytes
            .iter()
            .map(|n| device.buffer(*n).unwrap())
            .collect::<Vec<_>>();
        let baseline_buffers = [16, 20, 20, 4].map(|n| device.buffer(n).unwrap());
        let zero = device.buffer(bytes[6].max(bytes[11])).unwrap();
        // SAFETY: these new shared allocations are idle and cover every byte.
        unsafe {
            for (buffer, size) in buffers.iter().zip(&bytes) {
                ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), 0, *size);
            }
            ptr::write_bytes(zero.contents().as_ptr().cast::<u8>(), 0, zero.length());
        }
        let sample = |device: &mut MetalDevice, baseline: bool, host: bool| {
            let begun = Instant::now();
            // SAFETY: the previous sample is fenced; each retained allocation
            // covers these scalar words and remains alive through completion.
            unsafe {
                if baseline {
                    ptr::copy_nonoverlapping(
                        [a as u32, b as u32, 0, 1].as_ptr(),
                        baseline_buffers[0].contents().as_ptr().cast::<u32>(),
                        4,
                    );
                } else {
                    ptr::copy_nonoverlapping(
                        [a as u64, b as u64].as_ptr(),
                        buffers[0].contents().as_ptr().cast::<u64>(),
                        2,
                    );
                    ptr::copy_nonoverlapping(
                        [1_000_000u64, 1_000_000, 1_000_000, 1_000_000, 1, 0].as_ptr(),
                        buffers[3].contents().as_ptr().cast::<u64>(),
                        6,
                    );
                }
            }
            let command = device.queue.commandBuffer().unwrap();
            if !baseline {
                let blit = command.blitCommandEncoder().unwrap();
                for index in [6, 11] {
                    // SAFETY: both retained buffers cover the copied extent;
                    // reset and compute share one ordered command buffer.
                    unsafe {
                        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                            &zero,
                            0,
                            &buffers[index],
                            0,
                            bytes[index],
                        );
                    }
                }
                blit.endEncoding();
            }
            let refs = if baseline {
                baseline_buffers.iter().map(|b| &**b).collect::<Vec<_>>()
            } else {
                buffers.iter().map(|b| &**b).collect::<Vec<_>>()
            };
            device
                .encode(
                    &command,
                    if baseline { &handwritten } else { &generated },
                    &refs,
                    1,
                    if baseline { 256 } else { 32 },
                )
                .unwrap();
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(
                command.status(),
                MTLCommandBufferStatus::Completed,
                "{:?}",
                command.error()
            );
            // SAFETY: completion precedes these bounded shared-memory reads.
            let (value, status, done) = unsafe {
                if baseline {
                    let out = baseline_buffers[2].contents().as_ptr().cast::<u32>();
                    (ptr::read(out) as i32, ptr::read(out.add(1)) as u64, true)
                } else {
                    let out = buffers[1].contents().as_ptr().cast::<u64>();
                    let state = buffers[11].contents().as_ptr().cast::<u64>();
                    (
                        ptr::read(out.add(1)) as i32,
                        ptr::read(out),
                        ptr::read(state.add(12)) == 0,
                    )
                }
            };
            let wall = begun.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(
                (value, status, done),
                (expected, 0, true),
                "{id}({a},{b}) baseline={baseline}"
            );
            let gpu = (command.GPUEndTime() - command.GPUStartTime()) * 1000.0;
            assert!(gpu.is_finite() && gpu > 0.0);
            if host { wall } else { gpu }
        };
        for session in 0..3 {
            for host in [false, true] {
                for _ in 0..10 {
                    sample(&mut device, false, host);
                    sample(&mut device, true, host);
                }
                for block in 0..30 {
                    let first_baseline = (block + session) % 2 == 0;
                    let first = sample(&mut device, first_baseline, host);
                    let second = sample(&mut device, !first_baseline, host);
                    let (generated, baseline) = if first_baseline {
                        (second, first)
                    } else {
                        (first, second)
                    };
                    println!(
                        "P5B_D5,{id},{a},{b},{},{},{session},{block},{generated:.9},{baseline:.9}",
                        columns[4],
                        if host { "host" } else { "resident" }
                    );
                }
            }
        }
    }
}
