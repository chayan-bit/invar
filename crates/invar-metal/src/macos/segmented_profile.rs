//! Resident 10/30-sample comparison, including isolated per-dispatch timestamps.
//! Run alone, in the foreground, on an otherwise idle Mac.
use super::*;

type Launch = (usize, Vec<usize>, u32, u32);

fn submit(
    device: &mut MetalDevice,
    pipes: &[PipelineRef],
    buffers: &[BufferRef],
    launches: &[Launch],
) -> f64 {
    let command = device.queue.commandBuffer().unwrap();
    for (kernel, slots, grid, lanes) in launches {
        let bound = slots.iter().map(|&s| &*buffers[s]).collect::<Vec<_>>();
        device
            .encode(&command, &pipes[*kernel], &bound, *grid, *lanes)
            .unwrap();
    }
    let mut work = MetalWork::prepared(command.clone(), buffers.to_vec(), pipes.to_vec());
    work.submit().unwrap();
    assert_eq!(
        device.await_terminal(work, Instant::now()).unwrap().state,
        WorkState::Completed
    );
    (command.GPUEndTime() - command.GPUStartTime()) * 1000.0
}

fn copy(device: &mut MetalDevice, input: &BufferRef, output: &BufferRef, bytes: usize) -> f64 {
    let command = device.queue.commandBuffer().unwrap();
    let encoder = command.blitCommandEncoder().unwrap();
    // SAFETY: both buffers cover bytes, retained through terminal completion.
    unsafe {
        encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
            input, 0, output, 0, bytes,
        );
    }
    encoder.endEncoding();
    let mut work =
        MetalWork::prepared(command.clone(), vec![input.clone(), output.clone()], vec![]);
    work.submit().unwrap();
    assert_eq!(
        device.await_terminal(work, Instant::now()).unwrap().state,
        WorkState::Completed
    );
    (command.GPUEndTime() - command.GPUStartTime()) * 1000.0
}

fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty() && values.iter().all(|v| v.is_finite() && *v > 0.0));
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    (values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.0
}

fn dispatch_times(
    device: &mut MetalDevice,
    pipes: &[PipelineRef],
    buffers: &[BufferRef],
    launches: &[Launch],
) -> Vec<f64> {
    let times = launches
        .iter()
        .enumerate()
        .map(|(index, launch)| {
            let times = (0..5)
                .map(|_| {
                    if index > 0 {
                        submit(device, pipes, buffers, &launches[..index]);
                    }
                    submit(device, pipes, buffers, std::slice::from_ref(launch))
                })
                .collect::<Vec<_>>();
            median(&times)
        })
        .collect();
    submit(device, pipes, buffers, launches);
    times
}

fn read(buffer: &BufferRef, words: usize) -> &[u32] {
    assert!(
        words <= buffer.length() / 4,
        "GPU output length exceeds allocation"
    );
    // SAFETY: callers pass the allocation's word count after terminal completion.
    unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast(), words) }
}

#[test]
#[ignore = "exclusive foreground Apple M4 timing sweep, up to 2^28 words"]
fn issue_m4_04_metal_profile() {
    // Profiles compile generated and baseline MSL: the developer source path.
    let mut device = MetalDevice::open_runtime_source().unwrap();
    println!(
        "PROFILE protocol=ABBA_15_rounds_10_warmups device={} caps={:?}",
        device.name(),
        device.capabilities()
    );
    let sizes = std::env::var("INVAR_PROFILE_SIZES").unwrap_or_else(|_| {
        "1000000,1048576,2097152,4194304,8388608,16777216,33554432,67108864,134217728,268435456"
            .into()
    });
    for n in sizes.split(',').map(|v| v.parse::<u32>().unwrap()) {
        assert!((1..=seg::MAX_ELEMENTS).contains(&n) && n.is_multiple_of(16));
        let source = include_str!("segmented_baseline.metal").replace("@N@", &n.to_string());
        let baseline_pipes = [
            "segsum",
            "compact_count",
            "compact_offsets",
            "compact_scatter",
            "map16",
        ]
        .map(|name| device.compile(&source, name).unwrap().0);
        let input = (0..n)
            .map(|i| i.wrapping_mul(1664525).wrapping_add(1013904223))
            .collect::<Vec<_>>();
        let offsets = (0..=n / 16).map(|s| s * 16).collect::<Vec<_>>();
        for family in [
            seg::Family::Reduce(seg::Law::WrappingSum),
            seg::Family::Map(vec![
                seg::MapOp::WrappingAdd(7),
                seg::MapOp::WrappingMul(2654435761),
                seg::MapOp::AddSegment,
            ]),
            seg::Family::Compact {
                predicate: seg::Predicate::Less,
                cutoff: 1 << 31,
            },
        ] {
            let compact = matches!(family, seg::Family::Compact { .. });
            let name = match family {
                seg::Family::Map(_) => "map",
                seg::Family::Reduce(_) => "sum",
                _ => "compact",
            };
            let plan = seg::verify(
                seg::Ir {
                    version: seg::VERSION,
                    span: invar_device::Span { start: 1, end: 2 },
                    source_identity: 1,
                    graph_revision: 1,
                    function_contract: [1; 32],
                    family,
                    elements: n,
                    segments: if compact { 0 } else { n / 16 },
                },
                device.caps,
                invar_device::Provenance {
                    source_identity: 1,
                    graph_revision: 1,
                },
            )
            .unwrap();
            let source = seg::emit_msl(&plan).unwrap();
            let pipes = plan
                .kernels()
                .iter()
                .map(|name| device.compile(&source, name).unwrap().0)
                .collect::<Vec<_>>();
            let words = plan.buffer_words();
            let buffers = words
                .iter()
                .map(|&w| device.buffer(w.max(1) * 4).unwrap())
                .collect::<Vec<_>>();
            for (values, buffer) in [input.as_slice(), if compact { &[] } else { &offsets }]
                .into_iter()
                .zip(&buffers)
            {
                // SAFETY: exact admitted buffer sizes, prior to any submission.
                unsafe {
                    ptr::copy_nonoverlapping(
                        values.as_ptr(),
                        buffer.contents().as_ptr().cast(),
                        values.len(),
                    );
                }
            }
            let launches = plan
                .launches(&input, if compact { &[] } else { &offsets })
                .unwrap()
                .into_iter()
                .map(|l| (l.kernel, l.buffers.to_vec(), l.blocks, seg::LANES))
                .collect::<Vec<_>>();
            let base_buffers = vec![
                buffers[0].clone(),
                device.buffer(n as usize * 4).unwrap(),
                device.buffer(n.div_ceil(256) as usize * 4).unwrap(),
                device.buffer(4).unwrap(),
            ];
            let base_launches = if compact {
                vec![
                    (1, vec![0, 2], n.div_ceil(256), 256),
                    (2, vec![2, 3], 1, 1024),
                    (3, vec![0, 2, 1], n.div_ceil(256), 256),
                ]
            } else {
                vec![(
                    if name == "sum" { 0 } else { 4 },
                    vec![0, 1],
                    (n / 16).div_ceil(256),
                    256,
                )]
            };
            let copy_out = device.buffer(n as usize * 4).unwrap();
            for _ in 0..10 {
                submit(&mut device, &pipes, &buffers, &launches);
                submit(&mut device, &baseline_pipes, &base_buffers, &base_launches);
                copy(&mut device, &buffers[0], &copy_out, n as usize * 4);
            }
            // Fixed ABBA order, 15 rounds = 30 samples per implementation.
            // Pairing avoids attributing clock/thermal drift to the compiler.
            let mut raw = Vec::new();
            let mut base_raw = Vec::new();
            let mut copies = Vec::new();
            for _ in 0..15 {
                raw.push(submit(&mut device, &pipes, &buffers, &launches));
                base_raw.push(submit(
                    &mut device,
                    &baseline_pipes,
                    &base_buffers,
                    &base_launches,
                ));
                base_raw.push(submit(
                    &mut device,
                    &baseline_pipes,
                    &base_buffers,
                    &base_launches,
                ));
                raw.push(submit(&mut device, &pipes, &buffers, &launches));
                copies.push(copy(&mut device, &buffers[0], &copy_out, n as usize * 4));
                copies.push(copy(&mut device, &buffers[0], &copy_out, n as usize * 4));
            }
            drop(copy_out);
            let passes = dispatch_times(&mut device, &pipes, &buffers, &launches);
            let base_passes =
                dispatch_times(&mut device, &baseline_pipes, &base_buffers, &base_launches);
            let ms = median(&raw);
            let base_ms = median(&base_raw);
            let out_slot = if compact { 1 } else { 2 };
            let length = if compact {
                read(&buffers[4], 1)[0] as usize
            } else {
                words[out_slot]
            };
            let output = read(&buffers[out_slot], length);
            if compact {
                assert!(
                    input
                        .iter()
                        .copied()
                        .filter(|v| *v < 1 << 31)
                        .eq(output.iter().copied())
                );
            } else if name == "sum" {
                assert!(
                    input
                        .chunks(16)
                        .map(|xs| xs.iter().fold(0u32, |a, v| a.wrapping_add(*v)))
                        .eq(output.iter().copied())
                );
            } else {
                assert!(
                    input
                        .iter()
                        .enumerate()
                        .map(|(i, v)| v
                            .wrapping_add(7)
                            .wrapping_mul(2654435761)
                            .wrapping_add(i as u32 / 16))
                        .eq(output.iter().copied())
                );
            }
            let base_len = if compact {
                read(&base_buffers[3], 1)[0] as usize
            } else {
                length
            };
            assert_eq!(base_len, length);
            assert_eq!(read(&base_buffers[1], length), output);
            let copy_ms = median(&copies);
            let required = (n as usize + length) * 4;
            println!(
                "ROW\t{name}\t{n}\t{ms:.6}\t{base_ms:.6}\t{copy_ms:.6}\t{:.6}\t{:.6}\t{}\t{:?}\t{:?}\t{:016x}",
                n as f64 * 8.0 / copy_ms / 1e6,
                required as f64 / ms / 1e6,
                launches.len(),
                passes,
                base_passes,
                source_hash(&source)
            );
            println!("SAMPLES\t{name}\t{n}\tinvar={raw:?}\tbaseline={base_raw:?}\tcopy={copies:?}");
        }
    }
}

#[test]
#[ignore = "requires Apple7+ Metal; cross-tile carry and head-flag regression"]
fn issue_m4_04_metal_tiled_boundaries() {
    // Profiles compile generated and baseline MSL: the developer source path.
    let mut device = MetalDevice::open_runtime_source().unwrap();
    let n = 131_111;
    // Odd, nonzero values prevent wrapping products from masking missing carries.
    let input = (0..n)
        .map(|i: u32| i.wrapping_mul(1664525).wrapping_add(1013904223) | 1)
        .collect::<Vec<_>>();
    let offsets = [
        0, 0, 1, 256, 256, 257, 513, 514, 777, 4097, 65_537, 65_538, n, n,
    ];
    for family in [
        seg::Family::Reduce(seg::Law::WrappingSum),
        seg::Family::Reduce(seg::Law::WrappingProduct),
        seg::Family::Compact {
            predicate: seg::Predicate::Less,
            cutoff: 1 << 31,
        },
    ] {
        let compact = matches!(family, seg::Family::Compact { .. });
        let bounds = if compact { &[][..] } else { &offsets[..] };
        let plan = seg::verify(
            seg::Ir {
                version: seg::VERSION,
                span: invar_device::Span { start: 1, end: 2 },
                source_identity: 1,
                graph_revision: 1,
                function_contract: [1; 32],
                family,
                elements: n,
                segments: if compact { 0 } else { offsets.len() as u32 - 1 },
            },
            device.caps,
            invar_device::Provenance {
                source_identity: 1,
                graph_revision: 1,
            },
        )
        .unwrap();
        let expected = seg::reference(&plan, &input, bounds).unwrap();
        let run = device.run_segmented(&plan, &input, bounds).unwrap();
        assert_eq!(run.output, expected, "family={:?}", plan.ir().family);
        println!(
            "boundary family={:?} words={} launches={} bit_exact=true",
            plan.ir().family,
            run.output.len(),
            plan.launches(&input, bounds).unwrap().len()
        );
    }
}
