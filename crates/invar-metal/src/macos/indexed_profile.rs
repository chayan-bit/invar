//! Test-only resident profiler. ABBA pairing, GPU timestamps and same-run blit.
use super::*;
use invar_device::{AddressSpace, Buffer, Layout, Span, indexed as ir};

fn read(buffer: &BufferRef, n: usize) -> &[u32] {
    assert!(n <= buffer.length() / 4);
    // SAFETY: all callers wait for terminal completion before reading.
    unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast(), n) }
}
fn write(buffer: &BufferRef, words: &[u32]) {
    assert!(words.len() <= buffer.length() / 4);
    // SAFETY: retained allocation, no outstanding GPU work, disjoint host input.
    unsafe {
        ptr::copy_nonoverlapping(
            words.as_ptr(),
            buffer.contents().as_ptr().cast(),
            words.len(),
        )
    }
}
fn clear(buffer: &BufferRef) {
    // SAFETY: initialized shared allocation with no submitted users.
    unsafe { ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), 0, buffer.length()) }
}
fn complete(
    device: &mut MetalDevice,
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    buffers: &[BufferRef],
    pipes: &[PipelineRef],
) -> f64 {
    let mut work = MetalWork::prepared(command.clone(), buffers.to_vec(), pipes.to_vec());
    work.submit().unwrap();
    assert_eq!(
        device.await_terminal(work, Instant::now()).unwrap().state,
        WorkState::Completed
    );
    (command.GPUEndTime() - command.GPUStartTime()) * 1000.0
}
fn median(xs: &[f64]) -> f64 {
    assert!(xs.iter().all(|x| x.is_finite() && *x > 0.0));
    let mut xs = xs.to_vec();
    xs.sort_by(f64::total_cmp);
    (xs[(xs.len() - 1) / 2] + xs[xs.len() / 2]) / 2.0
}
fn before_dimensions(p: &ir::Verified) -> (u32, u32) {
    if matches!(p.ir().operation, ir::Operation::Frontier { .. }) {
        (1, 1)
    } else {
        (p.ir().indices.elements.div_ceil(64).clamp(1, 65535), 64)
    }
}
fn generated(
    device: &mut MetalDevice,
    p: &ir::Verified,
    pipes: &[PipelineRef],
    buffers: &[BufferRef],
    rounds: &[BufferRef],
    separate: bool,
) -> f64 {
    objc2::rc::autoreleasepool(|_| {
        for b in &buffers[2..] {
            clear(b);
        }
        if matches!(p.ir().operation, ir::Operation::Frontier { .. }) {
            write(&buffers[3], &[1]);
            write(&buffers[5], &[0]);
            write(&buffers[6], &[1, 0]);
        }
        let mut command = device.queue.commandBuffer().unwrap();
        let mut ms = 0.0;
        for (round, kernel) in ir::schedule(p) {
            let (blocks, lanes) = if separate {
                before_dimensions(p)
            } else {
                ir::phase_launch_dimensions(p, kernel)
            };
            let mut refs = buffers.iter().map(|b| &**b).collect::<Vec<_>>();
            refs.push(&rounds[round as usize]);
            device
                .encode(&command, &pipes[kernel], &refs, blocks, lanes)
                .unwrap();
            if separate {
                ms += complete(device, command, buffers, pipes);
                command = device.queue.commandBuffer().unwrap();
            }
        }
        if !separate {
            ms = complete(device, command, buffers, pipes);
        }
        ms
    })
}
fn baseline(
    device: &mut MetalDevice,
    family: &str,
    v: u32,
    pipe: &PipelineRef,
    buffers: &[BufferRef],
) -> f64 {
    objc2::rc::autoreleasepool(|_| {
        clear(&buffers[2]);
        if family != "frontier" {
            let command = device.queue.commandBuffer().unwrap();
            let refs = buffers[..3].iter().map(|b| &**b).collect::<Vec<_>>();
            device
                .encode(
                    &command,
                    pipe,
                    &refs,
                    (buffers[1].length() as u32 / 4).div_ceil(256),
                    256,
                )
                .unwrap();
            return complete(device, command, buffers, std::slice::from_ref(pipe));
        }
        clear(&buffers[3]);
        clear(&buffers[4]);
        write(&buffers[3], &[1]);
        let mut ms = 0.0;
        for _ in 0..16 {
            clear(&buffers[5]);
            let command = device.queue.commandBuffer().unwrap();
            let refs = [0, 2, 3, 4, 5].map(|i| &*buffers[i]);
            device
                .encode(&command, pipe, &refs, v.div_ceil(256), 256)
                .unwrap();
            ms += complete(device, command, buffers, std::slice::from_ref(pipe));
            // The pinned baseline copies and resets its next mask on the host.
            write(&buffers[3], read(&buffers[4], v as usize));
            clear(&buffers[4]);
            if read(&buffers[5], 1)[0] == 0 {
                break;
            }
        }
        ms
    })
}
fn copy_time(device: &mut MetalDevice, a: &BufferRef, b: &BufferRef) -> f64 {
    objc2::rc::autoreleasepool(|_| {
        let command = device.queue.commandBuffer().unwrap();
        let encoder = command.blitCommandEncoder().unwrap();
        // SAFETY: distinct retained equal-sized allocations, no pending users.
        unsafe {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                a,
                0,
                b,
                0,
                a.length(),
            );
        }
        encoder.endEncoding();
        complete(device, command, &[a.clone(), b.clone()], &[])
    })
}
#[test]
#[ignore = "exclusive foreground M4 resident timing, 10 warmups plus 30 ABBA samples"]
fn issue_m4_05_metal_profile() {
    // Profiles compile generated and baseline MSL: the developer source path.
    let mut device = MetalDevice::open_runtime_source().unwrap();
    let before = std::env::var_os("INVAR_PROFILE_BEFORE").is_some();
    let sizes = std::env::var("INVAR_PROFILE_SIZES").unwrap_or_else(|_| {
        "1024,1048576,2097152,4194304,8388608,16777216,33554432,67108864,134217728,268435456".into()
    });
    let families = std::env::var("INVAR_PROFILE_FAMILIES")
        .unwrap_or_else(|_| "gather,scatter,histogram,frontier".into());
    println!(
        "PROFILE device={} caps={:?} before={before} warmups=10 samples=30 ABBA",
        device.name(),
        device.caps
    );
    for cells in sizes.split(',').map(|x| x.parse::<u32>().unwrap()) {
        assert!(cells.is_power_of_two() && (1024..=ir::MAX_ELEMENTS).contains(&cells));
        for family in families.split(',') {
            objc2::rc::autoreleasepool(|_| {
                let v = if family == "frontier" {
                    cells.isqrt()
                } else {
                    0
                };
                let n = if v > 0 { v } else { cells };
                let op = match family {
                    "gather" => ir::Operation::Gather,
                    "scatter" => ir::Operation::InjectiveScatter,
                    "histogram" => ir::Operation::Histogram,
                    "frontier" => ir::Operation::Frontier {
                        vertices: v,
                        rounds: 16,
                        capacity: v,
                    },
                    _ => panic!("unknown family"),
                };
                let output = if v > 0 {
                    v * 17
                } else if family == "histogram" {
                    cells.min(65536)
                } else {
                    cells
                };
                let buf = |id, elements| Buffer {
                    id,
                    elements,
                    address: AddressSpace::Device,
                    layout: Layout::ContiguousU32,
                };
                let raw = ir::Ir {
                    version: ir::VERSION,
                    source_identity: 1,
                    graph_revision: 0,
                    source_contract: [1; 32],
                    span: Span { start: 1, end: 2 },
                    operation: op,
                    values: buf(0, cells),
                    indices: buf(1, if v > 0 { 0 } else { cells }),
                    output: buf(2, output),
                    law: (family == "histogram").then(|| ir::HISTOGRAM_LAW.into()),
                };
                let plan = ir::verify(raw.clone(), &raw, device.caps).unwrap();
                let source = ir::emit_msl(&plan).unwrap();
                let source = if before && v > 0 {
                    let header = source.split("kernel void").next().unwrap();
                    format!(
                        "{header}{}",
                        include_str!("indexed_before.metal").replace("@V@", &v.to_string())
                    )
                } else if before {
                    let old = before_dimensions(&plan);
                    let new = ir::launch_dimensions(&plan);
                    source.replace(
                        &format!("i+={}u", new.0 * new.1),
                        &format!("i+={}u", old.0 * old.1),
                    )
                } else {
                    source
                };
                let started = Instant::now();
                let pipes = ir::kernel_names(&plan)
                    .iter()
                    .map(|name| device.compile(&source, name).unwrap().0)
                    .collect::<Vec<_>>();
                let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
                let baseline_source = include_str!("indexed_baseline.metal")
                    .replace("@N@", &cells.to_string())
                    .replace("@V@", &v.to_string());
                let started = Instant::now();
                let pipe = device
                    .compile(
                        &baseline_source,
                        &format!(
                            "{}_{}",
                            if v > 0 { "frontier" } else { family },
                            if v > 0 { "round" } else { "hw" }
                        ),
                    )
                    .unwrap()
                    .0;
                let base_compile_ms = started.elapsed().as_secs_f64() * 1000.0;
                let lengths = [
                    cells,
                    if v > 0 { 1 } else { cells },
                    output,
                    n.min(v).max(1),
                    n.min(v).max(1),
                    n.min(v).max(1),
                    2,
                ];
                let buffers = lengths
                    .iter()
                    .map(|&x| device.buffer(x.max(1) as usize * 4).unwrap())
                    .collect::<Vec<_>>();
                for b in &buffers {
                    clear(b);
                }
                // SAFETY: initialized private shared buffers; no GPU submission yet.
                unsafe {
                    let a = std::slice::from_raw_parts_mut(
                        buffers[0].contents().as_ptr().cast::<u32>(),
                        cells as usize,
                    );
                    if v > 0 {
                        for u in 0..v {
                            a[(u * v + (u + 1) % v) as usize] = 1;
                            a[(u * v + (u * 5 + 1) % v) as usize] = 1;
                        }
                    } else {
                        for (i, x) in a.iter_mut().enumerate() {
                            *x = (i as u32).wrapping_mul(1664525).wrapping_add(1013904223);
                        }
                    }
                    if v == 0 {
                        let ix = std::slice::from_raw_parts_mut(
                            buffers[1].contents().as_ptr().cast::<u32>(),
                            cells as usize,
                        );
                        for (i, x) in ix.iter_mut().enumerate() {
                            *x = if family == "histogram" {
                                i as u32 % output
                            } else {
                                (i as u32).wrapping_mul(2654435761) & (cells - 1)
                            };
                        }
                    }
                }
                let rounds = (0..16)
                    .map(|r| {
                        let b = device.buffer(4).unwrap();
                        write(&b, &[r]);
                        b
                    })
                    .collect::<Vec<_>>();
                // ABBA runs sequentially with an immediate full-output check. Share
                // the output/scratch to keep the 2^28 sweep within laptop memory.
                let base = buffers.clone();
                let copy_out = if matches!(family, "gather" | "scatter") {
                    buffers[2].clone()
                } else {
                    device.buffer(cells as usize * 4).unwrap()
                };
                let mut expected: Vec<u32> = if matches!(family, "gather" | "scatter") {
                    vec![]
                } else {
                    vec![0; output as usize]
                };
                if v > 0 {
                    let mut front = vec![0; v as usize];
                    front[0] = 1;
                    let a = read(&buffers[0], cells as usize);
                    for round in 0..16usize {
                        let mut next = vec![0; v as usize];
                        for u in 0..v as usize {
                            if front[u] == 1 && expected[u] == 0 {
                                expected[u] = 1;
                                for w in 0..v as usize {
                                    if a[u * v as usize + w] == 1 && expected[w] == 0 {
                                        next[w] = 1;
                                    }
                                }
                            }
                        }
                        expected[(round + 1) * v as usize..(round + 2) * v as usize]
                            .copy_from_slice(&next);
                        front = next;
                    }
                } else if family == "histogram" {
                    for (i, &ix) in read(&buffers[1], cells as usize).iter().enumerate() {
                        let value = read(&buffers[0], cells as usize)[i];
                        expected[ix as usize] = expected[ix as usize].wrapping_add(value);
                    }
                }
                let verify = |b: &[BufferRef], baseline: bool| {
                    let len = if baseline && v > 0 {
                        v as usize
                    } else {
                        output as usize
                    };
                    let got = read(&b[2], len);
                    if matches!(family, "gather" | "scatter") {
                        let multiplier = if family == "gather" {
                            2654435761u32
                        } else {
                            244002641u32
                        };
                        assert_eq!(2654435761u32.wrapping_mul(244002641), 1);
                        assert!(
                            got.iter().enumerate().all(|(i, &x)| {
                                let index = (i as u32).wrapping_mul(multiplier) & (cells - 1);
                                x == index.wrapping_mul(1664525).wrapping_add(1013904223)
                            }),
                            "family={family} cells={cells} baseline={baseline}"
                        );
                    } else {
                        assert_eq!(
                            got,
                            &expected[..len],
                            "family={family} cells={cells} baseline={baseline}"
                        );
                    }
                };
                for warmup in 0..10 {
                    if before && v > 0 {
                        println!("WARMUP frontier cells={cells} {}/10", warmup + 1);
                    }
                    generated(&mut device, &plan, &pipes, &buffers, &rounds, before);
                    verify(&buffers, false);
                    baseline(&mut device, family, v, &pipe, &base);
                    verify(&base, true);
                    copy_time(&mut device, &buffers[0], &copy_out);
                }
                let mut times = Vec::new();
                let mut bases = Vec::new();
                let mut copies = Vec::new();
                for pair in 0..15 {
                    if before && v > 0 {
                        println!("ABBA frontier cells={cells} {}/15", pair + 1);
                    }
                    times.push(generated(
                        &mut device,
                        &plan,
                        &pipes,
                        &buffers,
                        &rounds,
                        before,
                    ));
                    verify(&buffers, false);
                    bases.push(baseline(&mut device, family, v, &pipe, &base));
                    verify(&base, true);
                    bases.push(baseline(&mut device, family, v, &pipe, &base));
                    verify(&base, true);
                    times.push(generated(
                        &mut device,
                        &plan,
                        &pipes,
                        &buffers,
                        &rounds,
                        before,
                    ));
                    verify(&buffers, false);
                    copies.push(copy_time(&mut device, &buffers[0], &copy_out));
                    copies.push(copy_time(&mut device, &buffers[0], &copy_out));
                }
                let ms = median(&times);
                let bm = median(&bases);
                let cm = median(&copies);
                let payload = if v > 0 {
                    (v as u64 * v as u64 + output as u64) * 4
                } else {
                    (cells as u64 * 2 + output as u64) * 4
                };
                println!(
                    "ROW\t{family}\t{cells}\t{v}\t{ms:.6}\t{bm:.6}\t{cm:.6}\t{:.6}\t{:.6}\t{compile_ms:.6}\t{base_compile_ms:.6}\t{:?}\t{}\t{:016x}",
                    cells as f64 * 8.0 / cm / 1e6,
                    payload as f64 / ms / 1e6,
                    if before {
                        before_dimensions(&plan)
                    } else {
                        ir::launch_dimensions(&plan)
                    },
                    ir::schedule(&plan).len(),
                    source_hash(&source)
                );
                println!(
                    "SAMPLES\t{family}\t{cells}\tinvar={times:?}\tbaseline={bases:?}\tcopy={copies:?}"
                );
            });
        }
    }
}

#[test]
#[ignore = "real Metal adversarial ordered masks, cycles, self edges and round truncation"]
fn issue_m4_05_metal_ordering() {
    // Profiles compile generated and baseline MSL: the developer source path.
    let mut device = MetalDevice::open_runtime_source().unwrap();
    let n = 7u32;
    let mut seed = 0x0405u32;
    for case in 0..32 {
        let mut edges = vec![0; n as usize * n as usize];
        for bit in &mut edges {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *bit = u32::from(seed >> 29 == 0);
        }
        // Force simultaneously active vertices with both directions and loops.
        for (u, v) in [
            (0, 1),
            (0, 2),
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (2, 6),
            (6, 0),
        ] {
            edges[u * n as usize + v] = 1;
        }
        for rounds in [0, 1, 2, 16] {
            let buf = |id, elements| Buffer {
                id,
                elements,
                address: AddressSpace::Device,
                layout: Layout::ContiguousU32,
            };
            let raw = ir::Ir {
                version: ir::VERSION,
                source_identity: 1,
                graph_revision: 0,
                source_contract: [1; 32],
                span: Span { start: 1, end: 2 },
                operation: ir::Operation::Frontier {
                    vertices: n,
                    rounds,
                    capacity: n,
                },
                values: buf(0, n * n),
                indices: buf(1, 0),
                output: buf(2, n * (rounds + 1)),
                law: None,
            };
            let bound = ir::verify(raw.clone(), &raw, device.caps)
                .unwrap()
                .bind(&edges, &[], 0)
                .unwrap();
            let mut expected = vec![0; n as usize * (rounds as usize + 1)];
            let mut front = vec![0; n as usize];
            front[0] = 1;
            for round in 0..rounds as usize {
                let mut next = vec![0; n as usize];
                for u in 0..n as usize {
                    if front[u] == 1 && expected[u] == 0 {
                        expected[u] = 1;
                        for v in 0..n as usize {
                            if edges[u * n as usize + v] == 1 && expected[v] == 0 {
                                next[v] = 1;
                            }
                        }
                    }
                }
                expected[(round + 1) * n as usize..(round + 2) * n as usize].copy_from_slice(&next);
                front = next;
            }
            let want = ir::decode(bound.plan(), expected, &[0, 0]).unwrap();
            let got = device.run_indexed(&bound).unwrap();
            assert_eq!(got.output, want, "case={case} rounds={rounds}");
            assert_eq!(got.completed_phases, rounds * 2);
        }
    }
    println!("128 adversarial frontier runs bit-exact, including every mask and stable map");
}
