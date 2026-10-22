// Host harness for `p0a_trig_exhaustive_native`, compiled by rustc beside
// the LLVM helper object. The test appends `mod trig;` pointing at
// `crates/invar-device/src/trig_f32.rs`, so the expected bits come from the
// same single source of truth the object was generated from.
// Usage: harness FIRST_BLOCK BLOCKS THREADS; one line per 2^24-input block:
// "block <index> <sin> <cos> <sqrt> mismatches <n> first <x>".

unsafe extern "C" {
    fn invar_trig_sweep_sin(x: u32) -> u32;
    fn invar_trig_sweep_cos(x: u32) -> u32;
    fn invar_trig_sweep_sqrt(x: u32) -> u32;
}

fn native(f: usize, x: u32) -> u32 {
    // SAFETY: the helpers are pure functions of one word with no memory
    // arguments, defined by the linked object with the C ABI declared here.
    unsafe {
        match f {
            0 => invar_trig_sweep_sin(x),
            1 => invar_trig_sweep_cos(x),
            _ => invar_trig_sweep_sqrt(x),
        }
    }
}

fn host(f: usize, x: u32) -> u32 {
    match f {
        0 => trig::host_sin(x),
        1 => trig::host_cos(x),
        _ => trig::host_sqrt(x),
    }
}

fn fnv(hash: u64, word: u32) -> u64 {
    word.to_le_bytes()
        .iter()
        .fold(hash, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3))
}

fn block(index: u32) -> String {
    let mut digests = [0xcbf2_9ce4_8422_2325u64; 3];
    let (mut mismatches, mut first) = (0u64, 0u32);
    for offset in 0..1u32 << 24 {
        let x = index << 24 | offset;
        for (f, digest) in digests.iter_mut().enumerate() {
            let got = native(f, x);
            *digest = fnv(*digest, got);
            if got != host(f, x) {
                if mismatches == 0 {
                    first = x;
                }
                mismatches += 1;
            }
        }
    }
    format!(
        "block {index} {:016x} {:016x} {:016x} mismatches {mismatches} first {first:#010x}",
        digests[0], digests[1], digests[2]
    )
}

fn main() {
    let args: Vec<u32> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("numeric argument"))
        .collect();
    let [start, count, threads] = args[..] else {
        panic!("usage: harness FIRST_BLOCK BLOCKS THREADS");
    };
    let end = (start + count).min(256);
    let next = std::sync::atomic::AtomicU32::new(start);
    let mut lines: Vec<(u32, String)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads.max(1))
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if index >= end {
                            return done;
                        }
                        done.push((index, block(index)));
                    }
                })
            })
            .collect();
        workers.into_iter().flat_map(|w| w.join().unwrap()).collect()
    });
    lines.sort();
    for (_, line) in lines {
        println!("{line}");
    }
}
