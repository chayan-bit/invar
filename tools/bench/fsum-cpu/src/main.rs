//! CPU benchmark and calibration for the v0.2 fsum lane.
//!
//! `cargo run --release -- bench WORKERS`: medians of 30 runs (after 5
//! warmups) of a serial fold, rayon `par_iter().sum()` on a pool of WORKERS
//! threads, and Invar's `sum_tree`/`sum_exact` as placed by the cost model.
//! `cargo run --release -- calibrate`: the coefficients recorded in
//! `calibration/fsum-v1.json` and the measured parallel crossovers.
//! Every line ends with the host's one-minute load average.

use std::time::Instant;

use invar::{
    front::Span, fsum::Kind, mir_eval::ReduceCall, numeric_contract::F32Bits, schedule::reductions,
};
use rayon::prelude::*;

const KINDS: [Kind; 4] = [Kind::SumTree, Kind::DotTree, Kind::SumExact, Kind::DotExact];

fn load() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|text| text.split_whitespace().next().map(str::to_owned))
        .unwrap_or_else(|| "na".into())
}

fn median_ms(samples: usize, mut run: impl FnMut() -> u32) -> f64 {
    let mut times = (0..samples + 5)
        .map(|_| {
            let start = Instant::now();
            std::hint::black_box(run());
            start.elapsed().as_secs_f64() * 1000.0
        })
        .skip(5)
        .collect::<Vec<_>>();
    times.sort_by(f64::total_cmp);
    times[samples / 2]
}

fn words(n: usize, state: &mut u64) -> Vec<F32Bits> {
    (0..n)
        .map(|_| {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            let w = (*state as u32 & 0x807f_ffff) | ((120 + (*state >> 40) as u32 % 16) << 23);
            F32Bits::from_bits(w)
        })
        .collect()
}

fn call(kind: Kind, a: &[F32Bits]) -> ReduceCall<'_> {
    let dot = matches!(kind, Kind::DotTree | Kind::DotExact);
    ReduceCall {
        function: "bench",
        span: Span { start: 0, end: 0 },
        kind,
        a,
        b: dot.then_some(a),
    }
}

fn on(kind: Kind, a: &[F32Bits], workers: usize) -> u32 {
    reductions::cpu_on(&call(kind, a), workers)
        .unwrap()
        .to_bits()
}

fn bench(workers: usize) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap();
    let mut state = 0x9e37_79b9_u64;
    for log2 in [20u32, 24, 26] {
        let bits = words(1 << log2, &mut state);
        let floats = bits
            .iter()
            .map(|w| f32::from_bits(w.to_bits()))
            .collect::<Vec<_>>();
        let model = |kind| {
            reductions::cpu(&call(kind, &bits), workers)
                .unwrap()
                .to_bits()
        };
        let serial = median_ms(30, || {
            floats.iter().fold(0.0f32, |acc, &x| acc + x).to_bits()
        });
        let rayon = median_ms(30, || {
            pool.install(|| floats.par_iter().sum::<f32>()).to_bits()
        });
        let tree = median_ms(30, || model(Kind::SumTree));
        let exact = median_ms(30, || model(Kind::SumExact));
        let picked = |kind| reductions::cpu_workers(kind, bits.len(), workers);
        println!(
            "cpu n=2^{log2} workers={workers} tree_workers={} exact_workers={} serial_ms={serial:.3} rayon_ms={rayon:.3} sum_tree_ms={tree:.3} sum_exact_ms={exact:.3} tree/rayon={:.3} exact/rayon={:.3} load={}",
            picked(Kind::SumTree),
            picked(Kind::SumExact),
            tree / rayon,
            exact / rayon,
            load()
        );
    }
}

/// Start and join `workers` empty scoped threads, per worker, in nanoseconds.
fn spawn_ns(workers: usize) -> f64 {
    let ms = median_ms(30, || {
        std::thread::scope(|scope| {
            let handles = (0..workers)
                .map(|_| scope.spawn(|| 0u32))
                .collect::<Vec<_>>();
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        })
    });
    ms * 1e6 / workers as f64
}

fn calibrate() {
    let mut state = 0x5eed_u64;
    let big = words(1 << 22, &mut state);
    for kind in KINDS {
        let ns = median_ms(15, || on(kind, &big, 1)) * 1e6 / big.len() as f64;
        println!(
            "calibrate {} ns_per_term={ns:.4} load={}",
            reductions::family_id(kind),
            load()
        );
    }
    for workers in [2, 4, 8, 16, 32] {
        println!(
            "calibrate spawn workers={workers} spawn_ns_per_worker={:.0} load={}",
            spawn_ns(workers),
            load()
        );
    }
    for kind in KINDS {
        for workers in [4, 16] {
            crossover(kind, workers, &mut state);
        }
    }
}

/// Smallest power of two from which `workers` threads beat one, measured,
/// next to the size where the recorded model switches.
fn crossover(kind: Kind, workers: usize, state: &mut u64) {
    let measured = (8..=24).find(|&log2| {
        let a = words(1 << log2, state);
        median_ms(15, || on(kind, &a, workers)) < median_ms(15, || on(kind, &a, 1))
    });
    let model = (8..=26).find(|&log2| reductions::cpu_workers(kind, 1 << log2, workers) > 1);
    println!(
        "crossover {} workers={workers} measured=2^{} model=2^{} load={}",
        reductions::family_id(kind),
        measured.map_or("none".into(), |x| x.to_string()),
        model.map_or("none".into(), |x| x.to_string()),
        load()
    );
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    match args.get(1).map(String::as_str) {
        Some("bench") => bench(args.get(2).map_or(4, |w| w.parse().expect("workers"))),
        Some("calibrate") => calibrate(),
        _ => eprintln!("usage: fsum-cpu-bench bench WORKERS | calibrate"),
    }
}
