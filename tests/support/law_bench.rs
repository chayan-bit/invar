//! P8 template-only D5 experiment. The handwritten baseline uses a separate
//! algebra and hierarchical three-pass scan, with no generated helper calls.
#![allow(dead_code)]
use super::support;
use invar::general_net::{
    law_collective::{self, Calibration, LawRepr, Layout, Residency, Route, cuda, msl},
    normal_form::Rep,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::PathBuf};

pub const MIN: &str = "fn f(xs:read Array<i32>,n:u32,seed:i32)->i32 {var m=seed; for i in 0u32..n {if xs[i]<m {m=xs[i];}} return m;}";
pub fn repr(kind: usize) -> LawRepr {
    match kind {
        0 => support::repr(support::SUM, |r| matches!(r, Rep::LiftedSum(_))),
        1 => support::repr(MIN, |r| matches!(r, Rep::Select { .. })),
        2 => support::repr(support::AFFINE, |r| matches!(r, Rep::Affine(_))),
        _ => support::repr(support::TROPICAL, |r| matches!(r, Rep::Tropical { .. })),
    }
}
pub fn name(kind: usize) -> &'static str {
    ["sum-reduce", "min-reduce", "affine-scan", "maxplus-scan"][kind]
}
pub fn data(kind: usize, n: u32) -> (Vec<u32>, Vec<u32>) {
    let words = if kind == 0 || kind == 3 { 6 } else { 2 };
    let mut input = Vec::with_capacity(n as usize * words);
    let mut output = Vec::with_capacity(if kind >= 2 { n as usize * words } else { words });
    let wide = |x: i64| [x as u32, ((x as u64) >> 32) as u32];
    let (mut sum, mut low, mut high) = (0i64, 0i64, 0i64);
    let (mut a, mut b) = (1u32, 0u32);
    let mut bound = 0i64;
    let mut min = i32::MAX;
    for i in 0..n {
        let x = (i.wrapping_mul(17) % 5) as i32 - 2;
        match kind {
            0 => {
                let (v, l, h) = (
                    wide(i64::from(x)),
                    wide(i64::from(x.min(0))),
                    wide(i64::from(x.max(0))),
                );
                input.extend([v[0], v[1], l[0], l[1], h[0], h[1]]);
                sum += i64::from(x);
                low = low.min(sum);
                high = high.max(sum);
            }
            1 => {
                input.extend([1, x as u32]);
                min = min.min(x);
            }
            2 => {
                output.extend([a, b]);
                let ai = 2 * (i % 5) + 1;
                let bi = i.wrapping_mul(13);
                input.extend([ai, bi]);
                a = ai.wrapping_mul(a);
                b = ai.wrapping_mul(b).wrapping_add(bi);
            }
            _ => {
                let (s, b, v) = (wide(sum), wide(bound), wide(i64::from(x)));
                output.extend([s[0], s[1], b[0], b[1], 1, u32::from(i != 0)]);
                input.extend([v[0], v[1], 0, 0, 1, 1]);
                sum += i64::from(x);
                bound = (bound + i64::from(x)).max(0);
            }
        }
    }
    if kind == 0 {
        output = [support::wide(sum), support::wide(low), support::wide(high)].concat();
    }
    if kind == 1 {
        output = vec![1, min as u32];
    }
    (input, output)
}

/// The native path of a refused reduction: one sequential host fold of the
/// same AoS words with the law's ordered join and identity.
pub fn native(kind: usize, input: &[u32]) -> Vec<u32> {
    let get = |e: &[u32], j: usize| (u64::from(e[j]) | (u64::from(e[j + 1]) << 32)) as i64;
    match kind {
        0 => {
            let (mut sum, mut low, mut high) = (0i64, 0i64, 0i64);
            for e in input.chunks_exact(6) {
                low = low.min(sum.wrapping_add(get(e, 2)));
                high = high.max(sum.wrapping_add(get(e, 4)));
                sum = sum.wrapping_add(get(e, 0));
            }
            [support::wide(sum), support::wide(low), support::wide(high)].concat()
        }
        1 => {
            let (mut some, mut min) = (0u32, 0u32);
            for e in input.chunks_exact(2) {
                if some == 0 {
                    (some, min) = (e[0], e[1]);
                } else if e[0] != 0 && (e[1] as i32) < (min as i32) {
                    min = e[1];
                }
            }
            vec![some, min]
        }
        _ => panic!("scans have no native route"),
    }
}

/// The recorded calibration of this device for one reduction kind.
pub fn calibration(kind: usize) -> Calibration {
    let path = std::env::var_os("P8_CALIBRATION").expect("recorded calibration file");
    let all: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    Calibration::from_json(&all[name(kind)]).unwrap()
}

/// Independent calibration passes per figure; the best decides the route.
pub const CALIBRATION_PASSES: usize = 5;

/// Measure and record the cost-model calibration of one device: several
/// independent passes, each a median of ten samples after three warmups.
pub fn calibrate(
    metal: bool,
    device: &str,
    mut transfer: impl FnMut(usize) -> Result<[f64; 2], String>,
) {
    const WORDS: usize = 1 << 26;
    let at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let gpu = if metal { "metal" } else { "cuda" };
    // Full-size input (the largest D5 length), cold before every sample:
    // streaming a buffer far larger than any cache evicts it, and under host
    // memory pressure its pages may be out, so page-in shows as it would in a
    // session. Generated once, outside every timed window.
    let data = [0, 1].map(|kind| data(kind, 1 << 26));
    let flush = vec![1u8; 256 << 20];
    let mut passes = [Vec::new(), Vec::new()];
    for _ in 0..CALIBRATION_PASSES {
        let [peak, link] = transfer(WORDS).unwrap();
        for (kind, (input, expected)) in data.iter().enumerate() {
            let mut samples = Vec::new();
            for sample in 0..13 {
                let evict: u64 = flush.iter().step_by(64).map(|&b| u64::from(b)).sum();
                std::hint::black_box(evict);
                let begun = std::time::Instant::now();
                let output = native(kind, input);
                let ms = begun.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(&output, expected);
                if sample >= 3 {
                    samples.push(ms);
                }
            }
            samples.sort_by(f64::total_cmp);
            let ms = (samples[4] + samples[5]) / 2.0;
            passes[kind].push(json!({
                "peak_bytes_per_ms": peak,
                "link_bytes_per_ms": link,
                "native_bytes_per_ms": (input.len() * 4) as f64 / ms,
            }));
        }
    }
    let mut record = serde_json::Map::new();
    for (kind, passes) in passes.into_iter().enumerate() {
        let pick = |key: &str, best: bool| {
            let all = passes.iter().map(|p| p[key].as_f64().unwrap());
            if best {
                all.fold(f64::MIN, f64::max)
            } else {
                all.fold(f64::MAX, f64::min)
            }
        };
        let figures = |best: bool| {
            json!({
                "peak_bytes_per_ms": pick("peak_bytes_per_ms", best),
                "link_bytes_per_ms": pick("link_bytes_per_ms", best),
                "native_bytes_per_ms": pick("native_bytes_per_ms", best),
            })
        };
        let mut entry = figures(true);
        entry["worst"] = figures(false);
        entry["passes"] = json!(passes);
        entry["source"] = json!(format!(
            "p8_calibrate_{gpu} on {device}, unix time {at}: {CALIBRATION_PASSES} independent passes; each pass has peak, the median device-to-device copy of 2^26 words (read plus write), link, the median warm-path staging of the same words ({}), and native, the median single-thread host fold of 2^26 {} elements (the largest D5 length, cache-cold: 256 MiB streamed before each sample), each a median of 10 samples after 3 warmups; the top-level figures are the best pass of each and `worst` the slowest",
            if metal {
                "no-copy wrap of page-aligned host pages plus the first GPU read of the fresh wrap, over a resident read"
            } else {
                "pageable host copy through pinned staging"
            },
            name(kind)
        ));
        record.insert(name(kind).into(), entry);
    }
    let out = PathBuf::from(std::env::var_os("P8_D5_OUT").expect("lane evidence directory"));
    fs::create_dir_all(&out).unwrap();
    let text = serde_json::to_string_pretty(&record).unwrap();
    fs::write(out.join(format!("calibration-{gpu}.json")), &text).unwrap();
    for kind in 0..2 {
        let c = Calibration::from_json(&record[name(kind)]).unwrap();
        let d = law_collective::decide(&repr(kind), false, 1 << 26, Residency::Host, &c);
        println!(
            "{} host route {:?} ratio {:?} marginal {}",
            name(kind),
            d.route,
            d.ratio,
            d.marginal
        );
    }
    println!("{text}");
}

pub fn sources(kind: usize, metal: bool) -> [String; 2] {
    let repr = repr(kind);
    let emit = if metal {
        if kind >= 2 {
            msl::emit_scan
        } else {
            msl::emit_reduce
        }
    } else if kind >= 2 {
        cuda::emit_scan
    } else {
        cuda::emit_reduce
    };
    let mut generated = emit(&repr, Layout::default()).unwrap().source;
    generated.push_str(&reset(metal));
    [generated, handwritten(kind, metal)]
}
fn signature(name: &str, metal: bool) -> String {
    if metal {
        format!(
            "kernel void {name}(device const uint* input [[buffer(0)]],device uint* output [[buffer(1)]],constant uint& n [[buffer(2)]],device uint* aggregates [[buffer(3)]],device uint* prefixes [[buffer(4)]],constant uint& admitted [[buffer(5)]],uint t [[thread_index_in_threadgroup]],uint g [[threadgroup_position_in_grid]]) {{"
        )
    } else {
        format!(
            "extern \"C\" __global__ void {name}(const uint* input,uint* output,uint n,uint* aggregates,uint* prefixes,uint admitted) {{uint t=threadIdx.x,g=blockIdx.x;"
        )
    }
}
fn reset(metal: bool) -> String {
    format!(
        "\n{} uint i=g*256u+t;if(i<n) {{aggregates[i]=0u;prefixes[i]=0u;}}}}\n",
        signature("invar_law_reset_v1", metal)
    )
}
pub fn handwritten(kind: usize, metal: bool) -> String {
    let w = if kind == 0 || kind == 3 { 6 } else { 2 };
    let head = if metal {
        "#include <metal_stdlib>\nusing namespace metal;\ntypedef long I;typedef ulong U;\n#define F inline\n#define B threadgroup_barrier(mem_flags::mem_threadgroup)\n#define SH threadgroup\n#define D device\n"
    } else {
        "typedef unsigned int uint;typedef long long I;typedef unsigned long long U;\n#define F __device__ __forceinline__\n#define B __syncthreads()\n#define SH __shared__\n#define D\n"
    };
    let mut s = format!(
        "{head}\nstruct H {{uint w[{w}];}};\nF H zero(){{H h;for(uint j=0;j<{w};++j)h.w[j]=0;return h;}}\nF H load(D const uint* p,uint i){{H h;for(uint j=0;j<{w};++j)h.w[j]=p[i*{w}+j];return h;}}\nF void store(D uint* p,uint i,H h){{for(uint j=0;j<{w};++j)p[i*{w}+j]=h.w[j];}}\nF I get(H h,uint j){{return I(U(h.w[j])|(U(h.w[j+1])<<32));}}\nF H put(H h,uint j,I x){{h.w[j]=uint(U(x));h.w[j+1]=uint(U(x)>>32);return h;}}\n"
    );
    let id = match kind {
        2 => "h.w[0]=1;",
        3 => "h.w[4]=1;",
        _ => "",
    };
    s.push_str(&format!("F H identity(){{H h=zero();{id}return h;}}\n"));
    let op = match kind {
        0 => {
            "H h=zero();I v=get(a,0),lo=v+get(b,2),hi=v+get(b,4);h=put(h,0,v+get(b,0));h=put(h,2,get(a,2)<lo?get(a,2):lo);return put(h,4,get(a,4)>hi?get(a,4):hi);"
        }
        1 => "if(!a.w[0])return b;if(!b.w[0])return a;return int(b.w[1])<int(a.w[1])?b:a;",
        2 => "H h;h.w[0]=b.w[0]*a.w[0];h.w[1]=b.w[0]*a.w[1]+b.w[1];return h;",
        _ => {
            "H h=zero();h.w[4]=a.w[4]&b.w[4];if(h.w[4])h=put(h,0,get(a,0)+get(b,0));bool shifted=a.w[5]&&b.w[4];h.w[5]=uint(shifted||b.w[5]);I v=get(a,2)+get(b,0);if(shifted&&b.w[5])v=v>get(b,2)?v:get(b,2);else if(b.w[5])v=get(b,2);if(h.w[5])h=put(h,2,v);return h;"
        }
    };
    s.push_str(&format!("F H join(H a,H b){{{op}}}\n"));
    let up = "for(uint d=1;d<256;d*=2){uint k=(t+1)*2*d-1;if(k<256) shared[k]=join(shared[k-d],shared[k]);B;}";
    let down = "if(t==0)shared[255]=identity();B;for(uint d=128;d>0;d/=2){uint k=(t+1)*2*d-1;if(k<256){H x=shared[k-d];shared[k-d]=shared[k];shared[k]=join(shared[k],x);}B;}";
    s.push_str(&signature("invar_law_reduce_v1", metal));
    s.push_str(&format!("SH H shared[256];H v=identity();uint start=g*1024+t*4;for(uint j=0;j<4&&start+j<n;++j)v=join(v,load(input,start+j));shared[t]=v;B;{up}if(t==0)store(output,g,shared[255]);}}\n"));
    if kind >= 2 {
        s.push_str(&signature("invar_law_prefix_v1", metal));
        s.push_str(&format!("SH H shared[256];uint items=(n+255)/256,start=t*items;H v=identity();for(uint j=0;j<items&&start+j<n;++j)v=join(v,load(input,start+j));shared[t]=v;B;{up}{down}v=shared[t];for(uint j=0;j<items&&start+j<n;++j){{store(output,start+j,v);v=join(v,load(input,start+j));}}}}\n"));
        s.push_str(&signature("invar_law_scan_v1", metal));
        s.push_str(&format!("SH H shared[256];H v=identity();uint start=g*1024+t*4;for(uint j=0;j<4&&start+j<n;++j)v=join(v,load(input,start+j));shared[t]=v;B;{up}{down}v=join(load(prefixes,g),shared[t]);for(uint j=0;j<4&&start+j<n;++j){{store(output,start+j,v);v=join(v,load(input,start+j));}}}}\n"));
    }
    s
}

/// The native executor a run uses for generated warm.
pub type Native<'a> = Option<&'a dyn Fn(&[u32]) -> Vec<u32>>;

/// Pre-registered independent process sessions per device: 10 Metal and 3
/// CUDA, unless a registered run sets `P8_METAL_SESSIONS`/`P8_CUDA_SESSIONS`.
pub fn sessions(metal: bool) -> usize {
    let key = if metal {
        "P8_METAL_SESSIONS"
    } else {
        "P8_CUDA_SESSIONS"
    };
    std::env::var(key).map_or(if metal { 10 } else { 3 }, |n| {
        n.parse()
            .ok()
            .filter(|n| *n > 0)
            .expect("positive session count")
    })
}

pub fn session(
    metal: bool,
    mut run: impl FnMut(
        [&str; 2],
        &str,
        Vec<u32>,
        &[u32],
        u32,
        u32,
        bool,
        Native,
    ) -> Result<Vec<[f64; 4]>, String>,
) {
    let session: usize = std::env::var("P8_D5_SESSION")
        .expect("session index")
        .parse()
        .unwrap();
    assert!(session < sessions(metal));
    let out = PathBuf::from(std::env::var_os("P8_D5_OUT").expect("lane evidence directory"));
    fs::create_dir_all(&out).unwrap();
    let path = out.join(format!(
        "{}-{session}.jsonl",
        if metal { "metal" } else { "cuda" }
    ));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    for kind in 0..4 {
        let sources = sources(kind, metal);
        let repr = repr(kind);
        let words = repr.word_count();
        let scan = kind >= 2;
        let calibration = (!scan).then(|| calibration(kind));
        let hashes = sources.each_ref().map(|s| {
            Sha256::digest(s.as_bytes())
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        });
        for power in 20..=26 {
            let n = 1u32 << power;
            // Freeze storage refusal before allocation. Exact capacities keep
            // host staging below the shared 12 GiB job ceiling.
            let storage = (u64::from(n) * u64::from(words) * 4) * (if scan { 2 } else { 1 })
                + u64::from(n.div_ceil(1024)) * u64::from(words) * 16;
            // Resident input lives on the device and warm input on the host.
            let decisions = [Residency::Device, Residency::Host].map(|residency| {
                calibration
                    .as_ref()
                    .map(|c| law_collective::decide(&repr, scan, n, residency, c))
            });
            let routes = decisions.map(|d| d.map_or(Route::Collective, |d| d.route));
            let host = decisions[1];
            assert_eq!(routes[0], Route::Collective);
            let folds = std::cell::Cell::new(0usize);
            let fold = |input: &[u32]| {
                folds.set(folds.get() + 1);
                native(kind, input)
            };
            let native: Native = match routes[1] {
                Route::Native(_) => Some(&fold),
                Route::Collective => None,
            };
            let result = if storage > invar_device::MAX_GNET_STORAGE_BYTES as u64 {
                Err("benchmark storage exceeds 8 GiB".into())
            } else {
                let (input, expected) = data(kind, n);
                run(
                    [&sources[0], &sources[1]],
                    if scan {
                        "invar_law_scan_v1"
                    } else {
                        "invar_law_reduce_v1"
                    },
                    input,
                    &expected,
                    words,
                    n,
                    scan,
                    native,
                )
            };
            // The refusal is taken: every generated warm call ran the native fold.
            if result.is_ok() {
                assert_eq!(folds.get(), if native.is_some() { 40 } else { 0 });
            }
            let warm_route = match routes[1] {
                Route::Native(_) => "native",
                Route::Collective => "collective",
            };
            let record = match result {
                Ok(raw) => json!({"samples_ms":raw}),
                Err(reason) => json!({"unavailable":reason}),
            };
            writeln!(file,"{}",json!({"kind":name(kind),"length":n,"session":session,"flat_limits":[1.0,1.05],"source_sha256":hashes,"warm_route":warm_route,"warm_ratio":host.and_then(|d| d.ratio),"warm_marginal":host.is_some_and(|d| d.marginal),"calibration":calibration.as_ref().map(|c| c.source.clone()),"result":record})).unwrap();
            file.flush().unwrap();
            println!(
                "P8 D5 {} 2^{power}: {}",
                name(kind),
                if record.get("unavailable").is_some() {
                    "unavailable"
                } else {
                    "30 pairs checked"
                }
            );
            println!("P8 D5 {} 2^{power} warm route: {warm_route}", name(kind));
        }
    }
}
