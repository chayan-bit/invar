#![allow(dead_code)]
use invar::{
    eval,
    general_net::{
        law_collective::{Emission, LawRepr, Layout, cuda},
        normal_form::{self, Rep},
    },
};
use std::{
    cell::RefCell,
    path::PathBuf,
    process::{Command, Stdio},
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};

#[path = "normal_form.rs"]
mod normal;
pub fn compile(source: &str) -> (invar::general_net::Net, invar::general_net::Proof) {
    normal::compile(source)
}
pub fn repr(source: &str, predicate: impl Fn(&Rep) -> bool) -> LawRepr {
    let (net, proof) = normal::compile(source);
    let form = normal_form::extract(&net, &proof).unwrap();
    normal_form::certificate::check_all(&net, &proof, &form).unwrap();
    let cert = form
        .certificates()
        .find(|c| c.rep.as_ref().is_some_and(&predicate))
        .unwrap_or_else(|| panic!("no collective in {form:?}"));
    LawRepr::from_certificate(&net, &proof, cert).unwrap()
}
pub fn wide(x: i64) -> Vec<u32> {
    let x = x as u64;
    vec![x as u32, (x >> 32) as u32]
}
pub fn read_wide(w: &[u32]) -> i64 {
    (u64::from(w[0]) | (u64::from(w[1]) << 32)) as i64
}
pub fn lifted(x: i32) -> Vec<u32> {
    let x = i64::from(x);
    [wide(x), wide(x.min(0)), wide(x.max(0))].concat()
}
pub fn product(x: i32) -> Vec<u32> {
    [wide(i64::from(x)), wide(i64::from(x).abs().max(1))].concat()
}
pub fn seed(words: u32, x: i64) -> Vec<u32> {
    let mut s = vec![0; words as usize];
    if words == 1 {
        s[0] = x as u32;
    } else {
        s[..2].copy_from_slice(&wide(x));
    }
    s
}
pub fn array_i32(xs: &[i32]) -> eval::Value {
    eval::Value::ArrayI32(Rc::new(RefCell::new(xs.to_vec())))
}
pub fn array_u32(xs: &[u32]) -> eval::Value {
    eval::Value::ArrayU32(Rc::new(RefCell::new(xs.to_vec())))
}

pub const SUM: &str = "fn f(xs: read Array<i32>, n: u32, seed: i32) -> i32 {var s=seed; for i in 0u32..n {s=s+xs[i];} return s;}";
pub const PRODUCT: &str = "fn f(xs: read Array<i32>, n: u32, seed: i32) -> i32 {var s=seed; for i in 0u32..n {s=s*xs[i];} return s;}";
pub const AFFINE: &str = "fn f(xs: read Array<u32>, ys: read Array<u32>, n: u32, seed: u32) -> u32 {var s=seed; for i in 0u32..n {s=wrapping_add(wrapping_mul(s,xs[i]),ys[i]);} return s;}";
pub const RESIDUE: &str = "fn f(xs: read Array<i32>, ys: read Array<i32>, n: u32, seed: i32) -> i32 {var s=seed; for i in 0u32..n {s=(s*xs[i]+ys[i])%10007;} return s;}";
pub const TROPICAL: &str = "fn f(xs: read Array<i32>, n: u32, seed: i32) -> i32 {var s=seed; for i in 0u32..n {s=s+xs[i]; if s<0 {s=0;}} return s;}";
pub const SELECT: &str = "fn f(xs: read Array<f32>, n: u32, seed: f32) -> u32 {var m=seed; var at=4294967295u32; for i in 0u32..n {if xs[i]<m {m=xs[i]; at=i;}} return at;}";

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    pub fn int(&mut self) -> i32 {
        match self.next() % 6 {
            0 => i32::MAX,
            1 => i32::MIN,
            2 => 0,
            _ => (self.next() % 21) as i32 - 10,
        }
    }
}

/// Compile the *emitted* CUDA algebra as ordinary C++ on staging. This proves
/// target arithmetic, not CUDA scheduling, compilation or hardware acceptance.
pub struct CpuAlgebra {
    dir: PathBuf,
    pub emission: Emission,
}
impl CpuAlgebra {
    pub fn new(repr: &LawRepr, mutation: Option<(&str, &str)>) -> Self {
        static SERIAL: AtomicU64 = AtomicU64::new(0);
        let base = std::env::var_os("TMPDIR")
            .map(PathBuf::from)
            .expect("TMPDIR required");
        let dir = base.join(format!(
            "p8-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        let emission = cuda::emit_scan(repr, Layout::default()).unwrap();
        let mut source = emission.source.clone();
        source.push_str(DRIVER);
        if let Some((from, to)) = mutation {
            assert!(source.contains(from), "mutation target missing: {from}");
            source = source.replacen(from, to, 1);
        }
        std::fs::write(dir.join("algebra.cpp"), source).unwrap();
        let result = Command::new("c++")
            .args([
                "-std=c++17",
                "-O1",
                "-DP8_CPU",
                "algebra.cpp",
                "-o",
                "algebra",
            ])
            .current_dir(&dir)
            .output()
            .expect("C++ compiler required on CPU host");
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        Self { dir, emission }
    }
    pub fn run(&self, cases: &[(Vec<u32>, Vec<Vec<u32>>)]) -> Vec<Vec<u32>> {
        use std::fmt::Write;
        let mut data = String::new();
        for (s, xs) in cases {
            write!(data, "{} ", xs.len()).unwrap();
            for x in s.iter().chain(xs.iter().flatten()) {
                write!(data, "{x} ").unwrap();
            }
            data.push('\n');
        }
        let input = self.dir.join("input.txt");
        let output = self.dir.join("output.txt");
        std::fs::write(&input, data).unwrap();
        let status = Command::new(self.dir.join("algebra"))
            .stdin(Stdio::from(std::fs::File::open(input).unwrap()))
            .stdout(Stdio::from(std::fs::File::create(&output).unwrap()))
            .status()
            .unwrap();
        assert!(status.success());
        std::fs::read_to_string(output)
            .unwrap()
            .lines()
            .map(|line| {
                line.split_whitespace()
                    .map(|x| x.parse().unwrap())
                    .collect()
            })
            .collect()
    }
}
impl Drop for CpuAlgebra {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.dir).unwrap();
    }
}
const DRIVER: &str = r#"
#include <iostream>
#include <vector>
#include <functional>
P8 tree(const std::vector<P8>& a,size_t lo,size_t hi) {
 if(lo==hi)return p8_id(); if(hi-lo==1)return p8_then(p8_id(),a[lo]);
 size_t mid=lo+(hi-lo)/2;return p8_then(tree(a,lo,mid),tree(a,mid,hi));
}
void scan(const std::vector<P8>& a,size_t lo,size_t hi,P8 offset,std::vector<P8>& out) {
 if(lo==hi)return; if(hi-lo==1){out[lo]=offset;return;}
 size_t mid=lo+(hi-lo)/2;scan(a,lo,mid,offset,out);scan(a,mid,hi,p8_then(offset,tree(a,lo,mid)),out);
}
int main() {
 size_t n;
 while(std::cin>>n) {
  P8 seed=p8_zero(); for(uint& w:seed.w)std::cin>>w;
  std::vector<P8>a(n),prefix(n);
  for(P8& x:a)for(uint& w:x.w)std::cin>>w;
  P8 total=tree(a,0,n);scan(a,0,n,p8_id(),prefix);
  P8 applied=p8_apply(total,seed,uint(n));
  for(uint w:applied.w)std::cout<<w<<' ';
  uint first=uint(n);
  for(uint i=0;i<n;++i)if(!p8_guard(p8_then(prefix[i],a[i]),seed)){first=i;break;}
  std::cout<<uint(p8_guard(total,seed))<<' '<<first;
  for(P8 p:prefix)for(uint w:p.w)std::cout<<' '<<w;
  for(size_t i=0;i<prefix.size();++i){P8 v=p8_apply(prefix[i],seed,uint(i));for(uint w:v.w)std::cout<<' '<<w;}
  std::cout<<'\n';
 }
}
"#;

pub fn oracle_result(
    net: &invar::general_net::Net,
    proof: &invar::general_net::Proof,
    args: &[eval::Value],
) -> Result<eval::Value, invar::eval::EvalError> {
    invar::general_net::oracle::observe_call(
        net,
        proof,
        "f",
        args,
        invar::general_net::oracle::Limits {
            fuel: 2_000_000,
            transitions: 20_000_000,
            trace_events: 1 << 20,
            array_elements: 1 << 20,
        },
    )
    .unwrap()
    .semantic
    .result
}

/// Layouts for hardware tests: one warp, eight warps (the D5 layout), and a
/// run of 3*2 words that is not a multiple of four, which keeps scalar loads.
pub const HARDWARE_LAYOUTS: [Layout; 3] = [
    Layout {
        lanes: 32,
        items_per_lane: 4,
    },
    Layout {
        lanes: 256,
        items_per_lane: 4,
    },
    Layout {
        lanes: 64,
        items_per_lane: 3,
    },
];

/// Named mutations of the vector, warp-shuffle and reduction fast paths.
/// Each must change the affine output at the D5 layout.
pub const FAST_PATH_MUTATIONS: [(&str, &str, &str); 4] = [
    ("vector-unpack", "raw[4u*k+1u]=x.y;", "raw[4u*k+1u]=x.x;"),
    (
        "vector-pack",
        "raw[j*P8_WORDS+w]=x.w[w];",
        "raw[j*P8_WORDS+w]=x.w[0];",
    ),
    (
        "warp-scan-offset",
        "if(lane>=d) v=p8_then(b,v);",
        "if(lane>d) v=p8_then(b,v);",
    ),
    (
        "reduce-order",
        "P8 b=p8_down(value,d); value=p8_then(value,b);",
        "P8 b=p8_down(value,d); value=p8_then(b,value);",
    ),
];

/// Wrapping affine pairs and their exclusive prefixes, as in the D5 template.
pub fn affine_case(n: u32) -> (Vec<u32>, Vec<u32>) {
    let (mut input, mut expected) = (Vec::new(), Vec::new());
    let (mut a, mut b) = (1u32, 0u32);
    for i in 0..n {
        let ai = (i % 5) * 2 + 1;
        let bi = i.wrapping_mul(13);
        expected.extend([a, b]);
        input.extend([ai, bi]);
        a = ai.wrapping_mul(a);
        b = ai.wrapping_mul(b).wrapping_add(bi);
    }
    (input, expected)
}
