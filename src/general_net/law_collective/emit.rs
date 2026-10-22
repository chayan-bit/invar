//! Target spelling is separate from the shared ordered collective algorithms.
use super::*;
use crate::laws::Law;
use normal_form::{IntTy, Order, Rep, Ring, ScalarTy};

#[derive(Clone, Copy)]
pub(super) enum Target {
    Metal,
    Cuda,
}

pub(super) fn emit(
    repr: &LawRepr,
    layout: Layout,
    scan: bool,
    target: Target,
) -> Result<Emission, Error> {
    validate(repr, layout)?;
    let words = repr.words;
    let tile = layout.lanes * layout.items_per_lane;
    let mut source = match target {
        Target::Metal => "#include <metal_stdlib>\nusing namespace metal;\ntypedef long p8_i64;\ntypedef ulong p8_u64;\n#define P8_INLINE inline\n".to_owned(),
        Target::Cuda => "#ifdef P8_CPU\n#include <cstdint>\n#define P8_INLINE inline\n#else\n#define P8_INLINE __device__ __forceinline__\n#endif\ntypedef unsigned int uint;\ntypedef unsigned long long p8_u64;\ntypedef long long p8_i64;\n".to_owned(),
    };
    source.push_str(&format!("#define P8_WORDS {words}u\n#define P8_LANES {}u\n#define P8_ITEMS {}u\n#define P8_TILE {tile}u\n", layout.lanes, layout.items_per_lane));
    source.push_str("#define P8_REVERSE 0u\n#define P8_USE_PREFIXES 1u\n#define P8_POLLS 128u\n");
    // 32-bit word offsets are exact when every in-bounds offset fits; see `narrow_index`.
    let index = if narrow_index(words, tile) {
        "uint"
    } else {
        "p8_u64"
    };
    let vector = u32::from((layout.items_per_lane * words).is_multiple_of(4));
    source.push_str(&format!(
        "typedef {index} p8_ix;\n#define P8_VEC {vector}u\n#define P8_WARPS {}u\n",
        layout.lanes / 32
    ));
    source.push_str(COMMON);
    source.push_str(&operations(repr));
    source.push_str("\n#ifndef P8_CPU\n");
    match target {
        Target::Metal => source.push_str(METAL_IO),
        Target::Cuda => source.push_str(CUDA_IO),
    }
    let entry = if scan {
        "invar_law_scan_v1"
    } else {
        "invar_law_reduce_v1"
    };
    let signature = match target {
        Target::Metal => format!(
            "kernel void {entry}(device const uint* input [[buffer(0)]], device uint* output [[buffer(1)]], constant uint& n [[buffer(2)]], device atomic_uint* aggregates [[buffer(3)]], device atomic_uint* prefixes [[buffer(4)]], constant uint& admitted [[buffer(5)]], uint t [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {{\nthreadgroup P8 shared[P8_LANES];\nthreadgroup P8 carry;\nthreadgroup uint remaining, kind;\n"
        ),
        Target::Cuda => format!(
            "extern \"C\" __global__ void {entry}(const uint* __restrict__ input, uint* __restrict__ output, uint n, uint* aggregates, uint* prefixes, uint admitted) {{\nuint t=threadIdx.x, group=blockIdx.x;\n__shared__ P8 shared[P8_LANES];\n__shared__ P8 carry;\n__shared__ uint remaining, kind;\n"
        ),
    };
    source.push_str(&signature);
    source.push_str(PROLOGUE);
    if scan {
        source.push_str(SCAN);
    } else {
        source.push_str(REDUCE);
    }
    source.push_str("}\n");
    if scan && repr.trap_recovery() == TrapRecovery::PrefixBounds {
        let locate = match target {
            Target::Metal => {
                "kernel void invar_law_locate_v1(device const uint* input [[buffer(0)]], device const uint* exclusive [[buffer(1)]], constant uint& n [[buffer(2)]], device const uint* seed [[buffer(3)]], device atomic_uint* earliest [[buffer(4)]], constant uint& admitted [[buffer(5)]], uint i [[thread_position_in_grid]]) { if(!admitted || i>=n || n>67108864u) return; P8 p=p8_then(p8_load(exclusive,i),p8_load(input,i)); if(!p8_guard(p,p8_load(seed,0u))) atomic_fetch_min_explicit(earliest,i,memory_order_relaxed); }\n"
            }
            Target::Cuda => {
                "extern \"C\" __global__ void invar_law_locate_v1(const uint* input,const uint* exclusive,uint n,const uint* seed,uint* earliest,uint admitted) { uint i=blockIdx.x*blockDim.x+threadIdx.x; if(!admitted || i>=n || n>67108864u) return; P8 p=p8_then(p8_load(exclusive,i),p8_load(input,i)); if(!p8_guard(p,p8_load(seed,0u))) atomicMin(earliest,i); }\n"
            }
        };
        source.push_str(locate);
    }
    source.push_str("#endif\n");
    Ok(Emission {
        source,
        entry,
        words,
        tile_elements: tile,
        threadgroup_bytes: (layout.lanes + 1) * words * 4 + 8,
    })
}

const COMMON: &str = r#"
struct P8 { uint w[P8_WORDS]; };
P8_INLINE P8 p8_zero() { P8 r; for(uint i=0u;i<P8_WORDS;++i) r.w[i]=0u; return r; }
P8_INLINE p8_u64 p8_get(P8 r,uint i) { return p8_u64(r.w[i]) | (p8_u64(r.w[i+1u])<<32u); }
P8_INLINE P8 p8_put(P8 r,uint i,p8_u64 x) { r.w[i]=uint(x); r.w[i+1u]=uint(x>>32u); return r; }
P8_INLINE p8_i64 p8_signed(p8_u64 x) { return (p8_i64)x; }
P8_INLINE p8_u64 p8_abs(p8_i64 x) { return x<0 ? 0ul-p8_u64(x) : p8_u64(x); }
P8_INLINE p8_u64 p8_min(p8_u64 a,p8_u64 b) { return a<b?a:b; }
P8_INLINE p8_u64 p8_max(p8_u64 a,p8_u64 b) { return a>b?a:b; }
P8_INLINE p8_i64 p8_smin(p8_i64 a,p8_i64 b) { return a<b?a:b; }
P8_INLINE p8_i64 p8_smax(p8_i64 a,p8_i64 b) { return a>b?a:b; }
P8_INLINE bool p8_nan(uint x) { return (x&0x7fffffffu)>0x7f800000u; }
// Ordered-bit comparison avoids target denormal flushing and preserves payloads.
P8_INLINE bool p8_flt(uint a,uint b) {
 if ((a&0x7fffffffu)==0u && (b&0x7fffffffu)==0u) return false;
 uint ka=(a&0x80000000u)?~a:(a^0x80000000u);
 uint kb=(b&0x80000000u)?~b:(b^0x80000000u);
 return ka<kb;
}
P8_INLINE p8_u64 p8_satmul(p8_u64 a,p8_u64 b) {
 const p8_u64 cap=8589934592ul;
 return b!=0ul && a>cap/b ? cap : p8_min(a*b,cap);
}
P8_INLINE p8_i64 p8_product(p8_i64 a,p8_i64 b) {
 p8_u64 m=p8_satmul(p8_abs(a),p8_abs(b));
 return (p8_i64)(((a<0)!=(b<0)) ? 0ul-m : m);
}
"#;

fn range(ty: IntTy, expression: &str) -> String {
    match ty {
        IntTy::I32 => format!("({expression})>=(-2147483647l-1l) && ({expression})<=2147483647l"),
        IntTy::U32 => format!("({expression})>=0l && p8_u64({expression})<=4294967295ul"),
    }
}

fn operations(repr: &LawRepr) -> String {
    let mut id = "P8 r=p8_zero();".to_owned();
    let mut join = "P8 r=p8_zero();".to_owned();
    let mut apply = "P8 r=p8_zero();".to_owned();
    let mut guard = "return true;".to_owned();
    match &repr.representation {
        Representation::ResidueSum { modulus } => {
            join.push_str(&format!(
                "r.w[0]=uint((p8_u64(a.w[0])+b.w[0])%{modulus}ul);"
            ));
            apply.push_str(&format!(
                "r.w[0]=uint((p8_u64(p.w[0])+s.w[0])%{modulus}ul);"
            ));
        }
        Representation::Carried(rep) => match rep {
            Rep::Sealed(law) => {
                let (op, identity) = match law {
                    Law::WrappingSum => ("+", 0),
                    Law::WrappingProduct => ("*", 1),
                };
                id.push_str(&format!("r.w[0]={identity}u;"));
                join.push_str(&format!("r.w[0]=a.w[0]{op}b.w[0];"));
                apply.push_str(&format!("r.w[0]=s.w[0]{op}p.w[0];"));
            }
            Rep::LiftedSum(ty) => {
                join.push_str("p8_u64 at=p8_get(a,0u); r=p8_put(r,0u,at+p8_get(b,0u)); r=p8_put(r,2u,p8_u64(p8_smin(p8_signed(p8_get(a,2u)),p8_signed(at+p8_get(b,2u))))); r=p8_put(r,4u,p8_u64(p8_smax(p8_signed(p8_get(a,4u)),p8_signed(at+p8_get(b,4u)))));");
                apply.push_str("r=p8_put(r,0u,p8_get(s,0u)+p8_get(p,0u));");
                guard = format!(
                    "p8_i64 low=p8_signed(p8_get(s,0u)+p8_get(p,2u)); p8_i64 high=p8_signed(p8_get(s,0u)+p8_get(p,4u)); return ({}) && ({});",
                    range(*ty, "low"),
                    range(*ty, "high")
                );
            }
            Rep::SaturatedProduct(ty) => {
                id.push_str("r=p8_put(r,0u,1ul); r=p8_put(r,2u,1ul);");
                join.push_str("p8_i64 x=p8_signed(p8_get(a,0u)), y=p8_signed(p8_get(b,0u)); r=p8_put(r,0u,p8_u64(p8_product(x,y))); r=p8_put(r,2u,p8_max(p8_get(a,2u),p8_satmul(p8_abs(x),p8_get(b,2u))));");
                apply.push_str("r=p8_put(r,0u,p8_u64(p8_product(p8_signed(p8_get(s,0u)),p8_signed(p8_get(p,0u)))));");
                // Conservative for i32::MIN: a false result requests exact replay.
                let limit = if *ty == IntTy::I32 {
                    2147483647u64
                } else {
                    4294967295
                };
                guard = format!(
                    "return p8_satmul(p8_abs(p8_signed(p8_get(s,0u))),p8_get(p,2u))<={limit}ul;"
                );
            }
            Rep::Affine(ring) => {
                let w = ring_words(*ring);
                let get = |v: &str, i: u32| {
                    if w == 1 {
                        format!("p8_u64({v}.w[{i}u])")
                    } else {
                        format!("p8_get({v},{}u)", i * w)
                    }
                };
                let put = |r: &str, i: u32, x: String| {
                    if w == 1 {
                        format!("{r}.w[{i}u]=uint({x});")
                    } else {
                        format!("{r}=p8_put({r},{}u,{x});", i * w)
                    }
                };
                id.push_str(&put("r", 0, "1ul".to_owned()));
                join.push_str(&put("r", 0, mul(*ring, &get("b", 0), &get("a", 0))));
                join.push_str(&put(
                    "r",
                    1,
                    add(*ring, &mul(*ring, &get("b", 0), &get("a", 1)), &get("b", 1)),
                ));
                apply.push_str(&put(
                    "r",
                    0,
                    add(*ring, &mul(*ring, &get("p", 0), &get("s", 0)), &get("p", 1)),
                ));
                if let Ring::Checked64(ty) = ring {
                    guard = format!(
                        "P8 v=p8_apply(p,s,1u); p8_i64 x=p8_signed(p8_get(v,0u)); return {};",
                        range(*ty, "x")
                    );
                }
            }
            Rep::Matrix { ring, vars } => {
                let n = vars.len() + 1;
                let w = ring_words(*ring) as usize;
                let get = |v: &str, i: usize| {
                    if w == 1 {
                        format!("p8_u64({v}.w[{i}u])")
                    } else {
                        format!("p8_get({v},{}u)", i * w)
                    }
                };
                let put = |i: usize, x: &str| {
                    if w == 1 {
                        format!("r.w[{i}u]=uint({x});")
                    } else {
                        format!("r=p8_put(r,{}u,{x});", i * w)
                    }
                };
                for i in 0..n {
                    id.push_str(&put(i * n + i, "1ul"));
                }
                for i in 0..n {
                    for j in 0..n {
                        join.push_str("{p8_u64 x=0ul;");
                        for k in 0..n {
                            join.push_str(&format!(
                                "x={};",
                                add(
                                    *ring,
                                    "x",
                                    &mul(*ring, &get("b", i * n + k), &get("a", k * n + j))
                                )
                            ));
                        }
                        join.push_str(&put(i * n + j, "x"));
                        join.push('}');
                    }
                }
                for i in 0..vars.len() {
                    apply.push_str(&format!("{{p8_u64 x={};", get("p", i * n + n - 1)));
                    for j in 0..vars.len() {
                        apply.push_str(&format!(
                            "x={};",
                            add(*ring, "x", &mul(*ring, &get("p", i * n + j), &get("s", j)))
                        ));
                    }
                    apply.push_str(&put(i, "x"));
                    apply.push('}');
                }
                if let Ring::Checked64(ty) = ring {
                    guard = "P8 v=p8_apply(p,s,1u);".to_owned();
                    for i in 0..vars.len() {
                        guard.push_str(&format!(
                            "{{p8_i64 x=p8_signed(p8_get(v,{}u)); if (!({})) return false;}}",
                            i * w,
                            range(*ty, "x")
                        ));
                    }
                    guard.push_str("return true;");
                }
            }
            Rep::Tropical { order, ty } => {
                let best = if *order == Order::Max {
                    "p8_smax"
                } else {
                    "p8_smin"
                };
                id.push_str("r.w[4]=1u;");
                join.push_str(&format!("r.w[4]=a.w[4]&b.w[4]; if(r.w[4]) r=p8_put(r,0u,p8_get(a,0u)+p8_get(b,0u)); bool shifted=a.w[5]&&b.w[4]; r.w[5]=uint(shifted||b.w[5]); p8_u64 x=p8_get(a,2u)+p8_get(b,0u); if(shifted&&b.w[5]) x=p8_u64({best}(p8_signed(x),p8_signed(p8_get(b,2u)))); else if(b.w[5]) x=p8_get(b,2u); if(r.w[5]) r=p8_put(r,2u,x);"));
                apply.push_str(&format!("p8_u64 x=p8_get(s,0u)+p8_get(p,0u); if(p.w[5]) x=p.w[4]?p8_u64({best}(p8_signed(x),p8_signed(p8_get(p,2u)))):p8_get(p,2u); r=p8_put(r,0u,x);"));
                guard = format!(
                    "P8 v=p8_apply(p,s,1u); p8_i64 x=p8_signed(p8_get(v,0u)); return {};",
                    range(*ty, "x")
                );
            }
            Rep::Select {
                order,
                strict,
                elem,
                ..
            } => {
                let less = match elem {
                    ScalarTy::F32 => "p8_flt(b.w[1],a.w[1])".to_owned(),
                    ScalarTy::I32 => "(int(b.w[1])<int(a.w[1]))".to_owned(),
                    ScalarTy::U32 => "(b.w[1]<a.w[1])".to_owned(),
                };
                let greater = match elem {
                    ScalarTy::F32 => "p8_flt(a.w[1],b.w[1])".to_owned(),
                    ScalarTy::I32 => "(int(a.w[1])<int(b.w[1]))".to_owned(),
                    ScalarTy::U32 => "(a.w[1]<b.w[1])".to_owned(),
                };
                let cmp = match (order, strict) {
                    (Order::Min, true) => less.clone(),
                    (Order::Min, false) => format!("!{greater}"),
                    (Order::Max, true) => greater.clone(),
                    (Order::Max, false) => format!("!{less}"),
                };
                if *elem == ScalarTy::F32 {
                    join.push_str("if(p8_nan(a.w[1])) a.w[0]=0u; if(p8_nan(b.w[1])) b.w[0]=0u;");
                }
                join.push_str(&format!(
                    "if(!a.w[0]) return b; if(!b.w[0]) return a; return ({cmp})?b:a;"
                ));
                if *elem == ScalarTy::F32 {
                    apply.push_str("if(p8_nan(s.w[1])) return s;");
                }
                apply.push_str("return p8_then(s,p);");
            }
            Rep::Last => {
                join.push_str("return b.w[0]?b:a;");
                apply.push_str("return p.w[0]?p:s;");
            }
            Rep::Counter(ty) => {
                join.push_str("r=p8_put(r,0u,p8_get(a,0u)+p8_get(b,0u));");
                apply.push_str("r=p8_put(r,0u,p8_get(s,0u)+p8_get(p,0u));");
                guard = format!(
                    "p8_i64 x=p8_signed(p8_get(s,0u)+p8_get(p,0u)); return {};",
                    range(*ty, "x")
                );
            }
            // `LawRepr::new` refuses lifted representations (#281).
            Rep::Lifted(_) => unreachable!("no collective for a lifted law"),
        },
    }
    format!(
        "P8_INLINE P8 p8_id() {{ {id} return r; }}\nP8_INLINE P8 p8_then(P8 a,P8 b) {{ {join} return r; }}\nP8_INLINE P8 p8_apply(P8 p,P8 s,uint count) {{ if(count==0u) return s; {apply} return r; }}\nP8_INLINE bool p8_guard(P8 p,P8 s) {{ {guard} }}\n"
    )
}

fn add(ring: Ring, a: &str, b: &str) -> String {
    ring_op(ring, a, b, "+")
}
fn mul(ring: Ring, a: &str, b: &str) -> String {
    ring_op(ring, a, b, "*")
}
fn ring_op(ring: Ring, a: &str, b: &str, op: &str) -> String {
    match ring {
        // Truncation mod 2^32 is a ring homomorphism, so 32-bit ops are exact.
        Ring::Wrapping32 => format!("p8_u64(uint({a}){op}uint({b}))"),
        Ring::Checked64(_) => format!("(({a}){op}({b}))"),
        Ring::Residue { modulus, .. } => format!("((({a}){op}({b}))%{modulus}ul)"),
    }
}

/// Every in-bounds word offset `i*words+w` with `i < 2^26 + tile` fits in u32.
/// Callers reject `n > 2^26`, so the narrow offset is exact, never truncated.
fn narrow_index(words: u32, tile: u32) -> bool {
    (u64::from(words) * ((1u64 << 26) + u64::from(tile))) <= u64::from(u32::MAX)
}

// Each 32-bit look-back slot carries its own ready bit, so relaxed accesses
// suffice on both targets: Metal relaxed atomics, CUDA volatile (relaxed.sys).
const METAL_IO: &str = r#"
#define p8_barrier() threadgroup_barrier(mem_flags::mem_threadgroup)
#define P8_UNROLL _Pragma("unroll")
#define p8_aligned(p) ((reinterpret_cast<ulong>(p)&15ul)==0ul)
#define P8_V4(p,i) (((device const uint4*)(p))[i])
#define P8_VSTORE(p,i,a,b,c,d) (((device uint4*)(p))[i]=uint4(a,b,c,d))
inline P8 p8_load(device const uint* input,uint i) { P8 r; for(uint w=0u;w<P8_WORDS;++w)r.w[w]=input[p8_ix(i)*P8_WORDS+w]; return r; }
inline void p8_write(device uint* output,uint i,P8 r) { for(uint w=0u;w<P8_WORDS;++w)output[p8_ix(i)*P8_WORDS+w]=r.w[w]; }
inline void p8_publish(device atomic_uint* slots,uint block,P8 r) {
 for(uint w=0u;w<P8_WORDS;++w) {
  uint i=(block*P8_WORDS+w)*2u;
  atomic_store_explicit(slots+i,(r.w[w]&65535u)*2u+1u,memory_order_relaxed);
  atomic_store_explicit(slots+i+1u,(r.w[w]>>16u)*2u+1u,memory_order_relaxed);
 }
}
inline bool p8_try(device atomic_uint* slots,uint block,thread P8& r) {
 bool ready=true;
 for(uint w=0u;w<P8_WORDS;++w) {
  uint i=(block*P8_WORDS+w)*2u;
  uint lo=atomic_load_explicit(slots+i,memory_order_relaxed);
  uint hi=atomic_load_explicit(slots+i+1u,memory_order_relaxed);
  ready=ready && ((lo&hi&1u)!=0u); r.w[w]=(lo>>1u)|((hi>>1u)<<16u);
 }
 return ready;
}
inline P8 p8_up(P8 v,uint d) { P8 r; for(uint w=0u;w<P8_WORDS;++w) r.w[w]=simd_shuffle_up(v.w[w],ushort(d)); return r; }
inline P8 p8_down(P8 v,uint d) { P8 r; for(uint w=0u;w<P8_WORDS;++w) r.w[w]=simd_shuffle_down(v.w[w],ushort(d)); return r; }
"#;
const CUDA_IO: &str = r#"
#define p8_barrier() __syncthreads()
#define P8_UNROLL _Pragma("unroll")
#define p8_aligned(p) ((reinterpret_cast<unsigned long long>(p)&15ull)==0ull)
struct alignas(16) p8_v4 { uint x,y,z,w; };
#define P8_V4(p,i) (reinterpret_cast<const p8_v4*>(p)[i])
#define P8_VSTORE(p,i,a,b,c,d) (reinterpret_cast<p8_v4*>(p)[i]=p8_v4{a,b,c,d})
__device__ P8 p8_load(const uint* __restrict__ input,uint i) { P8 r; for(uint w=0u;w<P8_WORDS;++w)r.w[w]=input[p8_ix(i)*P8_WORDS+w]; return r; }
__device__ void p8_write(uint* __restrict__ output,uint i,P8 r) { for(uint w=0u;w<P8_WORDS;++w)output[p8_ix(i)*P8_WORDS+w]=r.w[w]; }
__device__ void p8_publish(uint* slots,uint block,P8 r) {
 volatile uint* v=slots;
 for(uint w=0u;w<P8_WORDS;++w) {
  uint i=(block*P8_WORDS+w)*2u;
  v[i]=(r.w[w]&65535u)*2u+1u;
  v[i+1u]=(r.w[w]>>16u)*2u+1u;
 }
}
__device__ bool p8_try(uint* slots,uint block,P8& r) {
 const volatile uint* v=slots;
 bool ready=true;
 for(uint w=0u;w<P8_WORDS;++w) {
  uint i=(block*P8_WORDS+w)*2u;
  uint lo=v[i], hi=v[i+1u];
  ready=ready && ((lo&hi&1u)!=0u); r.w[w]=(lo>>1u)|((hi>>1u)<<16u);
 }
 return ready;
}
P8_INLINE P8 p8_up(P8 v,uint d) { P8 r; for(uint w=0u;w<P8_WORDS;++w) r.w[w]=__shfl_up_sync(0xffffffffu,v.w[w],d); return r; }
P8_INLINE P8 p8_down(P8 v,uint d) { P8 r; for(uint w=0u;w<P8_WORDS;++w) r.w[w]=__shfl_down_sync(0xffffffffu,v.w[w],d); return r; }
"#;
// Each lane folds P8_ITEMS consecutive elements. Full aligned runs read them as
// 16-byte vectors; ragged tails and other layouts keep the scalar loads.
const PROLOGUE: &str = r#"if (!admitted || n > 67108864u) return;
uint tiles=(n+P8_TILE-1u)/P8_TILE; if(tiles==0u) tiles=1u; if(group>=tiles) return;
uint block=P8_REVERSE ? tiles-1u-group : group;
uint start=block*P8_TILE + t*P8_ITEMS;
uint lane=t&31u, warp=t>>5u;
P8 local[P8_ITEMS];
P8 value=p8_id();
#if P8_VEC
if (start+P8_ITEMS<=n && p8_aligned(input)) {
 uint raw[P8_ITEMS*P8_WORDS];
 p8_ix base=p8_ix(start)*P8_WORDS/4u;
 P8_UNROLL for(uint k=0u;k<P8_ITEMS*P8_WORDS/4u;++k) { auto x=P8_V4(input,base+k); raw[4u*k]=x.x; raw[4u*k+1u]=x.y; raw[4u*k+2u]=x.z; raw[4u*k+3u]=x.w; }
 P8_UNROLL for(uint j=0u;j<P8_ITEMS;++j) { P8 x; P8_UNROLL for(uint w=0u;w<P8_WORDS;++w) x.w[w]=raw[j*P8_WORDS+w]; local[j]=value; value=p8_then(value,x); }
} else
#endif
for (uint j=0u; j<P8_ITEMS; ++j) {
 local[j]=value;
 if (start+j<n) value=p8_then(value,p8_load(input,start+j));
}
"#;
const REDUCE: &str = r#"
// Adjacent subtrees only: affine composition and tie order are not commutative.
// Doubling shuffles leave lane l with the ordered fold of lanes [l, l+2d).
P8_UNROLL for(uint d=1u;d<32u;d*=2u) { P8 b=p8_down(value,d); value=p8_then(value,b); }
if(lane==0u) shared[warp]=value;
p8_barrier();
if(warp==0u) {
 P8 x=lane<P8_WARPS?shared[lane]:p8_id();
 for(uint d=1u;d<P8_WARPS;d*=2u) { P8 b=p8_down(x,d); x=p8_then(x,b); }
 if(lane==0u) p8_write(output,block,x);
}
"#;
const SCAN: &str = r#"
// Ordered inclusive warp scans, then one warp scans the warp totals.
P8 v=value;
P8_UNROLL for(uint d=1u;d<32u;d*=2u) { P8 b=p8_up(v,d); if(lane>=d) v=p8_then(b,v); }
P8 up1=p8_up(v,1u);
if(lane==31u) shared[warp]=v;
p8_barrier();
if(warp==0u) {
 P8 x=lane<P8_WARPS?shared[lane]:p8_id();
 for(uint d=1u;d<P8_WARPS;d*=2u) { P8 b=p8_up(x,d); if(lane>=d) x=p8_then(b,x); }
 if(lane<P8_WARPS) shared[lane]=x;
}
p8_barrier();
P8 lane_before=warp==0u?p8_id():shared[warp-1u];
if(lane!=0u) lane_before=p8_then(lane_before,up1);
P8 total=shared[P8_WARPS-1u];
if(t==0u) {p8_publish(aggregates,block,total);carry=p8_id();remaining=block;}
// Thread 0 walks predecessors without group barriers; the group joins only to
// replay a tile whose records stay unavailable after P8_POLLS polls.
for(;;) {
 if(t==0u) {
  kind=2u;
  while(remaining>0u) {
   uint previous=remaining-1u; P8 candidate=p8_id(); uint found=0u;
   for(uint poll=0u;poll<=P8_POLLS && found==0u;++poll) {
    P8 prefix, partial;
    bool inclusive=p8_try(prefixes,previous,prefix) && P8_USE_PREFIXES;
    bool ready=p8_try(aggregates,previous,partial);
    if(inclusive) {candidate=prefix;found=2u;}
    else if(ready) {candidate=partial;found=1u;}
   }
   if(found==0u) {kind=0u;break;}
   carry=p8_then(candidate,carry);
   remaining=found==2u?0u:previous;
  }
 }
 p8_barrier();
 if(kind==2u) break;
 // As in M4-07 summary.rs: bounded polls, then a cooperative tile replay.
 // ponytail: worst-case quadratic tile work; tune only against measured D5 data.
 uint begin=(remaining-1u)*P8_TILE+t*P8_ITEMS;
 P8 candidate=p8_id();
 for(uint j=0u;j<P8_ITEMS && begin+j<n;++j)
  candidate=p8_then(candidate,p8_load(input,begin+j));
 shared[t]=candidate;p8_barrier();
 for(uint step=1u;step<P8_LANES;step*=2u) {
  uint left=t*step*2u;
  if(left+step<P8_LANES)shared[left]=p8_then(shared[left],shared[left+step]);
  p8_barrier();
 }
 if(t==0u) {carry=p8_then(shared[0],carry);remaining-=1u;}
 p8_barrier();
}
if(t==0u)p8_publish(prefixes,block,p8_then(carry,total));
P8 offset=p8_then(carry,lane_before);
#if P8_VEC
if(start+P8_ITEMS<=n && p8_aligned(output)) {
 uint raw[P8_ITEMS*P8_WORDS];
 P8_UNROLL for(uint j=0u;j<P8_ITEMS;++j) { P8 x=p8_then(offset,local[j]); P8_UNROLL for(uint w=0u;w<P8_WORDS;++w) raw[j*P8_WORDS+w]=x.w[w]; }
 p8_ix base=p8_ix(start)*P8_WORDS/4u;
 P8_UNROLL for(uint k=0u;k<P8_ITEMS*P8_WORDS/4u;++k) P8_VSTORE(output,base+k,raw[4u*k],raw[4u*k+1u],raw[4u*k+2u],raw[4u*k+3u]);
} else
#endif
for(uint j=0u;j<P8_ITEMS;++j)
 if(start+j<n) p8_write(output,start+j,p8_then(offset,local[j]));
"#;

#[cfg(test)]
mod tests {
    #[test]
    fn narrow_index_is_exact_at_the_length_ceiling() {
        // 63 words * (2^26 + 1024) < 2^32; 64 words would wrap the last offset.
        assert!(super::narrow_index(63, 1024));
        assert!(!super::narrow_index(64, 1024));
        assert!(super::narrow_index(6, 16384));
        assert!(!super::narrow_index(248, 1024));
    }
}
