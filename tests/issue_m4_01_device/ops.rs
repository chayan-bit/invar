#![allow(dead_code)]
use invar::{check, eval, f32_source, front, mir, numeric_contract::F32Bits};
use invar_device::{
    Capabilities, Provenance, Span,
    f32_ops::{self, Kind, Node, Program, Scalar},
    v2,
};

pub const WORDS: &[u32] = &[
    0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x00800001, 0x3f800000, 0xbf800000,
    0x3f800800, 0xbf801000, 0x4b800000, 0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc00001, 0xffc00002,
    0x7f800001,
];

pub fn plan(p: Program, caps: Capabilities) -> v2::Verified {
    let origin = Provenance {
        source_identity: 1,
        graph_revision: 1,
    };
    v2::verify(
        f32_ops::descriptor(p, Span { start: 0, end: 1 }, origin).unwrap(),
        origin,
        caps,
    )
    .unwrap()
}
pub fn programs() -> Vec<Program> {
    let input = [Scalar::F32, Scalar::F32, Scalar::F32];
    let mut cases = Vec::new();
    for op in [
        Node::Add(0, 1),
        Node::Sub(0, 1),
        Node::Mul(0, 1),
        Node::Fma(0, 1, 2),
    ] {
        cases.push(Program {
            length: WORDS.len() as u32,
            inputs: input.to_vec(),
            nodes: vec![Node::Input(0), Node::Input(1), Node::Input(2), op],
            kind: Kind::Map,
        });
    }
    for op in [
        f32_ops::Compare::Eq,
        f32_ops::Compare::Ne,
        f32_ops::Compare::Lt,
        f32_ops::Compare::Le,
        f32_ops::Compare::Gt,
        f32_ops::Compare::Ge,
    ] {
        cases.push(Program {
            length: WORDS.len() as u32,
            inputs: input[..2].to_vec(),
            nodes: vec![Node::Input(0), Node::Input(1), Node::Compare(op, 0, 1)],
            kind: Kind::Map,
        });
    }
    for (ty, op) in [
        (Scalar::U32, Node::FromU32(0)),
        (Scalar::I32, Node::FromI32(0)),
    ] {
        cases.push(Program {
            length: WORDS.len() as u32,
            inputs: vec![ty],
            nodes: vec![Node::Input(0), op],
            kind: Kind::Map,
        });
    }
    for length in [0, 1, 3, 31, 33] {
        for kind in [Kind::Reduce { initial: 0 }, Kind::Scan { initial: 0 }] {
            cases.push(Program {
                length,
                inputs: vec![Scalar::F32],
                nodes: vec![Node::Input(0)],
                kind,
            });
        }
    }
    cases
}
pub fn packed(p: &Program) -> Vec<u32> {
    (0..p.inputs.len())
        .flat_map(|plane| (0..p.length as usize).map(move |i| WORDS[(i + plane * 3) % WORDS.len()]))
        .collect()
}
pub fn same(actual: &[u32], expected: &[u32], float: bool) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            if float {
                F32Bits::from_bits(*a).strict_eq(F32Bits::from_bits(*b))
            } else {
                a == b
            },
            "word {i}: {a:#010x} != {b:#010x}"
        );
    }
}
pub fn sources() -> Vec<String> {
    let mut cases = Vec::new();
    for n in [0, 1, 3, 31, 33, 65] {
        for expression in [
            "a[i] + b[i]",
            "a[i] - b[i]",
            "a[i] * b[i]",
            "fma(a[i],b[i],-1.0)",
            "a[i] * b[i] + -1.0",
        ] {
            cases.push(format!("fn kernel(a: read Array<f32>,b: read Array<f32>,n:u32) -> Array<f32> {{ var out=filled_f32(n,0.0); for i in 0u32..n {{ out[i] = {expression}; }} return out; }} fn main()->i32 {{ let a=filled_f32({n}u32,1.00000011920928955078125); let b=filled_f32({n}u32,0.99999988079071044921875); let out=kernel(a,b,{n}u32); for i in 0u32..{n}u32 {{ if out[i] != {expression} {{ return 0; }} }} if len(out) == {n}u32 {{ return 7; }} return 0; }}"));
        }
        cases.push(format!("fn kernel(a:read Array<f32>,n:u32)->f32 {{ var sum=0.0; for i in 0u32..n {{ sum=sum+a[i]; }} return sum; }} fn main()->i32 {{ let a=filled_f32({n}u32,1.0); let sum=kernel(a,{n}u32); if sum == f32_from_u32({n}u32) {{ return 7; }} return 0; }}"));
        cases.push(format!("fn kernel(a:read Array<f32>,n:u32)->Array<f32> {{ var out=filled_f32(n,0.0); var sum=0.0; for i in 0u32..n {{ sum=sum+a[i]; out[i]=sum; }} return out; }} fn main()->i32 {{ let a=filled_f32({n}u32,1.0); let out=kernel(a,{n}u32); for i in 0u32..{n}u32 {{ if out[i] != f32_from_u32(i+1u32) {{ return 0; }} }} return 7; }}"));
    }
    cases.push("fn kernel(a:read Array<u32>,n:u32)->Array<f32> {var out=filled_f32(n,0.0);for i in 0u32..n {out[i]=f32_from_u32(a[i]);}return out;} fn main()->i32 {let a: Array<u32> = [16777217u32,4294967295u32];let out=kernel(a,2u32);if out[0u32]==16777216.0 and out[1u32]==4294967296.0{return 7;}return 0;}".into());
    cases.push("fn kernel(a:read Array<i32>,n:u32)->Array<f32> {var out=filled_f32(n,0.0);for i in 0u32..n {out[i]=f32_from_i32(a[i]);}return out;} fn main()->i32 {let a: Array<i32> = [-7i32,16777217i32];let out=kernel(a,2u32);if out[0u32]==-7.0 and out[1u32]==16777216.0{return 7;}return 0;}".into());
    cases.push("fn kernel(a:read Array<f32>,n:u32)->Array<i32> {var out=filled_i32(n,0);for i in 0u32..n {out[i]=trunc_i32(a[i]);}return out;} fn main()->i32 {let a:Array<f32> = [-7.9,0.0,2147483520.0];let out=kernel(a,3u32);if out[0u32]==-7 and out[2u32]==2147483520{return 7;}return 0;}".into());
    for op in ["==", "!=", "<", "<=", ">", ">="] {
        cases.push(format!("fn kernel(a:read Array<f32>,b:read Array<f32>,n:u32)->Array<u32>{{var out=filled_u32(n,0u32);for i in 0u32..n{{if a[i] {op} b[i]{{out[i]=1u32;}}else{{out[i]=0u32;}}}}return out;}} fn main()->i32{{let a:Array<f32> = [1.0,0.0/0.0,-0.0];let b:Array<f32> = [2.0,0.0/0.0,0.0];let out=kernel(a,b,3u32);for i in 0u32..3u32{{if a[i] {op} b[i]{{if out[i]!=1u32{{return 0;}}}}else{{if out[i]!=0u32{{return 0;}}}}}}return 7;}}"));
    }
    cases
}
pub fn scalar_sources() -> Vec<String> {
    let mut sources=vec!["fn kernel(a:Quantity<f32,m>,b:Quantity<f32,m>)->Quantity<f32,m>{return a+b;} fn main()->i32{return 7;}".into()];
    for op in ["==", "!=", "<", "<=", ">", ">="] {
        sources.push(format!(
            "fn kernel(a:f32,b:f32)->Bool{{return a {op} b;}} fn main()->i32{{return 7;}}"
        ));
    }
    sources
}
pub fn checked(source: &str) -> check::CheckedProgram {
    check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap()
}
pub fn reference(source: &str) -> eval::Observation {
    eval::observe(&checked(source), 1_000_000)
}
pub fn parity(run: &f32_source::Run, expected: &eval::Observation) {
    assert_eq!(run.observation.result, expected.result);
    assert_eq!(run.observation.steps_used, expected.steps_used);
    assert_eq!(run.observation.cleanup_trace, expected.cleanup_trace);
}
pub fn mir_program(source: &str) -> mir::Program {
    mir::lower(&checked(source)).unwrap()
}
