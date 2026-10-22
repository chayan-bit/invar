use invar::{check, front, stencil_source, transpose_source};
use invar_device::{Capabilities, Provenance, stencil::Boundary, stencil_u32, transpose};

pub const SHAPES: &[[u32; 2]] = &[
    [0, 33],
    [33, 0],
    [1, 1],
    [31, 33],
    [127, 257],
    [1, 127],
    [127, 1],
];
pub fn transpose_source() -> &'static str {
    "fn flip(a:read Array<u32>,rows:u32,cols:u32)->Array<u32>{var b=filled_u32(rows*cols,0u32);for r in 0u32..rows{for c in 0u32..cols{b[c*rows+r]=a[r*cols+c];}}return b;} fn main()->i32{return 7;}"
}
pub fn stencil_source(boundary: Boundary) -> String {
    let center = "xs[r*cols+c]";
    let edge = |n: &str| {
        if boundary == Boundary::Clamp {
            format!("else {{{n}={center};}}")
        } else {
            String::new()
        }
    };
    format!(
        "fn step(xs:read Array<u32>,rows:u32,cols:u32)->Array<u32>{{var out=filled_u32(rows*cols,0u32);for r in 0u32..rows{{for c in 0u32..cols{{var n=0u32;if r>0u32{{n=xs[(r-1u32)*cols+c];}}{} var s=0u32;if r+1u32<rows{{s=xs[(r+1u32)*cols+c];}}{} var w=0u32;if c>0u32{{w=xs[r*cols+(c-1u32)];}}{} var e=0u32;if c+1u32<cols{{e=xs[r*cols+(c+1u32)];}}{} out[r*cols+c]=wrapping_add(wrapping_add(wrapping_add(wrapping_add({center},n),s),w),e);}}}}return out;}} fn main()->i32{{return 7;}}",
        edge("n"),
        edge("s"),
        edge("w"),
        edge("e")
    )
}
pub fn checked(source: &str) -> check::CheckedProgram {
    check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap()
}
pub fn transpose_plan(shape: [u32; 2], caps: Capabilities) -> transpose::Verified {
    let contract =
        transpose_source::Contract::from_checked(&checked(transpose_source()), "flip").unwrap();
    let emission = contract
        .emit_cuda(
            shape,
            u64::from(shape[0]) * u64::from(shape[1]),
            Capabilities::cuda_12_9_sm89(),
        )
        .unwrap();
    let ir = emission.plan().ir().clone();
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    transpose::verify(ir, caps, origin).unwrap()
}

pub fn stencil_plan(
    shape: [u32; 2],
    steps: u32,
    boundary: Boundary,
    caps: Capabilities,
) -> stencil_u32::Verified {
    stencil_source::Contract::from_checked(&checked(&stencil_source(boundary)), "step")
        .unwrap()
        .device_u32(shape, steps, caps)
        .unwrap()
}
pub fn input(shape: [u32; 2]) -> Vec<u32> {
    (0..shape[0] * shape[1])
        .map(|i| i.wrapping_mul(0x9e3779b9).rotate_left(i % 32) ^ u32::MAX)
        .collect()
}
pub fn stencil_oracle(shape: [u32; 2], steps: u32, boundary: Boundary, input: &[u32]) -> Vec<u32> {
    let [rows, cols] = shape;
    let mut a = input.to_vec();
    for _ in 0..steps {
        let mut b = vec![0; a.len()];
        for r in 0..rows {
            for c in 0..cols {
                let mut value = 0u32;
                for (dr, dc) in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let mut y = i64::from(r) + dr;
                    let mut x = i64::from(c) + dc;
                    if boundary == Boundary::Clamp {
                        y = y.clamp(0, i64::from(rows) - 1);
                        x = x.clamp(0, i64::from(cols) - 1);
                    }
                    if y >= 0 && x >= 0 && y < i64::from(rows) && x < i64::from(cols) {
                        value = value.wrapping_add(a[(y as u32 * cols + x as u32) as usize]);
                    }
                }
                b[(r * cols + c) as usize] = value;
            }
        }
        a = b;
    }
    a
}
pub fn assert_transpose(shape: [u32; 2], input: &[u32], output: &[u32]) {
    assert_eq!(input.len(), output.len());
    for r in 0..shape[0] {
        for c in 0..shape[1] {
            assert_eq!(
                output[(c * shape[0] + r) as usize],
                input[(r * shape[1] + c) as usize],
                "{shape:?} {r},{c}"
            );
        }
    }
}

/// Optional bounded evidence export. Never includes machine paths or input secrets.
pub fn archive(label: &str, ir: &str, source: &str) {
    if let Some(dir) = std::env::var_os("INVAR_M402_EVIDENCE_DIR") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("{label}.ir.txt")), ir).unwrap();
        std::fs::write(dir.join(format!("{label}.source.txt")), source).unwrap();
    }
}

pub fn padded_stencil(steps: u32, caps: Capabilities) -> stencil_u32::Verified {
    let p = stencil_plan([31, 33], steps, Boundary::Clamp, caps);
    let mut ir = p.ir().clone();
    for buffer in [&mut ir.input, &mut ir.output] {
        buffer.offset = 3;
        buffer.elements += 6;
    }
    let origin = Provenance {
        source_identity: ir.source_identity,
        graph_revision: ir.graph_revision,
    };
    stencil_u32::verify(ir, caps, origin).unwrap()
}
pub fn assert_padded_stencil(output: &[u32], steps: u32, input: &[u32]) {
    let end = input.len() - 3;
    assert_eq!(
        &output[3..end],
        stencil_oracle([31, 33], steps, Boundary::Clamp, &input[3..end])
    );
    for i in [0, 1, 2, end, end + 1, end + 2] {
        assert_eq!(
            output[i],
            if steps.is_multiple_of(2) { input[i] } else { 0 }
        );
    }
}

pub fn archive_oracle(
    shape: [u32; 2],
    steps: u32,
    boundary: Boundary,
    input: &[u32],
    output: &[u32],
) {
    if let Some(dir) = std::env::var_os("INVAR_M402_EVIDENCE_DIR") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        let value = serde_json::json!({"schema":"m4-02-cpu-oracle-v1","shape":shape,"boundary":boundary.name(),"steps":steps,"input":input,"output":output});
        std::fs::write(
            dir.join(format!(
                "oracle-{}-{}-{}-{steps}.json",
                boundary.name(),
                shape[0],
                shape[1]
            )),
            serde_json::to_vec_pretty(&value).unwrap(),
        )
        .unwrap();
    }
}
