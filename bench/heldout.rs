//! Reviewed later: candidate oracle programs and separate performance recipes.
use super::{corpus, hash::sha256};
use invar::{check, contraction_source, eval, front, frontier_source, mir, mir_eval};
use serde_json::{Value, json};

pub const FAMILIES: [&str; 10] = [
    "map",
    "reduction",
    "scan",
    "compaction",
    "transpose",
    "stencil",
    "matrix",
    "segmented",
    "gather_scatter",
    "frontier",
];
pub const EXTRA_PER_FAMILY: usize = 32;
const FUEL: u64 = 20_000_000;

/// Forms that became available after #106 (pair reduction #115, segmented map
/// #100, Tensor Core source contract #110). They use their own PRNG stream so
/// the 560 earlier programs keep their bytes and order.
pub const EXTRA_FORM_COUNT: usize = 16;
const FORM_STREAM: u64 = 0x666f_726d_7332_3032;

fn u32_array(xs: &[u32]) -> String {
    format!(
        "[{}]",
        xs.iter()
            .map(|x| format!("{x}u32"))
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn f32_array(xs: &[f32]) -> String {
    format!(
        "[{}]",
        xs.iter()
            .map(|x| format!("{x:.1}"))
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn checks(expected: &str, ty: &str) -> String {
    // Finite nonzero equality is bit exact. Reciprocals distinguish +0/-0,
    // which ordinary source equality deliberately treats as equal.
    let zero = if ty == "f32" {
        "if wanted[q]==0.0 {if 1.0/answer[q]!=1.0/wanted[q]{return 1;}}"
    } else {
        ""
    };
    format!(
        "let wanted: Array<{ty}> = {expected}; if len(answer) != len(wanted) {{ return 1; }} for q in 0u32..len(wanted) {{ if answer[q] != wanted[q] {{ return 1; }} {zero} }} return 0;"
    )
}

fn finish(leaf: &str, setup: &str, expected: &[u32]) -> String {
    format!(
        "{leaf} fn main()->i32 {{ {setup} {} }}",
        checks(&u32_array(expected), "u32")
    )
}
fn finish_float(leaf: &str, setup: &str, expected: &[f32]) -> String {
    format!(
        "{leaf} fn main()->i32 {{ {setup} {} }}",
        checks(&f32_array(expected), "f32")
    )
}
fn shuffle(xs: &mut [u32], state: &mut u64) {
    for i in (1..xs.len()).rev() {
        xs.swap(i, corpus::next(state) as usize % (i + 1));
    }
}
fn shape_proposals() -> Value {
    json!({
        "status":"proposal-pending-review",
        "scope":"CPU oracle invocations and performance recipes are separate; no large source invocation or GPU performance acceptance is claimed",
        "map":{"exact_elements":[0,1,31,32,33,255,256,257,4093],"powers_of_two_exponents":[10,12,14,16,18,20,22,24,26,28],"budget_rule":"retain every requested shape; resource refusal is explicit, never remove a failed case"},
        "reduction":{"empty":0,"singleton":1,"odd":33,"ragged_block":257,"multi_block":4093,"pair_law":"ordered affine composition, wrapping u32"},
        "scan":{"boundary_elements":[0,1,31,32,33,255,256,257,4093],"long_sequence":65537,"nonidentity_seed":7},
        "compaction":{"exact_selectivity_percent":[0,1,50,99,100],"elements":10000},
        "transpose":{"exact_dimensions":[[31,33],[127,257],[1024,1024]],"larger_bounded_cases":[[2048,1025]],"strided_read_view":{"dimensions":[31,33],"stride":2}},
        "stencil":{"tiny_domains":[[0,0],[1,1],[1,3]],"halo_boundaries":[[2,2],[3,5]],"odd_widths":[[31,33]],"wide_tall":[[3,257],[257,3]],"larger_square":[[1024,1024]],"ping_pong_steps":[1,2,3,16]},
        "matrix":{"exact_square_sizes_where_budget_permits":[256,1024,4096],"rectangular":[[31,33,17]],"one_sized_dimensions":[[1,1,1],[1,33,17],[31,1,17],[31,33,1]],"zero_reduction_extent":[[31,0,17]],"odd_dimensions":[[31,33,17]],"policies":["strict ordered binary32","explicit Tensor Core tc-f16-f32-v1 (D5 as amended 2026-10-14)"]},
        "segmented":{"empty_segments":[0,0,0],"uniform_lengths":[4,4,4,4],"skewed_lengths":[0,1,2,13],"one_dominant_segment":[0,0,15,1],"ragged_totals":[3,5,9]},
        "gather_scatter":{"elements":[0,1,31,33,257,4093],"index_classes":["sorted","random","duplicates","out_of_bounds rejection"],"conflict_policy":"injective scatter; wrapping modular sum histogram"},
        "frontier":{"chains":[1,5,16],"stars":[5,17],"balanced_small_graphs":[7,15],"skewed_small_graphs":[8,17],"bounded_sparse_larger_cases":[257],"rounds":"vertex count","observations":"reachability set and per-level masks; serial pointer chasing is a separate negative control"}
    })
}

// Full boundary-size CPU witnesses augment the unchanged three-element corpus.
const BOUNDARIES: [u32; 9] = [0, 1, 31, 32, 33, 255, 256, 257, 4093];
// ponytail: this adapter accepts the existing generator spelling; return
// structured map recipes if that spelling changes.
fn map_boundary(template: &Value, n: u32) -> Result<String, String> {
    let source = template["source"].as_str().ok_or("map source missing")?;
    let start = source.find("if result0[").ok_or("map oracle missing")?;
    let mut body = source[..start]
        .replace("filled_u32(3u32,", &format!("filled_u32({n}u32,"))
        .replace(" .. 3u32 {", &format!(" .. {n}u32 {{"));
    for output in 0..template["outputs"].as_u64().ok_or("map outputs missing")? {
        let marker = format!("if result{output}[0u32] != ");
        let wanted = source
            .split(&marker)
            .nth(1)
            .and_then(|s| s.split("u32").next())
            .ok_or("map expected value missing")?;
        body.push_str(&format!(
            "if oracle(result{output},{wanted}u32)!=0{{return 1;}}"
        ));
    }
    body.push_str("return 0; } fn oracle(xs:read Array<u32>,wanted:u32)->i32{for q in 0u32..len(xs){if xs[q]!=wanted{return 1;}}return 0;}");
    Ok(body)
}
fn reduction_boundary(state: &mut u64, n: u32, product: bool) -> String {
    let value = corpus::next(state);
    let seed = 1 + corpus::next(state) % 9;
    let expected = (0..n).fold(u128::from(seed), |s, _| {
        if product {
            (s * u128::from(value)) % (1u128 << 32)
        } else {
            (s + u128::from(value)) % (1u128 << 32)
        }
    }) as u32;
    let op = if product {
        "wrapping_mul"
    } else {
        "wrapping_add"
    };
    format!(
        "fn main()->i32{{let xs:Array<u32> =filled_u32({n}u32,{value}u32);var acc={seed}u32;for i in 0u32..len(xs){{acc={op}(acc,xs[i]);}}if acc!={expected}u32{{return 1;}}return 0;}}"
    )
}
fn scan_boundary(state: &mut u64, n: u32, mode: usize) -> String {
    let a = 1 + corpus::next(state) % 4;
    let b = corpus::next(state);
    let seed = 1 + corpus::next(state) % 9;
    let mut expected = u128::from(seed);
    let mut prefixes = Vec::new();
    for _ in 0..n {
        let previous = expected;
        expected = (u128::from(a) * expected + u128::from(b)) % (1u128 << 32);
        prefixes.push(if mode == 1 {
            previous as u32
        } else {
            expected as u32
        });
    }
    let publication = if mode == 1 { "out[i]=state;" } else { "" };
    let after = if mode == 2 { "out[i]=state;" } else { "" };
    let allocate = if mode == 0 {
        String::new()
    } else {
        format!("var out:Array<u32> =filled_u32({n}u32,0u32);")
    };
    let compare = if mode == 0 {
        String::new()
    } else {
        format!(
            "let wanted:Array<u32> ={};if oracle(out,wanted)!=0{{return 1;}}",
            u32_array(&prefixes)
        )
    };
    let mut source = format!(
        "fn main()->i32{{let aa:Array<u32> =filled_u32({n}u32,{a}u32);let bb:Array<u32> =filled_u32({n}u32,{b}u32);var state={seed}u32;{allocate}for i in 0u32..len(aa){{{publication}state=wrapping_add(wrapping_mul(aa[i],state),bb[i]);{after}}}{compare}if state!={expected}u32{{return 1;}}return 0;}}"
    );
    if mode != 0 {
        source.push_str("fn oracle(xs:read Array<u32>,wanted:read Array<u32>)->i32{if len(xs)!=len(wanted){return 1;}for q in 0u32..len(xs){if xs[q]!=wanted[q]{return 1;}}return 0;}");
    }
    source
}

fn compaction(state: &mut u64, ordinal: usize) -> Result<(String, Value), String> {
    let percent = [0, 1, 50, 99, 100][ordinal % 5];
    let mut xs: Vec<u32> = (0..100).collect();
    shuffle(&mut xs, state);
    let expected: Vec<u32> = xs.iter().copied().filter(|&x| x < percent).collect();
    let leaf = "fn work(xs: read Array<u32>, count: u32, cutoff: u32) -> Array<u32> { var selected=0u32; for i in 0u32..count { if xs[i] < cutoff { selected=selected+1u32; } } var out=filled_u32(selected,0u32); var pos=0u32; for j in 0u32..count { if xs[j] < cutoff { out[pos]=xs[j]; pos=pos+1u32; } } return move out; }";
    let prepared = invar::compaction_source::Contract::from_source(leaf.as_bytes(), "work")
        .map_err(|e| format!("{e:?}"))?;
    let actual = prepared
        .prepare(
            &xs,
            100,
            percent,
            invar::compaction_source::Budget {
                max_elements: 100,
                max_bytes: 2000,
                max_work: 700,
            },
        )
        .and_then(|p| p.execute())
        .map_err(|e| format!("{e:?}"))?;
    if actual != expected {
        return Err("compaction host oracle mismatch".into());
    }
    Ok((
        finish(
            leaf,
            &format!(
                "let xs: Array<u32> = {}; let answer=work(xs,100u32,{percent}u32);",
                u32_array(&xs)
            ),
            &expected,
        ),
        json!({"elements":100,"selectivity_percent":percent}),
    ))
}
fn transpose(state: &mut u64, ordinal: usize) -> Result<(String, Value), String> {
    let [rows, cols] = [[0, 1], [1, 1], [3, 5], [31, 33]][ordinal % 4];
    let xs: Vec<u32> = (0..rows * cols).map(|_| corpus::next(state)).collect();
    let expected: Vec<u32> = (0..cols)
        .flat_map(|c| (0..rows).map(move |r| (r, c)))
        .map(|(r, c)| xs[(r * cols + c) as usize])
        .collect();
    let leaf = "fn work(a: read Array<u32>, rows: u32, cols: u32) -> Array<u32> { var b=filled_u32(rows*cols,0u32); for r in 0u32..rows { for c in 0u32..cols { b[c*rows+r]=a[r*cols+c]; } } return move b; }";
    Ok((
        finish(
            leaf,
            &format!(
                "let xs: Array<u32> = {}; let answer=work(xs,{rows}u32,{cols}u32);",
                u32_array(&xs)
            ),
            &expected,
        ),
        json!({"dimensions":[rows,cols]}),
    ))
}
fn stencil(state: &mut u64, ordinal: usize) -> Result<(String, Value), String> {
    let [rows, cols] = [[0, 0], [1, 1], [3, 5], [5, 3]][ordinal % 4];
    let steps = 1 + ordinal % 3;
    let clamp = ordinal % 2 == 1;
    let xs: Vec<f32> = (0..rows * cols)
        .map(|_| (corpus::next(state) % 9) as f32 - 4.0)
        .collect();
    let mut expected = xs.clone();
    for _ in 0..steps {
        let load = |r: i32, c: i32, center: f32| {
            if r < 0 || c < 0 || r >= rows || c >= cols {
                if clamp { center } else { 0.0 }
            } else {
                expected[(r * cols + c) as usize]
            }
        };
        expected = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (r, c)))
            .map(|(r, c)| {
                let v = expected[(r * cols + c) as usize];
                (((v + load(r - 1, c, v)) + load(r + 1, c, v)) + load(r, c - 1, v))
                    + load(r, c + 1, v)
            })
            .collect();
    }
    let edge = |v: &str| {
        if clamp {
            format!("else {{ {v}=a[r*cols+c]; }}")
        } else {
            String::new()
        }
    };
    let leaf = format!(
        "fn work(a:read Array<f32>,rows:u32,cols:u32)->Array<f32>{{var b=filled_f32(rows*cols,0.0);for r in 0u32..rows{{for c in 0u32..cols{{var n=0.0; if r>0u32{{n=a[(r-1u32)*cols+c];}} {ne} var s=0.0; if r+1u32<rows{{s=a[(r+1u32)*cols+c];}} {se} var w=0.0; if c>0u32{{w=a[r*cols+(c-1u32)];}} {we} var e=0.0; if c+1u32<cols{{e=a[r*cols+(c+1u32)];}} {ee} b[r*cols+c]=(((a[r*cols+c]+n)+s)+w)+e;}}}}return b;}}",
        ne = edge("n"),
        se = edge("s"),
        we = edge("w"),
        ee = edge("e")
    );
    let mut setup = format!("let xs: Array<f32> = {};", f32_array(&xs));
    let mut current = "xs".to_owned();
    for step in 0..steps {
        setup.push_str(&format!(
            "let step{step}=work({current},{rows}u32,{cols}u32);"
        ));
        current = format!("step{step}");
    }
    setup.push_str(&format!("let answer=move {current};"));
    Ok((
        finish_float(&leaf, &setup, &expected),
        json!({"dimensions":[rows,cols],"steps":steps,"boundary":if clamp{"clamp"}else{"constant_zero"}}),
    ))
}
fn matrix(state: &mut u64, ordinal: usize) -> Result<(String, Value), String> {
    let [rows, inner, cols] = [[1, 1, 1], [3, 0, 5], [3, 5, 1], [3, 5, 7]][ordinal % 4];
    let a: Vec<f32> = (0..rows * inner)
        .map(|_| (corpus::next(state) % 5) as f32 - 2.0)
        .collect();
    let b: Vec<f32> = (0..inner * cols)
        .map(|_| (corpus::next(state) % 5) as f32 - 2.0)
        .collect();
    let expected: Vec<f32> = (0..rows)
        .flat_map(|r| (0..cols).map(move |c| (r, c)))
        .map(|(r, c)| {
            (0..inner).fold(0.0, |sum, k| {
                sum + a[(r * inner + k) as usize] * b[(k * cols + c) as usize]
            })
        })
        .collect();
    let leaf = "fn work(a:read Array<f32>,b:read Array<f32>,rows:u32,inner:u32,cols:u32)->Array<f32>{var out=filled_f32(rows*cols,0.0);for i in 0u32..rows{for j in 0u32..cols{var sum=0.0;for k in 0u32..inner{sum=sum+a[i*inner+k]*b[k*cols+j];}out[i*cols+j]=sum;}}return move out;}";
    Ok((
        finish_float(
            leaf,
            &format!(
                "let a: Array<f32> ={};let b: Array<f32> ={};let answer=work(a,b,{rows}u32,{inner}u32,{cols}u32);",
                f32_array(&a),
                f32_array(&b)
            ),
            &expected,
        ),
        json!({"dimensions":[rows,inner,cols],"policy":"strict ordered binary32, separately rounded mul and add"}),
    ))
}
fn segmented(state: &mut u64, ordinal: usize) -> Result<(String, Value), String> {
    let offsets: Vec<u32> = [
        vec![0, 0, 0],
        vec![0, 4, 8, 12],
        vec![0, 0, 1, 3, 16],
        vec![0, 0, 0, 15, 16],
        vec![0, 3, 8, 17],
    ][ordinal % 5]
        .clone();
    let product = ordinal % 2 == 1;
    let xs: Vec<u32> = (0..*offsets.last().unwrap())
        .map(|_| corpus::next(state))
        .collect();
    let expected: Vec<u32> = offsets
        .windows(2)
        .map(|w| {
            xs[w[0] as usize..w[1] as usize].iter().fold(
                if product { 1u128 } else { 0u128 },
                |s, &x| {
                    if product {
                        (s * u128::from(x)) % (1u128 << 32)
                    } else {
                        (s + u128::from(x)) % (1u128 << 32)
                    }
                },
            ) as u32
        })
        .collect();
    let count = offsets.len() - 1;
    let leaf = format!(
        "fn work(xs:read Array<u32>,offsets:read Array<u32>,count:u32)->Array<u32>{{var out=filled_u32(count,0u32);for s in 0u32..count{{var acc={}u32;for i in offsets[s]..offsets[s+1u32]{{acc={}(acc,xs[i]);}}out[s]=acc;}}return move out;}}",
        if product { 1 } else { 0 },
        if product {
            "wrapping_mul"
        } else {
            "wrapping_add"
        }
    );
    let c = invar::segmented_source::Contract::from_source(leaf.as_bytes(), "work")
        .map_err(|e| format!("{e:?}"))?;
    let actual = c
        .prepare(
            &xs,
            &offsets,
            count as u32,
            invar::segmented_source::Budget {
                max_elements: 100,
                max_segments: 100,
                max_output_bytes: 400,
                max_work: 400,
            },
        )
        .and_then(|p| p.execute())
        .map_err(|e| format!("{e:?}"))?;
    if actual != expected {
        return Err("segmented host oracle mismatch".into());
    }
    Ok((
        finish(
            &leaf,
            &format!(
                "let xs: Array<u32> ={};let offsets: Array<u32> ={};let answer=work(xs,offsets,{count}u32);",
                u32_array(&xs),
                u32_array(&offsets)
            ),
            &expected,
        ),
        json!({"offsets":offsets,"law":if product{"wrapping_product"}else{"wrapping_sum"}}),
    ))
}
fn indexed(state: &mut u64, ordinal: usize) -> Result<(String, Value), String> {
    let n = [0, 1, 7, 31, 33][ordinal % 5];
    let mode = ordinal % 3;
    let xs: Vec<u32> = (0..n).map(|_| corpus::next(state)).collect();
    let mut ix: Vec<u32> = (0..n).collect();
    if ordinal % 2 == 1 {
        shuffle(&mut ix, state);
    }
    if mode != 1 && n > 0 && ordinal.is_multiple_of(2) {
        for i in &mut ix {
            *i = corpus::next(state) % n;
        }
    }
    let expected: Vec<u32> = if mode == 0 {
        ix.iter().map(|&i| xs[i as usize]).collect()
    } else {
        (0..n)
            .map(|bin| {
                let values: Vec<u32> = ix
                    .iter()
                    .enumerate()
                    .filter(|(_, i)| **i == bin)
                    .map(|(i, _)| xs[i])
                    .collect();
                if mode == 1 {
                    values.first().copied().unwrap_or(0)
                } else {
                    (values.iter().map(|&x| u64::from(x)).sum::<u64>() % (1u64 << 32)) as u32
                }
            })
            .collect()
    };
    let store = match mode {
        0 => "out[i]=a[ix[i]];",
        1 => "out[ix[i]]=a[i];",
        _ => "out[ix[i]]=wrapping_add(out[ix[i]],a[i]);",
    };
    let leaf = format!(
        "fn work(a:read Array<u32>,ix:read Array<u32>{})->Array<u32>{{var out=filled_u32({},0u32);for i in 0u32..len(ix){{{store}}}return out;}}",
        if mode == 0 { "" } else { ",n:u32" },
        if mode == 0 { "len(ix)" } else { "n" }
    );
    let checked = check::check(
        front::parse(leaf.as_bytes()).map_err(|e| format!("{e:?}"))?,
        false,
    )
    .map_err(|e| format!("{e:?}"))?;
    let actual = if mode == 2 {
        invar::histogram_source::Contract::from_checked(leaf.as_bytes(), &checked, "work")
            .map_err(|e| format!("{e:?}"))?
            .candidate(&xs, &ix, n)
            .map_err(|e| format!("{e:?}"))?
            .output()
            .to_vec()
    } else {
        invar::indexed_source::Contract::from_checked(leaf.as_bytes(), &checked, "work")
            .map_err(|e| format!("{e:?}"))?
            .candidate(&xs, &ix, if mode == 0 { None } else { Some(n) })
            .map_err(|e| format!("{e:?}"))?
            .output()
            .to_vec()
    };
    if actual != expected {
        return Err("indexed host oracle mismatch".into());
    }
    Ok((
        finish(
            &leaf,
            &format!(
                "let a: Array<u32> ={};let ix: Array<u32> ={};let answer=work(a,ix{});",
                u32_array(&xs),
                u32_array(&ix),
                if mode == 0 {
                    String::new()
                } else {
                    format!(",{n}u32")
                }
            ),
            &expected,
        ),
        json!({"elements":n,"mode":(["gather","injective_scatter","modular_histogram"][mode]),"indices":ix}),
    ))
}
fn frontier(state: &mut u64, ordinal: usize) -> Result<(String, Value), String> {
    let n = 3 + ordinal % 6;
    let topology = ordinal % 4;
    let mut edges = vec![0; n * n];
    for v in 1..n {
        let parent = match topology {
            0 => v - 1,
            1 => 0,
            2 => (v - 1) / 2,
            _ => corpus::next(state) as usize % v,
        };
        edges[parent * n + v] = 1;
    }
    let start = corpus::next(state) as usize % n;
    let mut seen = vec![0; n];
    let mut queue = std::collections::VecDeque::from([start]);
    while let Some(u) = queue.pop_front() {
        if seen[u] == 1 {
            continue;
        }
        seen[u] = 1;
        for v in 0..n {
            if edges[u * n + v] == 1 && seen[v] == 0 {
                queue.push_back(v);
            }
        }
    }
    let leaf = include_str!("general-net-v01/current/frontier_graph_a.inv")
        .split("\nfn main()")
        .next()
        .unwrap()
        .replace("reach", "work")
        .replace("5u32", &format!("{n}u32"));
    let checked = check::check(
        front::parse(leaf.as_bytes()).map_err(|e| format!("{e:?}"))?,
        false,
    )
    .map_err(|e| format!("{e:?}"))?;
    let c = frontier_source::Contract::from_checked(leaf.as_bytes(), &checked, "work")
        .map_err(|e| format!("{e:?}"))?;
    let actual = c
        .candidate(&edges, start as u32, Default::default())
        .map_err(|e| format!("{e:?}"))?;
    if actual.seen() != seen {
        return Err("frontier reachability oracle mismatch".into());
    }
    let source = format!(
        "{leaf} fn main()->i32{{let edges: Array<u32> ={};if work(edges,{start}u32)!={}{{return 1;}}return 0;}}",
        u32_array(&edges),
        seen.iter().sum::<u32>()
    );
    Ok((
        source,
        json!({"vertices":n,"rounds":n,"topology":(["chain","star","balanced","skewed"][topology]),"start":start,"expected_seen":seen}),
    ))
}

const PAIR_LEAF: &str = "fn work(aa: read Array<u32>, bb: read Array<u32>, n: u32, seed_a: u32, seed_b: u32) -> Array<u32> { var sa = seed_a; var sb = seed_b; for i in 0u32..n { sa = wrapping_mul(aa[i],sa); sb = wrapping_add(wrapping_mul(aa[i],sb),bb[i]); } return [sa,sb]; }";
fn pair_reduction(state: &mut u64, ordinal: usize) -> (String, Value) {
    let n = [0usize, 1, 2, 7, 31, 32, 33, 64][ordinal % 8];
    let aa: Vec<u32> = (0..n).map(|_| corpus::next(state) | 1).collect();
    let bb: Vec<u32> = (0..n).map(|_| corpus::next(state)).collect();
    // Even ordinals fold from the identity; odd ones from a live nonidentity seed.
    let seed = if ordinal.is_multiple_of(2) {
        [1, 0]
    } else {
        [corpus::next(state), corpus::next(state)]
    };
    // Widened arithmetic modulo 2^32, independent of the production law.
    let expected = aa.iter().zip(&bb).fold(seed, |[x, y], (&a, &b)| {
        [
            (u64::from(a) * u64::from(x)) as u32,
            (u64::from(a) * u64::from(y) + u64::from(b)) as u32,
        ]
    });
    let setup = format!(
        "let aa: Array<u32> = {}; let bb: Array<u32> = {}; let answer=work(aa,bb,{n}u32,{}u32,{}u32);",
        u32_array(&aa),
        u32_array(&bb),
        seed[0],
        seed[1]
    );
    (
        finish(PAIR_LEAF, &setup, &expected),
        json!({"elements":n,"seed":seed,"law":"ordered affine pair composition, wrapping u32","expected":expected}),
    )
}
fn segmented_map(state: &mut u64, ordinal: usize) -> (String, Value) {
    let offsets: Vec<u32> = [
        vec![0],
        vec![0, 0, 0],
        vec![0, 4, 8, 12, 16],
        vec![0, 0, 1, 3, 16],
        vec![0, 0, 0, 15, 16],
        vec![0, 3, 8, 17],
    ][ordinal % 6]
        .clone();
    let n = *offsets.last().unwrap();
    let xs: Vec<u32> = (0..n).map(|_| corpus::next(state)).collect();
    let add = corpus::next(state) % 97;
    let mul = corpus::next(state) | 1;
    let with_segment = !ordinal.is_multiple_of(3);
    let mut expected = vec![0u32; n as usize];
    for (s, w) in offsets.windows(2).enumerate() {
        for i in w[0] as usize..w[1] as usize {
            let v = (u64::from(xs[i]) + u64::from(add)) as u32;
            let v = (u64::from(v) * u64::from(mul)) as u32;
            expected[i] = if with_segment {
                (u64::from(v) + s as u64) as u32
            } else {
                v
            };
        }
    }
    let callback = format!("wrapping_mul(wrapping_add(xs[i], {add}u32), {mul}u32)");
    let callback = if with_segment {
        format!("wrapping_add({callback}, s)")
    } else {
        callback
    };
    let count = offsets.len() - 1;
    let leaf = format!(
        "fn work(xs: read Array<u32>, offsets: read Array<u32>, count: u32) -> Array<u32> {{ var out = filled_u32(len(xs), 0u32); for s in 0u32..count {{ for i in offsets[s]..offsets[s + 1u32] {{ out[i] = {callback}; }} }} return move out; }}"
    );
    let setup = format!(
        "let xs: Array<u32> ={};let offsets: Array<u32> ={};let answer=work(xs,offsets,{count}u32);",
        u32_array(&xs),
        u32_array(&offsets)
    );
    (
        finish(&leaf, &setup, &expected),
        json!({"offsets":offsets,"form":"per-segment map","segment_operand":with_segment}),
    )
}
/// A `tc-f16-f32-v1` contraction and its strict-f32 twin. Inputs are small
/// integers, exact in binary16, and every partial sum is an exact binary32
/// integer, so the exact-sum reference, the strict ordered result and every
/// result within the amended D5 bound coincide bit for bit.
fn tc_matrix(state: &mut u64, ordinal: usize) -> (String, String, Value) {
    let [rows, inner, cols] = [
        [1, 1, 1],
        [3, 0, 5],
        [3, 5, 1],
        [16, 16, 8],
        [17, 33, 9],
        [8, 64, 8],
    ][ordinal % 6];
    let a: Vec<f32> = (0..rows * inner)
        .map(|_| (corpus::next(state) % 9) as f32 - 4.0)
        .collect();
    let b: Vec<f32> = (0..inner * cols)
        .map(|_| (corpus::next(state) % 9) as f32 - 4.0)
        .collect();
    let expected: Vec<f32> = (0..rows)
        .flat_map(|r| (0..cols).map(move |c| (r, c)))
        .map(|(r, c)| {
            (0..inner)
                .map(|k| {
                    f64::from(a[(r * inner + k) as usize]) * f64::from(b[(k * cols + c) as usize])
                })
                .fold(0.0, |s, x| s + x) as f32
        })
        .collect();
    let step = |tc: bool| {
        if tc {
            "c[i*cols+j]=tc_mul_add(a[i*inner+k],b[k*cols+j],c[i*cols+j]);"
        } else {
            "c[i*cols+j]=c[i*cols+j]+a[i*inner+k]*b[k*cols+j];"
        }
    };
    let leaf = |tc: bool| {
        format!(
            "fn work(a:read Array<f32>,b:read Array<f32>,rows:u32,inner:u32,cols:u32)->Array<f32>{{var c=filled_f32(rows*cols,0.0);for i in 0u32..rows{{for j in 0u32..cols{{for k in 0u32..inner{{{}}}}}}}return move c;}}",
            step(tc)
        )
    };
    let setup = format!(
        "let a: Array<f32> ={};let b: Array<f32> ={};let answer=work(a,b,{rows}u32,{inner}u32,{cols}u32);",
        f32_array(&a),
        f32_array(&b)
    );
    (
        finish_float(&leaf(true), &setup, &expected),
        finish_float(&leaf(false), &setup, &expected),
        json!({"dimensions":[rows,inner,cols],"policy":"tc-f16-f32-v1","inputs":"small integers exact in binary16; exact binary32 partial sums","a":a,"b":b}),
    )
}

/// Validate a `tc-f16-f32-v1` program: it is ordinary checked source, its entry is the recognized explicit
/// Tensor Core contraction, the strict twin passes the source reference
/// oracle, and the exact expectation satisfies the amended D5 bound under the
/// device module's own checker.
pub fn validate_tc(source: &str, twin: &str, entry: &str, shape: &Value) -> Result<(), String> {
    let contract = contraction_source::Contract::from_source(source.as_bytes(), entry)
        .map_err(|e| format!("tc recognizer: {e:?}"))?;
    if contract.policy().id() != "tc-f16-f32-v1" {
        return Err("tc program recognized under another numeric policy".into());
    }
    // Ordinary checking admits explicit `tc_mul_add` source on main, so a
    // held-out tc program must also be an ordinary checked program.
    stored_oracle(source).map_err(|e| format!("stored tc program: {e}"))?;
    validate_source("matrix", twin).map_err(|e| format!("strict twin: {e}"))?;
    let dims: Vec<u32> = shape["dimensions"]
        .as_array()
        .ok_or("tc dimensions missing")?
        .iter()
        .map(|d| d.as_u64().map(|d| d as u32).ok_or("tc dimension"))
        .collect::<Result<_, _>>()?;
    let words = |key: &str| -> Result<Vec<u32>, String> {
        shape[key]
            .as_array()
            .ok_or("tc inputs missing")?
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|v| (v as f32).to_bits())
                    .ok_or_else(|| "tc input".to_string())
            })
            .collect()
    };
    let (a, b) = (words("a")?, words("b")?);
    let dims = [dims[0], dims[1], dims[2]];
    let reference =
        invar_device::tc::reference(dims, &a, &b).map_err(|e| format!("tc reference: {e:?}"))?;
    let [rows, inner, cols] = dims.map(|d| d as usize);
    let exact: Vec<u32> = (0..rows * cols)
        .map(|cell| {
            let (r, c) = (cell / cols, cell % cols);
            (0..inner)
                .map(|k| {
                    f64::from(f32::from_bits(a[r * inner + k]))
                        * f64::from(f32::from_bits(b[k * cols + c]))
                })
                .fold(0.0, |s, x| s + x) as f32
        })
        .map(f32::to_bits)
        .collect();
    if reference != exact {
        return Err("tc exact-sum reference differs from the independent host sum".into());
    }
    invar_device::tc::verify(dims, &a, &b, &reference).map_err(|e| format!("tc bound: {e:?}"))?;
    Ok(())
}

pub fn validate_source(family: &str, source: &str) -> Result<(), String> {
    let checked = check::check(
        front::parse(source.as_bytes()).map_err(|e| format!("parse: {e:?}"))?,
        true,
    )
    .map_err(|e| format!("check: {e:?}"))?;
    let compiled = mir::lower(&checked).map_err(|e| format!("MIR: {e:?}"))?;
    mir::verify(&compiled).map_err(|e| format!("MIR verify: {e:?}"))?;
    let entry = checked.program.functions[0].name.as_str();
    let pair = (family == "reduction")
        .then(|| invar::pair_reduction::extract(&compiled).ok())
        .flatten();
    match family {
        "reduction" if pair.is_some() => {
            let a = invar::admission::analyze(source.as_bytes(), &compiled)
                .map_err(|e| format!("admission: {e:?}"))?;
            if !a
                .regions()
                .iter()
                .any(|r| matches!(r.execution(), invar::plan::Execution::PairReduction(_)))
            {
                return Err("pair reduction extracted but not admitted".into());
            }
        }
        "map" | "reduction" | "scan" => {
            let a = invar::admission::analyze(source.as_bytes(), &compiled)
                .map_err(|e| format!("admission: {e:?}"))?;
            let kinds = match family {
                "map" => vec![invar::plan::PlanKind::Map, invar::plan::PlanKind::FusedMap],
                "reduction" => vec![invar::plan::PlanKind::Reduction],
                _ => vec![invar::plan::PlanKind::Scan],
            };
            // The separately evaluated oracle helper is not a workload kernel.
            let workload: Vec<_> = a
                .regions()
                .iter()
                .filter(|r| r.function() == "main")
                .collect();
            if workload.is_empty() || !workload.iter().all(|r| kinds.contains(&r.plan_kind())) {
                return Err("unexpected legacy family plan".into());
            }
        }
        "transpose" => {
            invar::transpose_source::Contract::from_checked(&checked, entry)
                .map_err(|e| format!("recognizer: {e:?}"))?;
        }
        "stencil" => {
            invar::stencil_source::Contract::from_checked(&checked, entry)
                .map_err(|e| format!("recognizer: {e:?}"))?;
        }
        "matrix" => {
            invar::tensor::Contraction::from_checked(&checked, entry)
                .map(|_| ())
                .or_else(|scalar| {
                    // M4-03's direct-cell strict form, `c[i,j] = c[i,j] + a*b`.
                    contraction_source::Contract::from_source(source.as_bytes(), entry)
                        .map_err(|cell| {
                            format!("recognizer: {scalar:?}; direct-cell recognizer: {cell:?}")
                        })
                        .and_then(|c| {
                            (c.policy().id() == "strict-f32-v1")
                                .then_some(())
                                .ok_or_else(|| "direct-cell form under another policy".to_string())
                        })
                })?;
        }
        "compaction" => {
            invar::compaction_source::Contract::from_source(source.as_bytes(), entry)
                .map_err(|e| format!("recognizer: {e:?}"))?;
        }
        "segmented" => {
            invar::segmented_source::Contract::from_source(source.as_bytes(), entry)
                .map(|_| ())
                .or_else(|reduce| {
                    invar::segmented_device::MapContract::from_source(source.as_bytes(), entry)
                        .map(|_| ())
                        .map_err(|map| {
                            format!("reduce recognizer: {reduce:?}; map recognizer: {map:?}")
                        })
                })?;
        }
        "gather_scatter" => {
            invar::indexed_source::Contract::from_checked(source.as_bytes(), &checked, entry)
                .map(|_| ())
                .or_else(|indexed_error| {
                    invar::histogram_source::Contract::from_checked(source.as_bytes(), &checked, entry)
                        .map(|_| ())
                        .map_err(|histogram_error| format!("indexed recognizer: {indexed_error:?}; histogram recognizer: {histogram_error:?}"))
                })?;
        }
        "frontier" => {
            frontier_source::Contract::from_checked(source.as_bytes(), &checked, entry)
                .map_err(|e| format!("recognizer: {e:?}"))?;
        }
        _ => return Err("unknown family".into()),
    }
    observe_oracle(&checked, &compiled)
}

/// Check, lower and verify a stored program, then run its own oracle.
pub fn stored_oracle(source: &str) -> Result<(), String> {
    let checked = check::check(
        front::parse(source.as_bytes()).map_err(|e| format!("parse: {e:?}"))?,
        true,
    )
    .map_err(|e| format!("check: {e:?}"))?;
    let compiled = mir::lower(&checked).map_err(|e| format!("MIR: {e:?}"))?;
    mir::verify(&compiled).map_err(|e| format!("MIR verify: {e:?}"))?;
    observe_oracle(&checked, &compiled)
}

/// The stored program's own oracle: `main` returns 0 on the reference
/// evaluator, and the MIR evaluator agrees on result, fuel and cleanup.
fn observe_oracle(checked: &check::CheckedProgram, compiled: &mir::Program) -> Result<(), String> {
    let reference = eval::observe(checked, FUEL);
    if reference.result != Ok(eval::Value::I32(0)) {
        return Err(format!("reference oracle: {:?}", reference.result));
    }
    let lowered = mir_eval::observe(compiled, FUEL);
    if reference.result != lowered.result
        || reference.steps_used != lowered.steps_used
        || reference.cleanup_trace != lowered.cleanup_trace
    {
        return Err("reference/MIR observation mismatch".into());
    }
    Ok(())
}

fn negative_controls() -> Result<Value, String> {
    let mut controls = Vec::new();
    for mode in 0..3 {
        let store = match mode {
            0 => "out[i]=a[ix[i]];",
            1 => "out[ix[i]]=a[i];",
            _ => "out[ix[i]]=wrapping_add(out[ix[i]],a[i]);",
        };
        let leaf = format!(
            "fn work(a:read Array<u32>,ix:read Array<u32>{})->Array<u32>{{var out=filled_u32({},0u32);for i in 0u32..len(ix){{{store}}}return out;}}",
            if mode == 0 { "" } else { ",n:u32" },
            if mode == 0 { "len(ix)" } else { "n" }
        );
        let source = format!(
            "{leaf} fn main()->i32{{let a:Array<u32> =[1u32];let ix:Array<u32> =[2u32];let answer=work(a,ix{});return 0;}}",
            if mode == 0 { "" } else { ",1u32" }
        );
        let checked = check::check(
            front::parse(source.as_bytes()).map_err(|e| format!("{e:?}"))?,
            true,
        )
        .map_err(|e| format!("{e:?}"))?;
        let refused = if mode == 2 {
            let c = invar::histogram_source::Contract::from_checked(
                source.as_bytes(),
                &checked,
                "work",
            )
            .map_err(|e| format!("{e:?}"))?;
            matches!(
                c.candidate(&[1], &[2], 1),
                Err(invar::histogram_source::Error::Bounds { iteration: 0 })
            )
        } else {
            let c =
                invar::indexed_source::Contract::from_checked(source.as_bytes(), &checked, "work")
                    .map_err(|e| format!("{e:?}"))?;
            matches!(
                c.candidate(&[1], &[2], if mode == 0 { None } else { Some(1) }),
                Err(invar::indexed_source::Error::Bounds { iteration: 0 })
            )
        };
        let reference = eval::observe(&checked, FUEL);
        let m = mir::lower(&checked).map_err(|e| format!("{e:?}"))?;
        let lowered = mir_eval::observe(&m, FUEL);
        if !refused
            || !matches!(
                reference.result,
                Err(eval::EvalError::Trap {
                    category: eval::TrapCategory::IndexOutOfBounds,
                    ..
                })
            )
            || reference.result != lowered.result
            || reference.steps_used != lowered.steps_used
            || reference.cleanup_trace != lowered.cleanup_trace
        {
            return Err("bounds negative control failed".into());
        }
        controls.push(json!({"id":format!("indexed-bounds-{mode}"),"source":source,"source_sha256":sha256(source.as_bytes()),"expected":"INDEX_OUT_OF_BOUNDS","oracle_status":"passed","recognizer_status":"recognized","candidate_status":"bounds-refused"}));
    }
    let source = "fn main()->i32{let links:Array<u32> =[1u32,2u32,3u32,3u32];var index=0u32;for step in 0u32..3u32{index=links[index];}if index!=3u32{return 1;}return 0;}";
    let checked = check::check(
        front::parse(source.as_bytes()).map_err(|e| format!("{e:?}"))?,
        true,
    )
    .map_err(|e| format!("{e:?}"))?;
    let m = mir::lower(&checked).map_err(|e| format!("{e:?}"))?;
    let a = invar::admission::analyze(source.as_bytes(), &m).map_err(|e| format!("{e:?}"))?;
    if eval::observe(&checked, FUEL).result != Ok(eval::Value::I32(0))
        || a.regions().is_empty()
        || !a
            .regions()
            .iter()
            .all(|r| r.plan_kind() == invar::plan::PlanKind::Serial)
    {
        return Err("pointer-chasing negative control was not serial".into());
    }
    controls.push(json!({"id":"frontier-pointer-chasing","source":source,"source_sha256":sha256(source.as_bytes()),"oracle_status":"passed","candidate_status":"serial-fallback","first_rejected_obligations":a.regions().iter().map(|r|r.first_rejected_obligation()).collect::<Vec<_>>()}));
    Ok(json!(controls))
}

pub fn generate(seed: u64) -> Result<Value, String> {
    if seed == 0 {
        return Err("xorshift64 seed must be nonzero".into());
    }
    let legacy = if seed == corpus::SEED {
        corpus::generate()?
    } else {
        corpus::generate_seeded(seed)?
    };
    let mut programs = legacy["programs"]
        .as_array()
        .ok_or("legacy programs missing")?
        .clone();
    for p in &mut programs {
        p["oracle_shape"] = json!({"elements":3});
    }
    let mut state = seed;
    for family in &FAMILIES[..3] {
        for ordinal in 0..EXTRA_PER_FAMILY {
            let n = BOUNDARIES[ordinal % BOUNDARIES.len()];
            let source = match *family {
                "map" => map_boundary(&legacy["programs"][ordinal], n)?,
                "reduction" => reduction_boundary(&mut state, n, ordinal % 2 == 1),
                _ => scan_boundary(&mut state, n, ordinal % 3),
            };
            validate_source(family, &source)
                .map_err(|e| format!("{family}-boundary-{ordinal:03}: {e}"))?;
            programs.push(json!({"id":format!("{family}-boundary-{ordinal:03}"),"family":family,"source":source,"source_sha256":sha256(source.as_bytes()),"oracle_shape":{"elements":n},"oracle_status":"passed","recognizer_status":"recognized","stages":if *family=="map" {legacy["programs"][ordinal]["stages"].clone()}else{json!(1)}}));
        }
    }
    for family in &FAMILIES[3..] {
        for ordinal in 0..EXTRA_PER_FAMILY {
            let (mut source, shape) = match *family {
                "compaction" => compaction(&mut state, ordinal),
                "transpose" => transpose(&mut state, ordinal),
                "stencil" => stencil(&mut state, ordinal),
                "matrix" => matrix(&mut state, ordinal),
                "segmented" => segmented(&mut state, ordinal),
                "gather_scatter" => indexed(&mut state, ordinal),
                "frontier" => frontier(&mut state, ordinal),
                _ => unreachable!(),
            }
            .map_err(|e| format!("{family}-{ordinal:03}: {e}"))?;
            // Names vary without changing semantics or teaching recognizers case IDs.
            let function = format!("kernel_{ordinal}");
            source = source.replace("work", &function);
            validate_source(family, &source).map_err(|e| format!("{family}-{ordinal:03}: {e}"))?;
            programs.push(json!({"id":format!("{family}-{ordinal:03}"),"family":family,"split":if ordinal%4==0{"held_out"}else{"tuning"},"entry":function,"source":source,"source_sha256":sha256(source.as_bytes()),"oracle_shape":shape,"oracle_status":"passed","recognizer_status":"recognized"}));
        }
    }
    let mut form_state = seed ^ FORM_STREAM;
    for (family, form) in [
        ("reduction", "pair"),
        ("segmented", "map"),
        ("matrix", "tc"),
    ] {
        for ordinal in 0..EXTRA_FORM_COUNT {
            let id = format!("{family}-{form}-{ordinal:03}");
            let function = format!("kernel_{form}_{ordinal}");
            let (source, shape) = match form {
                "pair" => pair_reduction(&mut form_state, ordinal),
                "map" => segmented_map(&mut form_state, ordinal),
                _ => {
                    let (tc, twin, mut shape) = tc_matrix(&mut form_state, ordinal);
                    let (tc, twin) = (
                        tc.replace("work", &function),
                        twin.replace("work", &function),
                    );
                    validate_tc(&tc, &twin, &function, &shape).map_err(|e| format!("{id}: {e}"))?;
                    shape["strict_twin_sha256"] = json!(sha256(twin.as_bytes()));
                    (tc, shape)
                }
            };
            let source = source.replace("work", &function);
            if form != "tc" {
                validate_source(family, &source).map_err(|e| format!("{id}: {e}"))?;
            }
            programs.push(json!({"id":id,"family":family,"form":form,"entry":function,"source":source,"source_sha256":sha256(source.as_bytes()),"oracle_shape":shape}));
        }
    }
    // Split depends only on family ordinals and its own PRNG, never results.
    // Pin subtype representatives so parity of an ordinal cannot hide a law
    // or graph topology from the held-out set (the legacy modulo split did).
    let mut split_state = seed ^ 0x6865_6c64_6f75_7431;
    for family in FAMILIES {
        let indices: Vec<usize> = programs
            .iter()
            .enumerate()
            .filter(|(_, p)| p["family"] == family)
            .map(|(i, _)| i)
            .collect();
        // Each later form keeps two representatives (pair: identity and live
        // seed; segmented map: with and without the segment operand; tc).
        let base = EXTRA_PER_FAMILY;
        let required: &[usize] = match family {
            "map" => &[0, 15, 63, 79],
            "reduction" => &[0, 1, 80 + base, 81 + base],
            "scan" => &[0, 1, 2],
            "segmented" => &[0, 1, 2, 3, 4, base, base + 1],
            "compaction" => &[0, 1, 2, 3, 4],
            "matrix" => &[0, 1, 2, 3, base, base + 1],
            "gather_scatter" => &[0, 1, 2],
            _ => &[0, 1, 2, 3],
        };
        // The split unit is a distinct program: identical sources up to the
        // generated entry name (degenerate shapes such as an empty transpose,
        // and a few legacy scans, repeat) always share one split, otherwise a
        // held-out program would also be a tuning program. One quarter of the
        // distinct programs per family is held out.
        let classes: Vec<String> = indices
            .iter()
            .map(|&i| canonical_program(programs[i]["source"].as_str().unwrap_or("")))
            .collect();
        let mut distinct: Vec<&String> = Vec::new();
        for c in &classes {
            if !distinct.contains(&c) {
                distinct.push(c);
            }
        }
        let mut held_classes: Vec<&String> = Vec::new();
        for &o in required {
            if !held_classes.contains(&&classes[o]) {
                held_classes.push(&classes[o]);
            }
        }
        let mut remaining: Vec<u32> = (0..distinct.len() as u32)
            .filter(|&i| !held_classes.contains(&distinct[i as usize]))
            .collect();
        shuffle(&mut remaining, &mut split_state);
        let quota = (distinct.len() / 4).saturating_sub(held_classes.len());
        held_classes.extend(remaining.iter().take(quota).map(|&i| distinct[i as usize]));
        let held: Vec<usize> = (0..indices.len())
            .filter(|&o| held_classes.contains(&&classes[o]))
            .collect();
        // Each distinct program is counted once: its first member in ordinal
        // order carries the split, and later identical members are recorded as
        // duplicates of it, outside both the held-out and the tuning sets, so
        // no degenerate workload is weighted more than once.
        let mut representative: Vec<(&String, String)> = Vec::new();
        for (ordinal, index) in indices.into_iter().enumerate() {
            let class = &classes[ordinal];
            if let Some((_, id)) = representative.iter().find(|(c, _)| *c == class) {
                programs[index]["split"] = json!("duplicate");
                programs[index]["duplicate_of"] = json!(id);
            } else {
                representative.push((
                    class,
                    programs[index]["id"].as_str().unwrap_or("").to_owned(),
                ));
                programs[index]["split"] = json!(if held.contains(&ordinal) {
                    "held_out"
                } else {
                    "tuning"
                });
            }
            programs[index]["oracle_status"] = json!("passed");
            programs[index]["recognizer_status"] = json!("recognized");
        }
    }
    let mut counts = json!({});
    for family in FAMILIES {
        let count = programs.iter().filter(|p| p["family"] == family).count();
        let distinct = |split: Option<&str>| {
            programs
                .iter()
                .filter(|p| p["family"] == family && split.is_none_or(|s| p["split"] == s))
                .map(|p| canonical_program(p["source"].as_str().unwrap_or("")))
                .collect::<std::collections::BTreeSet<_>>()
                .len()
        };
        counts[family] = json!({"generated":count,"valid":count,"recognized":count,"distinct":distinct(None),"held_out_distinct":distinct(Some("held_out")),"pending":0,"ignored":0,"unavailable_form_count":if family=="transpose"{1}else{0},"held_out":programs.iter().filter(|p|p["family"]==family&&p["split"]=="held_out").count()});
    }
    let corpus = json!({"programs":programs});
    let held = corpus::held_out_hash(&corpus)?;
    let negatives = negative_controls()?;
    Ok(
        json!({"schema_version":1,"status":"proposal-pending-review","generator":"xorshift64-v2-all-families-proposal","seed":seed,"held_out_sha256":held,"hash_scope":"compact serde_json serialization of full held_out program entries in corpus order; proposal metadata is excluded","split_policy":{"held_out_ratio_proposal":"1/4 (25%); numeric ratio is not specified by M5-01","rule":"one quarter of the distinct programs per family (identical sources up to the generated entry name form one unit); the first member of a unit carries its split and later members are duplicates outside both sets; predeclared subtype representatives plus a seeded shuffle of the remaining units; independent of oracle results"},"program_counts":{"total":programs.len(),"held_out":programs.iter().filter(|p|p["split"]=="held_out").count(),"tuning":programs.iter().filter(|p|p["split"]=="tuning").count(),"duplicate":programs.iter().filter(|p|p["split"]=="duplicate").count()},"family_counts":counts,"count_scope":"eligible bounded CPU oracle programs; unavailable forms are counted separately","unavailable_forms":[{"family":"transpose","form":"strided read view","status":"unavailable","reason":"the merged transpose source recognizer (M4-02, #104) admits direct row-major transpose only"}],"available_forms":[{"family":"reduction","form":"ordered affine pair composition","source":"#115 pair_reduction::extract plus Execution::PairReduction admission"},{"family":"segmented","form":"per-segment map","source":"#100 segmented_device::MapContract"},{"family":"matrix","form":"explicit Tensor Core tc-f16-f32-v1","source":"#110 contraction_source tc_mul_add; strict twin reference-evaluated, exact-sum reference checked independently"}],"shape_proposals":shape_proposals(),"pending_acceptance":["Review and bind proposed sizes for unsized shape classes","After M4-02..05 merge, rerun source recognizers and CPU oracles on the merged revision","Large performance recipe instantiation, resource admission and numeric adversarial coverage remain unexecuted","Strided-view transpose recognition remains pending family support (M4-02); not approximated by direct transpose","Freeze through reviewed manifest version bump before M4-06; this command cannot freeze"],"gpu_performance":"not_run","negative_controls":negatives,"programs":programs}),
    )
}

/// Store the candidate held-out set; tuning sources remain reproducible from
/// `generate` and do not need a second committed copy.
pub fn generate_proposal(seed: u64) -> Result<Value, String> {
    let mut candidate = generate(seed)?;
    candidate["programs"]
        .as_array_mut()
        .ok_or("missing generated programs")?
        .retain(|p| p["split"] == "held_out");
    candidate["programs_scope"] =
        json!("held_out_only; all tuning programs are regenerated by the same seeded generator");
    Ok(candidate)
}

/// Anti-overfitting boundary. Tuning, calibration and any later M4-06 search
/// see programs only through `tuning_programs` or `HeldOutIndex::admit`; both
/// refuse held-out programs, renamed copies of them and their data inputs.
pub struct HeldOutIndex {
    sources: std::collections::BTreeSet<String>,
    programs: Vec<std::collections::BTreeSet<String>>,
    inputs: std::collections::BTreeSet<String>,
}

/// Source tokens: identifiers, numeric literals with their suffix, and single
/// punctuation characters. Whitespace and layout never reach a token.
fn tokens(source: &str) -> Vec<&str> {
    let bytes = source.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        let start = i;
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_alphanumeric() || c == b'_' {
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric()
                    || bytes[i] == b'_'
                    || (bytes[i] == b'.'
                        && i + 1 < bytes.len()
                        && bytes[i + 1].is_ascii_digit()
                        && bytes[start].is_ascii_digit()))
            {
                i += 1;
            }
        } else {
            i += source[i..].chars().next().map_or(1, char::len_utf8);
        }
        out.push(&source[start..i]);
    }
    out
}

/// Canonical forms of every top-level function: each name the program
/// declares (functions, parameters, `let`, `var` and `for` bindings) is
/// replaced by its order of first use inside that function, so renaming or
/// reformatting a program, or wrapping it in other functions, keeps its
/// functions' identities. Keywords, types and intrinsics are kept.
pub fn canonical_functions(source: &str) -> std::collections::BTreeSet<String> {
    let toks = tokens(source);
    let mut declared = std::collections::BTreeSet::new();
    for (k, t) in toks.iter().enumerate() {
        if matches!(*t, "fn" | "let" | "var" | "for")
            && let Some(name) = toks.get(k + 1)
        {
            declared.insert(*name);
        }
        if matches!(*t, "(" | ",")
            && toks.get(k + 2) == Some(&":")
            && let Some(name) = toks.get(k + 1)
        {
            declared.insert(*name);
        }
    }
    let mut functions = std::collections::BTreeSet::new();
    let mut k = 0;
    while k < toks.len() {
        if toks[k] != "fn" {
            k += 1;
            continue;
        }
        let (mut depth, mut end) = (0usize, k);
        for (j, t) in toks.iter().enumerate().skip(k) {
            match *t {
                "{" => depth += 1,
                "}" => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        end = j;
                        break;
                    }
                }
                _ => {}
            }
            end = j;
        }
        let mut names: Vec<&str> = Vec::new();
        let canonical: Vec<String> = toks[k..=end]
            .iter()
            .map(|t| {
                if declared.contains(t) {
                    let at = names.iter().position(|n| n == t).unwrap_or_else(|| {
                        names.push(t);
                        names.len() - 1
                    });
                    format!("v{at}")
                } else {
                    (*t).to_owned()
                }
            })
            .collect();
        functions.insert(sha256(canonical.join(" ").as_bytes()));
        k = end + 1;
    }
    functions
}

/// One digest for a whole program from its canonical functions: programs
/// that differ only in names or layout are the same program.
pub fn canonical_program(source: &str) -> String {
    let functions: Vec<String> = canonical_functions(source).into_iter().collect();
    sha256(functions.join(",").as_bytes())
}

/// Distinctive literal data arrays: at least eight elements and three
/// distinct values. Shorter or low-cardinality literals (shape parameters,
/// segment offsets, 0/1 masks, one-cell matrices) occur in both splits by
/// construction and carry no held-out identity; the program itself is
/// protected by `canonical_functions`. This is a second guard only.
fn data_inputs(source: &str) -> Vec<String> {
    let compact: String = source.chars().filter(|c| !c.is_whitespace()).collect();
    compact
        .split('[')
        .skip(1)
        .filter_map(|tail| tail.split(']').next())
        .filter(|body| {
            let items: Vec<&str> = body.split(',').collect();
            let distinct: std::collections::BTreeSet<&str> = items.iter().copied().collect();
            items.len() >= 8
                && distinct.len() >= 3
                && items.iter().all(|x| {
                    !x.is_empty()
                        && x.trim_start_matches('-')
                            .chars()
                            .all(|c| c.is_ascii_digit() || matches!(c, '.' | 'u' | 'f' | '3' | '2'))
                })
        })
        .map(str::to_owned)
        .collect()
}

impl HeldOutIndex {
    pub fn from_programs(programs: &Value) -> Result<Self, String> {
        let mut index = Self {
            sources: Default::default(),
            programs: Vec::new(),
            inputs: Default::default(),
        };
        for p in programs.as_array().ok_or("held-out programs missing")? {
            if p["split"] != "held_out" {
                continue;
            }
            let source = p["source"].as_str().ok_or("held-out source missing")?;
            index.sources.insert(sha256(source.as_bytes()));
            index.programs.push(canonical_functions(source));
            index.inputs.extend(data_inputs(source));
        }
        if index.sources.is_empty() {
            return Err("empty held-out index".into());
        }
        Ok(index)
    }

    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Refuse a candidate tuning or calibration program that is, or embeds the
    /// data of, a held-out program.
    pub fn admit(&self, source: &str) -> Result<(), String> {
        if self.sources.contains(&sha256(source.as_bytes())) {
            return Err("held-out program refused on a tuning path".into());
        }
        // Every function of a held-out program, canonically renamed, inside
        // the candidate: the program itself, renamed, reformatted or wrapped.
        let candidate = canonical_functions(source);
        if self.programs.iter().any(|held| held.is_subset(&candidate)) {
            return Err("renamed or embedded held-out program refused on a tuning path".into());
        }
        if data_inputs(source).iter().any(|x| self.inputs.contains(x)) {
            return Err("held-out input data refused on a tuning path".into());
        }
        Ok(())
    }
}

/// The only corpus view a tuning path may use: the tuning split, regenerated
/// from the seed, each program admitted against the held-out index.
pub fn tuning_programs(seed: u64) -> Result<(HeldOutIndex, Vec<Value>), String> {
    let all = generate(seed)?;
    let index = HeldOutIndex::from_programs(&all["programs"])?;
    let mut tuning = Vec::new();
    for p in all["programs"].as_array().ok_or("programs missing")? {
        if p["split"] != "tuning" {
            continue;
        }
        let source = p["source"].as_str().ok_or("tuning source missing")?;
        index
            .admit(source)
            .map_err(|e| format!("{}: {e}", p["id"].as_str().unwrap_or("?")))?;
        tuning.push(p.clone());
    }
    Ok((index, tuning))
}
