//! Per-family held-out program generators and boundary recipes.

use super::*;

// Full boundary-size CPU witnesses augment the unchanged three-element corpus.
pub(super) const BOUNDARIES: [u32; 9] = [0, 1, 31, 32, 33, 255, 256, 257, 4093];
// ponytail: this adapter accepts the existing generator spelling; return
// structured map recipes if that spelling changes.
pub(super) fn map_boundary(template: &Value, n: u32) -> Result<String, String> {
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
pub(super) fn reduction_boundary(state: &mut u64, n: u32, product: bool) -> String {
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
pub(super) fn scan_boundary(state: &mut u64, n: u32, mode: usize) -> String {
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

pub(super) fn compaction(state: &mut u64, ordinal: usize) -> Result<(String, Value), String> {
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
pub(super) fn transpose(state: &mut u64, ordinal: usize) -> Result<(String, Value), String> {
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
pub(super) fn stencil(state: &mut u64, ordinal: usize) -> Result<(String, Value), String> {
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
pub(super) fn matrix(state: &mut u64, ordinal: usize) -> Result<(String, Value), String> {
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
pub(super) fn segmented(state: &mut u64, ordinal: usize) -> Result<(String, Value), String> {
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
pub(super) fn indexed(state: &mut u64, ordinal: usize) -> Result<(String, Value), String> {
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
pub(super) fn frontier(state: &mut u64, ordinal: usize) -> Result<(String, Value), String> {
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
    let leaf = include_str!("../../../bench/general-net-v01/current/frontier_graph_a.inv")
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

pub(super) const PAIR_LEAF: &str = "fn work(aa: read Array<u32>, bb: read Array<u32>, n: u32, seed_a: u32, seed_b: u32) -> Array<u32> { var sa = seed_a; var sb = seed_b; for i in 0u32..n { sa = wrapping_mul(aa[i],sa); sb = wrapping_add(wrapping_mul(aa[i],sb),bb[i]); } return [sa,sb]; }";
pub(super) fn pair_reduction(state: &mut u64, ordinal: usize) -> (String, Value) {
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
pub(super) fn segmented_map(state: &mut u64, ordinal: usize) -> (String, Value) {
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
pub(super) fn tc_matrix(state: &mut u64, ordinal: usize) -> (String, String, Value) {
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
