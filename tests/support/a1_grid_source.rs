pub fn transpose(floating: bool, rows: u32, cols: u32, short: bool, reached: bool) -> String {
    let (ty, fill, zero) = if floating {
        ("f32", "filled_f32", "0.0")
    } else {
        ("u32", "filled_u32", "0u32")
    };
    let n = if short { 1 } else { rows * cols };
    let values = if n == 0 {
        format!("{fill}(0u32,{zero})")
    } else {
        format!(
            "[{}]",
            (0..n)
                .map(|i| if floating {
                    format!("{}.25", i % 97)
                } else {
                    format!("{}u32", i.wrapping_mul(2654435761))
                })
                .collect::<Vec<_>>()
                .join(",")
        )
    };
    let expected = if short {
        String::new()
    } else {
        (0..rows * cols)
            .map(|i| {
                let at = (i % rows) * cols + i / rows;
                let value = if floating {
                    format!("{}.25", at % 97)
                } else {
                    format!("{}u32", at.wrapping_mul(2654435761))
                };
                format!("if b[{i}u32]!={value} {{return 0;}}")
            })
            .collect::<String>()
    };
    let body = format!("let a:Array<{ty}> ={values};let b=flip(a,{rows}u32,{cols}u32);{expected}");
    format!(
        "fn flip(a:read Array<{ty}>,rows:u32,cols:u32)->Array<{ty}>{{var b={fill}(rows*cols,{zero});for r in 0u32..rows{{for c in 0u32..cols{{b[c*rows+r]=a[r*cols+c];}}}}return b;}}fn main()->i32{{ {} return 7;}}",
        if reached {
            body
        } else {
            format!("if false {{{body}}}")
        }
    )
}

pub fn stencil(clamp: bool, floating: bool, rows: u32, cols: u32, steps: u32) -> String {
    let (ty, fill, zero) = if floating {
        ("f32", "filled_f32", "0.0")
    } else {
        ("u32", "filled_u32", "0u32")
    };
    let edge = |name: &str| {
        if clamp {
            format!("else{{{name}=a[r*cols+c];}}")
        } else {
            String::new()
        }
    };
    let sum = if floating {
        "(((a[r*cols+c]+n)+s)+w)+e"
    } else {
        "wrapping_add(wrapping_add(wrapping_add(wrapping_add(a[r*cols+c],n),s),w),e)"
    };
    let function = format!(
        "fn step(a:read Array<{ty}>,rows:u32,cols:u32)->Array<{ty}>{{var b={fill}(rows*cols,{zero});for r in 0u32..rows{{for c in 0u32..cols{{var n={zero};if r>0u32{{n=a[(r-1u32)*cols+c];}}{ne}var s={zero};if r+1u32<rows{{s=a[(r+1u32)*cols+c];}}{se}var w={zero};if c>0u32{{w=a[r*cols+(c-1u32)];}}{we}var e={zero};if c+1u32<cols{{e=a[r*cols+(c+1u32)];}}{ee}b[r*cols+c]={sum};}}}}return b;}}",
        ne = edge("n"),
        se = edge("s"),
        we = edge("w"),
        ee = edge("e")
    );
    let mut values = (0..rows * cols).map(|i| i + 1).collect::<Vec<_>>();
    let input = if values.is_empty() {
        format!("{fill}(0u32,{zero})")
    } else {
        format!(
            "[{}]",
            values
                .iter()
                .map(|v| if floating {
                    format!("{v}.0")
                } else {
                    format!("{v}u32")
                })
                .collect::<Vec<_>>()
                .join(",")
        )
    };
    for _ in 0..steps {
        values = (0..rows * cols)
            .map(|i| {
                let (r, c) = ((i / cols) as i64, (i % cols) as i64);
                [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
                    .into_iter()
                    .fold(0u32, |sum, (r, c)| {
                        if clamp || (r >= 0 && c >= 0 && r < i64::from(rows) && c < i64::from(cols))
                        {
                            sum.wrapping_add(
                                values[(r.clamp(0, i64::from(rows) - 1) as u32 * cols
                                    + c.clamp(0, i64::from(cols) - 1) as u32)
                                    as usize],
                            )
                        } else {
                            sum
                        }
                    })
            })
            .collect();
    }
    let expected = values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            format!(
                "if cur[{i}u32]!={}{{return 0;}}",
                if floating {
                    format!("{v}.0")
                } else {
                    format!("{v}u32")
                }
            )
        })
        .collect::<String>();
    format!(
        "{function}fn main()->i32{{var cur:Array<{ty}> ={input};for t in 0u32..{steps}u32{{let next=step(cur,{rows}u32,{cols}u32);cur=move next;}}{expected}return 7;}}"
    )
}
