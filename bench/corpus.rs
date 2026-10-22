use invar::{admission, check::check, eval, front::parse, mir, plan::PlanKind};
use serde_json::{Value, json};

use super::hash::sha256;

pub const SEED: u64 = 0x5a17_2026_0901_0001;
pub const COUNT: usize = 240;

pub(super) fn next(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 16) as u32
}

fn map_source(state: &mut u64, ordinal: usize) -> (String, usize, usize, usize) {
    let stages = 1 + ordinal % 16;
    let inputs = (1 + (ordinal / 16) % 4).min(stages + 1);
    let outputs = 1 + (ordinal / 4) % 4;
    let mut source = String::from("fn main() -> i32 { ");
    let mut values = Vec::new();
    for input in 0..inputs {
        let value = 1 + next(state) % 9;
        values.push(value);
        source.push_str(&format!(
            "let xs{input}: Array<u32> = filled_u32(3u32,{value}u32); "
        ));
    }
    let mut current = "xs0".to_string();
    let mut oracle = values[0];
    for stage in 0..stages {
        let other = if inputs == 1 {
            0
        } else {
            1 + stage % (inputs - 1)
        };
        let mul = next(state) & 1 == 0;
        let op = if mul { "wrapping_mul" } else { "wrapping_add" };
        oracle = if mul {
            oracle.wrapping_mul(values[other])
        } else {
            oracle.wrapping_add(values[other])
        };
        source.push_str(&format!("var out{stage}: Array<u32> = filled_u32(3u32,0u32); for i{stage} in 0u32 .. 3u32 {{ out{stage}[i{stage}] = {op}({current}[i{stage}],xs{other}[i{stage}]); }} let stable{stage}: Array<u32> = move out{stage}; "));
        current = format!("stable{stage}");
    }
    for output in 0..outputs {
        let value = (output as u32) + 1;
        source.push_str(&format!("var result{output}: Array<u32> = filled_u32(3u32,0u32); for j{output} in 0u32 .. 3u32 {{ result{output}[j{output}] = wrapping_add({current}[j{output}],{value}u32); }} "));
    }
    for output in 0..outputs {
        let value = (output as u32) + 1;
        for index in 0..3 {
            source.push_str(&format!(
                "if result{output}[{index}u32] != {}u32 {{ return 1; }} ",
                oracle.wrapping_add(value)
            ));
        }
    }
    source.push_str("return 0; }");
    (source, stages, inputs, outputs)
}

fn reduction_source(state: &mut u64, ordinal: usize) -> (String, usize, usize, usize) {
    let a = 1 + next(state) % 9;
    let b = 1 + next(state) % 9;
    let c = 1 + next(state) % 9;
    let seed = 1 + next(state) % 9;
    let product = ordinal % 2 == 1;
    let op = if product {
        "wrapping_mul"
    } else {
        "wrapping_add"
    };
    let expected = [a, b, c].into_iter().fold(seed, |x, y| {
        if product {
            x.wrapping_mul(y)
        } else {
            x.wrapping_add(y)
        }
    });
    (
        format!(
            "fn main() -> i32 {{ let xs: Array<u32> = [{a}u32,{b}u32,{c}u32]; var acc: u32 = {seed}u32; for i in 0u32 .. len(xs) {{ acc = {op}(acc,xs[i]); }} if acc != {expected}u32 {{ return 1; }} return 0; }}"
        ),
        1,
        1,
        1,
    )
}

fn scan_source(state: &mut u64, ordinal: usize) -> (String, usize, usize, usize) {
    let a = 1 + next(state) % 4;
    let b = 1 + next(state) % 9;
    let seed = 1 + next(state) % 9;
    let mut expected = seed;
    let mut before_values = Vec::new();
    let mut after_values = Vec::new();
    for _ in 0..3 {
        before_values.push(expected);
        expected = a.wrapping_mul(expected).wrapping_add(b);
        after_values.push(expected);
    }
    let mode = ordinal % 3;
    let body = match mode {
        1 => "out[i] = state; state = wrapping_add(wrapping_mul(aa[i],state),bb[i]);",
        2 => "state = wrapping_add(wrapping_mul(aa[i],state),bb[i]); out[i] = state;",
        _ => "state = wrapping_add(wrapping_mul(aa[i],state),bb[i]);",
    };
    let before = if mode != 0 {
        "var out: Array<u32> = filled_u32(3u32,0u32);"
    } else {
        ""
    };
    let mut check = String::new();
    if mode != 0 {
        let values = if mode == 1 {
            before_values
        } else {
            after_values
        };
        for (index, value) in values.into_iter().enumerate() {
            check.push_str(&format!(
                "if out[{index}u32] != {value}u32 {{ return 1; }} "
            ));
        }
    }
    (
        format!(
            "fn main() -> i32 {{ let aa: Array<u32> = filled_u32(3u32,{a}u32); let bb: Array<u32> = filled_u32(3u32,{b}u32); var state: u32 = {seed}u32; {before} for i in 0u32 .. len(aa) {{ {body} }} {check} if state != {expected}u32 {{ return 1; }} return 0; }}"
        ),
        1,
        2,
        if mode != 0 { 2 } else { 1 },
    )
}

pub fn generate() -> Result<Value, String> {
    generate_seeded(SEED)
}

pub fn generate_seeded(seed: u64) -> Result<Value, String> {
    let mut state = seed;
    let mut programs = Vec::with_capacity(COUNT);
    for ordinal in 0..COUNT {
        let family = match ordinal / 80 {
            0 => "map",
            1 => "reduction",
            _ => "scan",
        };
        let (source, stages, inputs, outputs) = match family {
            "map" => map_source(&mut state, ordinal),
            "reduction" => reduction_source(&mut state, ordinal),
            _ => scan_source(&mut state, ordinal),
        };
        let checked = check(
            parse(source.as_bytes()).map_err(|e| format!("{ordinal}: parse: {e:?}"))?,
            true,
        )
        .map_err(|e| format!("{ordinal}: check: {e:?}"))?;
        let program = mir::lower(&checked).map_err(|e| format!("{ordinal}: MIR: {e:?}"))?;
        let admitted = admission::analyze(source.as_bytes(), &program)
            .map_err(|e| format!("{ordinal}: admission: {e:?}"))?;
        let expected_plan = match family {
            "map" => [PlanKind::Map, PlanKind::FusedMap].as_slice(),
            "reduction" => [PlanKind::Reduction].as_slice(),
            _ => [PlanKind::Scan].as_slice(),
        };
        if admitted.regions().is_empty()
            || !admitted
                .regions()
                .iter()
                .all(|r| expected_plan.contains(&r.plan_kind()))
        {
            return Err(format!(
                "{ordinal}: {family} was not fully admitted: {:?}",
                admitted
                    .regions()
                    .iter()
                    .map(|r| r.plan_kind())
                    .collect::<Vec<_>>()
            ));
        }
        if eval::observe(&checked, 100_000).result != Ok(eval::Value::I32(0)) {
            return Err(format!("{ordinal}: independent small-input oracle failed"));
        }
        programs.push(json!({
            "id": format!("{family}-{ordinal:03}"), "family": family,
            "split": if ordinal % 4 == 0 {"held_out"} else {"tuning"},
            "stages": stages, "inputs": inputs, "outputs": outputs,
            "source_sha256": sha256(source.as_bytes()), "source": source,
        }));
    }
    Ok(json!({"schema_version":1,"generator":"xorshift64-v1","seed":seed,"programs":programs}))
}

pub fn held_out_hash(corpus: &Value) -> Result<String, String> {
    let programs = corpus["programs"].as_array().ok_or("missing programs")?;
    let held: Vec<&Value> = programs
        .iter()
        .filter(|p| p["split"] == "held_out")
        .collect();
    serde_json::to_vec(&held)
        .map(|bytes| sha256(&bytes))
        .map_err(|e| e.to_string())
}
