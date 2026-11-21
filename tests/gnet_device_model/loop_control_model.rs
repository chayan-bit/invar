//! Issue #261: `break` and `continue` on the compiled device route. Each
//! program goes through `Plan::from_source` for CUDA and Metal and runs on the
//! CPU group model, at full fuel and one step short, observation for
//! observation with the bounded net oracle.
use super::device::{FUEL, agree, config, drive, reference};
use invar::{
    eval::Value,
    general_net::{
        compiled::Target,
        device_host::{Limits, Plan, cpu::CpuModel},
    },
};

const SOURCES: &[&str] = &[
    // Labelled exits across a nested `while`, with an owner per outer trip.
    "fn run(n: u32) -> u32 {
        var s = 0u32;
        outer: for i in 0u32 .. n {
            let row = filled_u32(3u32, i);
            var j = 0u32;
            while j < 3u32 {
                j += 1u32;
                if wrapping_add(row[0u32], j) % 4u32 == 0u32 { continue outer; }
                if s > 40u32 { break outer; }
                s = wrapping_add(s, wrapping_mul(i, j));
            }
        }
        return s;
    }",
    // An innermost `continue`, and a `while true` left only by `break`.
    "fn run(n: u32) -> u32 {
        var s = 0u32;
        for i in 0u32 .. n { if i % 3u32 == 1u32 { continue; } s = wrapping_add(s, i); }
        var k = 0u32;
        while true { k += 1u32; if k > n { break; } s = wrapping_add(s, 100u32); }
        return s;
    }",
    // A trap after a `continue`, reached only when `n > 3`.
    "fn run(n: u32) -> u32 {
        var s = 0u32;
        for i in 0u32 .. n { if i == 1u32 { continue; } s += 12u32 / (3u32 - i); }
        return s;
    }",
];

#[test]
fn loop_control_on_the_device_model_agrees_with_the_oracle() {
    for target in [Target::Cuda, Target::Metal] {
        for source in SOURCES {
            let plan = Plan::from_source(source.as_bytes(), "run", target, Limits::default())
                .unwrap_or_else(|e| panic!("{target:?} admission {e:?}\n{source}"));
            for n in [0u32, 3, 9] {
                let args = [Value::U32(n)];
                let full = reference(&plan, &args, config(FUEL, 1_000_000, 16, 64, 2));
                let steps = full.semantic.steps_used;
                for fuel in [FUEL, steps.saturating_sub(1).max(1)] {
                    let expected = reference(&plan, &args, config(fuel, 1_000_000, 16, 64, 2));
                    let cfg = config(fuel, expected.events.len().max(1), 16, 64, 2);
                    let context = format!("{target:?} n={n} fuel={fuel}\n{source}");
                    let (got, _) = drive(&plan, &mut CpuModel::default(), &args, cfg)
                        .unwrap_or_else(|e| panic!("{context}: {e:?}"));
                    agree(&context, &plan, &got, &expected);
                }
            }
        }
    }
}
