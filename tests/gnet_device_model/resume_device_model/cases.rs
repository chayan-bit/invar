use super::*;

#[test]
fn resume_device_model_matches_cpu_every_epoch_and_oracle() {
    let mut observed = Vec::new();
    let mut statuses = BTreeSet::new();
    for case in CASES {
        let l = case.limits();
        let net = lowered(case.source);
        let cpu = emit(&net, case, compiled::Target::Cpu);
        let expected: Vec<_> = case
            .roots
            .iter()
            .map(|n| {
                if case.backend.iter().any(|(root, _)| root == n) {
                    None
                } else {
                    Some(expected(&net, case.entry, &[eval::Value::U32(*n)], l))
                }
            })
            .collect();
        for target in [compiled::Target::Cuda, compiled::Target::Metal] {
            let dev = emit(&net, case, target);
            assert_eq!(dev.target(), target);
            assert!(dev.source().contains("invar_gnet_resume_v3"));
            let model = build(
                &format!("{}-{target:?}", case.name),
                &harness(&cpu, &dev, case.roots, l),
            );
            for quantum in [1, 3, 17] {
                let label = format!("{}-{target:?}-{quantum}", case.name);
                let run = model.execute(&label, quantum);
                assert_eq!(run.outputs.len(), case.roots.len(), "{label}");
                for (lane, root) in case.roots.iter().enumerate() {
                    let output = &run.outputs[lane];
                    match &expected[lane] {
                        Some((header, trace)) => {
                            assert_eq!(&output[..9], &header[..], "{label} root {lane}");
                            assert_eq!(&run.traces[lane], trace, "{label} root {lane}");
                        }
                        None => {
                            let (_, status) = case.backend.iter().find(|(r, _)| r == root).unwrap();
                            assert_eq!(output[0], *status, "{label} root {lane}");
                        }
                    }
                    statuses.insert(output[0]);
                }
                // Epochs advance every unfinished root by one quantum: pending
                // roots never increase, and the final epoch leaves none.
                assert!(run.pending.windows(2).all(|w| w[1] <= w[0]), "{label}");
                assert_eq!(run.pending.first(), Some(&(case.roots.len() as u64)));
                assert_eq!(run.pending.last(), Some(&0));
                assert_eq!(run.pending.len() as u64, run.epochs + 1);
                observed.push((label, run.epochs));
            }
        }
    }
    // Success; overflow, division and index traps; fuel, call-depth,
    // transition, trace and source array-size limits; backend element,
    // allocation-slot and cleanup storage limits. Status 7 (invalid internal
    // state) has no ordinary-source producer; status 8 has its own test.
    assert_eq!(statuses, [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13].into());
    eprintln!("ABI3 device-model runs (label, epochs): {observed:?}");
}

#[test]
fn resume_device_model_invalid_quantum_finishes_with_status_8() {
    let case = &CASES[0];
    let net = lowered(case.source);
    let cpu = emit(&net, case, compiled::Target::Cpu);
    for target in [compiled::Target::Cuda, compiled::Target::Metal] {
        let dev = emit(&net, case, target);
        let model = build(
            &format!("status8-{target:?}"),
            &harness(&cpu, &dev, case.roots, case.limits()),
        );
        for quantum in [0, 1_000_001] {
            let label = format!("status8-{target:?}-{quantum}");
            let run = model.execute(&label, quantum);
            assert_eq!(run.epochs, 1, "{label}");
            assert_eq!(run.pending, [case.roots.len() as u64, 0], "{label}");
            assert!(
                run.outputs.iter().all(|o| o[0] == 8 && o[7] == 0),
                "{label}"
            );
        }
    }
}

/// Apply a text substitution to the device namespace only; the CPU reference
/// side of the generated model is left untouched.
fn mutate_device(text: &str, from: &str, to: &str) -> String {
    let start = text.find("namespace dev {").unwrap();
    let (head, tail) = text.split_at(start);
    assert!(tail.contains(from), "missing device text {from}");
    format!("{head}{}", tail.replacen(from, to, 1))
}

#[test]
fn resume_device_model_detects_device_lowering_mutations() {
    // Each mutation must fail deterministically with its specific exit code.
    let mutations: [(&str, usize, &str, &str, i32); 7] = [
        (
            "late-yield",
            0,
            "if (slice_steps==quantum) { GNET_YIELD; }",
            "if (slice_steps==quantum+1) { GNET_YIELD; }",
            TASK,
        ),
        ("no-lane-guard", 0, "if (lane>=config[0]) return;", "", TAIL),
        (
            "swapped-config",
            0,
            "config[1],config[2],config[3]",
            "config[2],config[1],config[3]",
            TASK,
        ),
        (
            "dropped-trace-offset",
            0,
            "(config[3]?trace+lane*config[3]*7:trace)",
            "trace",
            TRACE,
        ),
        // Filled below with case-specific heap strides. The far stride writes
        // just past the device heap allocation, so only ASan can (and must) stop it.
        ("short-heap-stride", 1, "", "", HEAP),
        ("far-heap-stride", 0, "", "", ASAN),
        // CUDA only: Metal takes the lane from thread_position_in_grid.
        (
            "no-block-offset",
            7,
            "(gnet_u)blockIdx.x * blockDim.x + threadIdx.x",
            "(gnet_u)threadIdx.x",
            TASK,
        ),
    ];
    for (name, case_index, from, to, code) in mutations {
        let case = &CASES[case_index];
        let net = lowered(case.source);
        let cpu = emit(&net, case, compiled::Target::Cpu);
        let heap = cpu.owned_layout().unwrap().heap_words();
        let (from, to) = match name {
            "short-heap-stride" => (
                format!("heap+lane*{heap}"),
                format!("heap+lane*{}", heap - 4),
            ),
            // Lane 1's first heap write (the live flag at offset 3 on task
            // start) lands exactly one element past the device heap vector of
            // LANES*H words plus the 64-word red zone.
            "far-heap-stride" => (
                format!("heap+lane*{heap}"),
                format!(
                    "heap+lane*{}",
                    case.roots.len().div_ceil(32) * 32 * heap + 64 - 3
                ),
            ),
            _ => (from.to_owned(), to.to_owned()),
        };
        let targets: &[_] = if name == "no-block-offset" {
            &[compiled::Target::Cuda]
        } else {
            &[compiled::Target::Cuda, compiled::Target::Metal]
        };
        for &target in targets {
            let dev = emit(&net, case, target);
            let text = harness(&cpu, &dev, case.roots, case.limits());
            let o = build(
                &format!("{name}-{target:?}"),
                &mutate_device(&text, &from, &to),
            )
            .run(3);
            assert!(!o.timed_out, "{name} {target:?}: {}", o.summary());
            // Windows' combined ASan/UBSan runtime keeps one common `exitcode`,
            // the last parsed (UBSAN_OPTIONS); the classified report below
            // still names AddressSanitizer.
            let exit = if cfg!(windows) && code == ASAN {
                UBSAN
            } else {
                code
            };
            assert_eq!(o.code, Some(exit), "{name} {target:?}: {}", o.summary());
            // A sanitizer exit must carry the expected classified report; a
            // comparison exit must carry none.
            let want = if code == ASAN {
                Sanitizer::Address("heap-buffer-overflow".into())
            } else {
                Sanitizer::None
            };
            assert_eq!(o.sanitizer, want, "{name} {target:?}: {}", o.summary());
            eprintln!(
                "mutation {name} {target:?}: exit {code}, sanitizer {:?}, {}",
                o.sanitizer,
                o.stdout.lines().last().unwrap_or("")
            );
        }
    }
}

#[test]
fn resume_device_model_numeric_owners_have_no_device_route() {
    // Forced device coverage must not silently fall back to the CPU ABI11 path.
    let net = lowered(
        "fn run(x: f32) -> Array<f32> { let a: Array<f32> = [x, -0.0]; return a; } fn main() -> i32 { return 0; }",
    );
    let proof = net.verify().unwrap();
    assert!(matches!(
        compiled::emit_resumable_owned_numeric(
            &net,
            &proof,
            "run",
            compiled::Target::Metal,
            compiled::Limits::default(),
            STORAGE
        ),
        Err(compiled::Error::NumericTarget)
    ));
    // Direct CUDA emission is the distinct ABI12 profile: the ABI3 device model
    // here never executes it, provenance admits it only as a compile receipt
    // without execution authority and no ABI12 runtime path exists
    // (cuda_owned_numeric_model tests), so no silent fallback.
    assert_eq!(
        compiled::emit_resumable_owned_numeric(
            &net,
            &proof,
            "run",
            compiled::Target::Cuda,
            compiled::Limits::default(),
            STORAGE
        )
        .unwrap()
        .abi_version(),
        12
    );
    assert!(
        compiled::emit_resumable_owned_numeric(
            &net,
            &proof,
            "run",
            compiled::Target::Cpu,
            compiled::Limits::default(),
            STORAGE
        )
        .is_ok()
    );
}

#[cfg(unix)]
#[test]
fn resume_device_model_bounded_runner_limits_and_classifies() {
    use crate::child_deadline;
    use std::{path::Path, time::Duration};
    // Wall deadline: the whole process group, including a backgrounded
    // descendant, is killed and the outcome is reported as timed out.
    let sh = Path::new("sh");
    let o = bounded(
        sh,
        &["-c".into(), "sleep 30 & echo $!; wait".into()],
        &[],
        Duration::from_millis(500),
        60,
    );
    assert!(
        o.timed_out && o.code.is_none() && o.signal == Some(9),
        "{o:?}"
    );
    let descendant = o.stdout.trim().to_owned();
    assert!(!descendant.is_empty(), "{o:?}");
    #[cfg(target_os = "linux")]
    {
        use std::{fs, path::PathBuf, time::Instant};
        let stat = PathBuf::from(format!("/proc/{descendant}/stat"));
        let begun = Instant::now();
        loop {
            match fs::read_to_string(&stat) {
                Err(_) => break,
                Ok(s) if s.rsplit(") ").next().is_some_and(|t| t.starts_with('Z')) => break,
                Ok(_) => {
                    assert!(
                        begun.elapsed() < child_deadline::wall(Duration::from_secs(5)),
                        "descendant survived"
                    );
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }
    }
    // CPU limit: a busy loop is stopped by SIGXCPU/SIGKILL long before the
    // generous wall deadline.
    let o = bounded(
        sh,
        &["-c".into(), "while :; do :; done".into()],
        &[],
        child_deadline::wall(WALL),
        1,
    );
    assert!(!o.timed_out && o.code.is_none(), "{o:?}");
    assert!(matches!(o.signal, Some(24) | Some(9)), "{o:?}");
    // Report classification is independent of exit codes.
    assert_eq!(
        sanitizer("==1==ERROR: AddressSanitizer: SEGV on unknown address 0x0"),
        Sanitizer::Address("SEGV".into())
    );
    assert_eq!(
        sanitizer("model.cpp:3:5: runtime error: signed integer overflow"),
        Sanitizer::Undefined("signed integer overflow".into())
    );
    assert_eq!(sanitizer("task mismatch epoch 1"), Sanitizer::None);
    // An actual UBSan abort is classified as undefined behaviour with its own
    // exit code, distinct from ASan's.
    let model = build(
        "ubsan-probe",
        "#include <climits>\n#include <cstdlib>\nint main(int argc, char** argv) { int x = std::atoi(argv[1]) + INT_MAX; return x == 7; }\n",
    );
    let o = model.run(1);
    assert_eq!(o.code, Some(UBSAN), "{}", o.summary());
    assert!(
        matches!(&o.sanitizer, Sanitizer::Undefined(m) if m.contains("overflow")),
        "{}",
        o.summary()
    );
}
