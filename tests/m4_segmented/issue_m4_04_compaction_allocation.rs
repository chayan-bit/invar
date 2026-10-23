#![forbid(unsafe_code)]
#[path = "../support/language_limits.rs"]
mod language_limits;
use invar::{
    check,
    compaction_source::{
        Budget, Contract, Error,
        allocation::{self, BudgetStatus, Refusal},
    },
    eval, front, general_net, mir, mir_eval,
};

fn budget() -> Budget {
    Budget {
        max_elements: 1024,
        max_bytes: 20_480,
        max_work: 7168,
    }
}
fn function(op: &str, grouped: bool, renamed: bool) -> String {
    let mut s = if grouped {
        format!("fn run(xs: read Array<u32>, count:u32, cutoff:u32)->Array<u32> {{
          var selected=(0u32);
          for i in (0u32)..(count) {{ if (((xs)[(i)]) {op} (cutoff)) {{ selected=((selected)+(1u32)); }} }}
          var out=(filled_u32((selected),(0u32))); var pos=(0u32);
          for j in (0u32)..(count) {{ if (((xs)[(j)]) {op} (cutoff)) {{ out[(pos)]=((xs)[(j)]);pos=((pos)+(1u32)); }} }}
          return move out;
        }}")
    } else {
        format!(
            "fn run(xs: read Array<u32>, count:u32, cutoff:u32)->Array<u32> {{
          var selected=0u32;
          for i in 0u32..count {{ if xs[i] {op} cutoff {{ selected=selected+1u32; }} }}
          var out=filled_u32(selected,0u32); var pos=0u32;
          for j in 0u32..count {{ if xs[j] {op} cutoff {{ out[pos]=xs[j];pos=pos+1u32; }} }}
          return move out;
        }}"
        )
    };
    if renamed {
        for (a, b) in [
            ("run", "select_values"),
            ("xs", "samples"),
            ("count", "extent"),
            ("cutoff", "threshold"),
            ("selected", "retained"),
            ("out", "result"),
            ("pos", "position"),
            ("for i", "for scan_index"),
            ("for j", "for copy_index"),
            ("[i]", "[scan_index]"),
            ("[j]", "[copy_index]"),
            ("(i)", "(scan_index)"),
            ("(j)", "(copy_index)"),
        ] {
            s = s.replace(a, b);
        }
    }
    s
}
fn caller(f: &str, input: &[u32]) -> String {
    let entry = if f.contains("select_values(") {
        "select_values"
    } else {
        "run"
    };
    let values = input
        .iter()
        .map(|v| format!("{v}u32"))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{f} fn main()->i32 {{let input:Array<u32> = [{values}];let answer={entry}(input,{}u32,5u32);return 7;}}",
        input.len()
    )
}
fn checked(s: &str) -> check::CheckedProgram {
    check::check(front::parse(s.as_bytes()).unwrap(), true).unwrap()
}
fn equal(a: &eval::Observation, b: &eval::Observation) {
    assert_eq!(a.result, b.result);
    assert_eq!(a.steps_used, b.steps_used);
    assert_eq!(a.cleanup_trace, b.cleanup_trace);
}
fn observe_fuels(s: &str, fuels: impl IntoIterator<Item = u64>) -> eval::Observation {
    let p = checked(s);
    let net = general_net::Net::lower(&p, general_net::Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    let m = net.compiled_mir(&proof).unwrap();
    let full = eval::observe(&p, 100_000);
    for fuel in fuels {
        let a = eval::observe(&p, fuel);
        equal(&a, &mir_eval::observe(&m, fuel));
        let got = general_net::oracle::observe(
            &net,
            &proof,
            general_net::oracle::Limits {
                fuel,
                array_elements: invar::array_limits::MAX_LIVE_ELEMENTS,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(got.failure, None);
        equal(&a, &got.semantic);
    }
    full
}
fn scalar(op: &str, input: &[u32]) -> Vec<u32> {
    input
        .iter()
        .copied()
        .filter(|v| match op {
            "<" => *v < 5,
            "<=" => *v <= 5,
            "==" => *v == 5,
            "!=" => *v != 5,
            ">=" => *v >= 5,
            ">" => *v > 5,
            _ => unreachable!(),
        })
        .collect()
}
fn inner(e: &front::Expr) -> &front::Expr {
    match &e.kind {
        front::ExprKind::Group(e) => inner(e),
        _ => e,
    }
}

#[test]
fn issue_m4_04_segmented_flattening_compaction_library_structure_fuel_and_allocation() {
    let mut fixtures = 0;
    let mut prefixes = 0;
    for op in ["<", "<=", "==", "!=", ">=", ">"] {
        for grouped in [false, true] {
            for renamed in [false, true] {
                for input in [&[][..], &[5][..], &[1, 7, 2, 9][..]] {
                    let f = function(op, grouped, renamed);
                    let s = caller(&f, input);
                    let entry = if renamed { "select_values" } else { "run" };
                    let c = Contract::from_source(s.as_bytes(), entry).unwrap();
                    c.verify_source(s.as_bytes()).unwrap();
                    c.verify_mir(&c.verified_mir().unwrap()).unwrap();
                    let p = c.prepare(input, input.len() as u32, 5, budget()).unwrap();
                    let expected = scalar(op, input);
                    assert_eq!(p.execute().unwrap(), expected);
                    let n = input.len() as u64;
                    let k = expected.len() as u64;
                    // Hand counts are independent of the module's checked-AST sums.
                    assert_eq!(p.remaining_region_ticks(), Some(14 * n + 13 * k + 9));
                    let allocation = p.allocation_description().unwrap();
                    let d = allocation.descriptor();
                    assert_eq!(d.ticks_to_intrinsic, 7 * n + 4 * k + 4);
                    assert_eq!(d.requested_elements, k as u32);
                    assert_eq!(d.statement_index, 2);
                    assert_eq!(d.source_hash, c.source_hash());
                    assert_eq!(d.mir_hash, c.mir_contract());
                    assert_eq!(d.function, entry);
                    assert_eq!(d.output_binding, c.names()[3]);
                    assert_eq!(
                        d.length_binding,
                        if renamed { "retained" } else { "selected" }
                    );
                    assert_eq!(d.loop_spans, c.loop_spans());
                    let ast = checked(&s);
                    let f = ast
                        .program
                        .functions
                        .iter()
                        .find(|f| f.name == entry)
                        .unwrap();
                    let statement = &f.body.statements[2];
                    assert_eq!(d.statement_span, statement.span);
                    let front::StmtKind::Binding { value, .. } = &statement.kind else {
                        panic!()
                    };
                    let call = inner(value);
                    assert_eq!(d.call_span, call.span);
                    let front::ExprKind::Call { args, .. } = &call.kind else {
                        panic!()
                    };
                    assert_eq!(d.argument_spans, [args[0].span, args[1].span]);
                    assert!(
                        d.loop_spans[0].end < d.statement_span.start
                            && d.statement_span.end < d.loop_spans[1].start
                    );
                    assert_eq!(allocation.required_live_elements(), k as usize);
                    assert_eq!(
                        allocation.maximum_live_before_allocation(),
                        Some(invar::array_limits::MAX_LIVE_ELEMENTS - k as usize)
                    );
                    assert_eq!(
                        allocation.diagnose_reported_live(None),
                        BudgetStatus::Refused(Refusal::MissingLiveState)
                    );
                    assert_eq!(
                        allocation.diagnose_reported_live(Some(0)),
                        BudgetStatus::RequiresRuntimeCheck
                    );
                    assert_eq!(
                        allocation.diagnose_reported_live(Some(
                            invar::array_limits::MAX_LIVE_ELEMENTS - k as usize
                        )),
                        BudgetStatus::RequiresRuntimeCheck
                    );
                    assert_eq!(
                        allocation.diagnose_reported_live(Some(
                            (invar::array_limits::MAX_LIVE_ELEMENTS + 1) - k as usize
                        )),
                        BudgetStatus::Refused(Refusal::LiveElements)
                    );
                    p.verify_allocation_descriptor(d).unwrap();
                    let full = eval::observe(&ast, 100_000);
                    assert_eq!(full.result, Ok(eval::Value::I32(7)));
                    assert_eq!(full.steps_used, 25 + n + 14 * n + 13 * k);
                    let got = observe_fuels(&s, 1..=full.steps_used + 1);
                    equal(&got, &full);
                    prefixes += full.steps_used + 1;
                    fixtures += 1;
                }
            }
        }
    }
    assert_eq!(fixtures, 72);
    println!("grouped/renamed library fixtures: {fixtures}; normal prefixes: {prefixes}");
}

#[test]
fn issue_m4_04_segmented_flattening_compaction_library_allocation_mutations() {
    let f = function(">=", true, true);
    let s = caller(&f, &[1, 7, 2, 9]);
    let c = Contract::from_source(s.as_bytes(), "select_values").unwrap();
    let p = c.prepare(&[1, 7, 2, 9], 4, 5, budget()).unwrap();
    let d = p.allocation_description().unwrap().descriptor();
    let mut variants = Vec::new();
    let mut v = d;
    v.source_hash[0] ^= 1;
    variants.push(v);
    let mut v = d;
    v.mir_hash[0] ^= 1;
    variants.push(v);
    let mut v = d;
    v.function = "other";
    variants.push(v);
    let mut v = d;
    v.statement_index = 3;
    variants.push(v);
    let mut v = d;
    v.statement_span.end += 1;
    variants.push(v);
    let mut v = d;
    v.call_span.start += 1;
    variants.push(v);
    let mut v = d;
    v.argument_spans.reverse();
    variants.push(v);
    let mut v = d;
    v.output_binding = "other";
    variants.push(v);
    let mut v = d;
    v.length_binding = "extent";
    variants.push(v);
    let mut v = d;
    v.loop_spans.reverse();
    variants.push(v);
    let mut v = d;
    v.requested_elements = invar::array_limits::MAX_ARRAY_LEN as u32 + 1;
    variants.push(v);
    let mut v = d;
    v.ticks_to_intrinsic += 1;
    variants.push(v);
    let mut v = d;
    v.per_array_limit = usize::MAX;
    variants.push(v);
    let mut v = d;
    v.live_limit = usize::MAX;
    variants.push(v);
    for v in variants {
        assert_eq!(
            p.verify_allocation_descriptor(v),
            Err(Error::AllocationDescriptorMismatch)
        );
    }
    let mut m = c.verified_mir().unwrap();
    let call = m
        .functions
        .iter_mut()
        .find(|f| f.name == "select_values")
        .unwrap()
        .blocks
        .iter_mut()
        .flat_map(|b| &mut b.instructions)
        .find(|i| matches!(&i.kind,mir::Operation::Call{name,..} if name=="filled_u32"))
        .unwrap();
    assert_eq!(call.span, d.call_span);
    call.span.end += 1;
    mir::verify(&m).unwrap();
    assert_eq!(c.verify_mir(&m), Err(Error::MirMismatch));
    let other = s.replace("(retained),(0u32)", "(retained),(1u32)");
    assert_ne!(s, other);
    assert_eq!(
        c.verify_source(other.as_bytes()),
        Err(Error::SourceMismatch)
    );
    assert_eq!(
        Contract::from_source(other.as_bytes(), "select_values").unwrap_err(),
        Error::Unrecognized
    );
}

fn filled_caller(
    f: &str,
    input_len: u32,
    other_len: u32,
    cutoff: u32,
    count: u32,
    padding: usize,
) -> String {
    let pad = language_limits::padding(padding);
    format!(
        "{f} fn main()->i32 {{{pad}let input=filled_u32({input_len}u32,7u32);let keep=filled_u32({other_len}u32,0u32);let answer=run(input,{count}u32,{cutoff}u32);return 7;}}"
    )
}
fn tool(out: &eval::Observation, message: &str, ticks: u64) {
    assert_eq!(
        out.result,
        Err(eval::EvalError::Tool {
            code: "ARRAY_SIZE_LIMIT",
            message: message.to_owned()
        })
    );
    assert_eq!(out.steps_used, ticks);
    assert!(out.cleanup_trace.is_empty());
}

#[test]
fn issue_m4_04_segmented_flattening_compaction_library_caller_live_owners_refuse() {
    let f = function(">=", false, false);
    let input = vec![7u32; 2];
    let padding = invar::array_limits::MAX_LIVE_ELEMENTS - 4;
    let ticks = 44 + 4 * padding.div_ceil(invar::array_limits::MAX_ARRAY_LEN) as u64;
    for (other_len, cutoff, refuses) in [(0, 5, false), (1, 5, true), (2, 5, true), (2, 8, false)] {
        let s = filled_caller(&f, 2, other_len, cutoff, 2, padding);
        let c = Contract::from_source(s.as_bytes(), "run").unwrap();
        let p = c.prepare(&input, 2, cutoff, budget()).unwrap();
        let d = p.allocation_description().unwrap();
        let live = padding + 2 + other_len as usize;
        let status = d.diagnose_reported_live(Some(live));
        if refuses {
            assert_eq!(status, BudgetStatus::Refused(Refusal::LiveElements));
        } else {
            assert_eq!(status, BudgetStatus::RequiresRuntimeCheck);
        }
        // An omitted caller ledger or a forged small number grants no permit.
        assert_eq!(
            d.diagnose_reported_live(None),
            BudgetStatus::Refused(Refusal::MissingLiveState)
        );
        assert_eq!(
            d.diagnose_reported_live(Some(input.len())),
            BudgetStatus::RequiresRuntimeCheck
        );
        // The local Vec result has no source publication route and cannot waive
        // the caller owner's source accounting. Run the intact source below.
        assert_eq!(
            p.execute().unwrap(),
            if cutoff == 5 { vec![7, 7] } else { vec![] }
        );
        assert_eq!(
            d.descriptor().ticks_to_intrinsic,
            if cutoff == 5 { 26 } else { 18 }
        );
        let result = observe_fuels(
            &s,
            if refuses {
                vec![ticks - 1, ticks, ticks + 1]
            } else {
                vec![100_000]
            },
        );
        if refuses {
            tool(&result, "live array element limit exceeded", ticks);
        } else {
            assert_eq!(result.result, Ok(eval::Value::I32(7)));
        }
    }
}

#[test]
fn issue_m4_04_segmented_flattening_compaction_library_per_array_guard_order() {
    assert_eq!(
        allocation::diagnose_reported_budget(invar::array_limits::MAX_ARRAY_LEN as u32 + 1, None),
        BudgetStatus::Refused(Refusal::PerArray)
    );
    assert_eq!(
        allocation::diagnose_reported_budget(
            invar::array_limits::MAX_ARRAY_LEN as u32 + 1,
            Some(invar::array_limits::MAX_LIVE_ELEMENTS)
        ),
        BudgetStatus::Refused(Refusal::PerArray)
    );
    assert_eq!(
        allocation::diagnose_reported_budget(2, Some(usize::MAX)),
        BudgetStatus::Refused(Refusal::ArithmeticOverflow)
    );
    assert_eq!(
        allocation::diagnose_reported_budget(0, Some(invar::array_limits::MAX_LIVE_ELEMENTS + 1)),
        BudgetStatus::Refused(Refusal::LiveElements)
    );
    let f = function(">=", false, false);
    let s = filled_caller(
        &f,
        invar::array_limits::MAX_ARRAY_LEN as u32 + 1,
        1_000_000,
        5,
        2,
        0,
    );
    let c = Contract::from_source(s.as_bytes(), "run").unwrap();
    assert_eq!(
        c.prepare(
            &vec![7u32; invar::array_limits::MAX_ARRAY_LEN + 1],
            2,
            5,
            budget()
        )
        .unwrap_err(),
        Error::Budget
    );
    let result = observe_fuels(&s, [3, 4, 5]);
    tool(&result, "array length limit exceeded", 4);
    // The recognized family proves K<=count<=input length<=2^24. A source change
    // that enlarges the intervening allocation is a miss, never waived by K=0.
    let changed = f.replace(
        "filled_u32(selected,0u32)",
        "filled_u32(selected+16777217u32,0u32)",
    );
    assert_eq!(
        Contract::from_source(changed.as_bytes(), "run").unwrap_err(),
        Error::Unrecognized
    );
    let s = filled_caller(&changed, 0, 1_000_000, 5, 0, 0);
    let result = observe_fuels(&s, [23, 24, 25]);
    tool(&result, "array length limit exceeded", 24);
}
