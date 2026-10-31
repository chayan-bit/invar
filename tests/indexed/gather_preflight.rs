//! Issue #5 (M4-05).
#![forbid(unsafe_code)]
use crate::language_limits;

// Test-local proposal checks, not runtime admission or publication authority.
// Keep the reviewed source module and shared exports/hooks unchanged.
use front::{Expr, ExprKind, Place, Span, StmtKind};
use indexed_source::{Access, Contract, Family};
use invar::indexed_source;
use invar::{check, eval, front, general_net, mir, mir_eval};
use sha2::{Digest, Sha256};

const GATHER: &str = "fn gather(a: read Array<u32>, ix: read Array<u32>) -> Array<u32> { var out = filled_u32(len(ix),0u32); for i in 0u32..len(ix) { out[i] = a[ix[i]]; } return out; }";
const BYTE_LIMIT: usize = 160 * 1024;
const VISIT_LIMIT: usize = 65_536;
const METADATA_LIMIT: usize = 4096;
/// Array size the maximum-arrays test exercises; no longer a refusal bound (#57 byte budgets).
const MAX_ARRAY: usize = 4096;

#[derive(Debug, PartialEq, Eq)]
enum Refusal {
    Source,
    Binding,
    Shape,
    Recipe,
    OutputShape,
    Index { iteration: usize },
    Metadata,
    Bytes,
    Visits,
    Arithmetic,
    Fuel,
}

#[derive(Debug, PartialEq, Eq)]
struct StaticRecipe {
    entry: String,
    source_contract: [u8; 32],
    loop_span: Span,
    ticks: Vec<Span>,
}

fn checked(source: &str) -> check::CheckedProgram {
    check::check(front::parse(source.as_bytes()).unwrap(), false).unwrap()
}

// Independent AST accounting follows source demand order. Groups are not
// semantic steps. This narrow walker rejects operations outside the leaf.
fn expression_ticks(expr: &Expr, ticks: &mut Vec<Span>) -> Result<(), Refusal> {
    if let ExprKind::Group(inner) = &expr.kind {
        return expression_ticks(inner, ticks);
    }
    ticks.push(expr.span);
    match &expr.kind {
        ExprKind::Name(_) => Ok(()),
        ExprKind::Index { base, index } => {
            expression_ticks(base, ticks)?;
            expression_ticks(index, ticks)
        }
        _ => Err(Refusal::Recipe),
    }
}

fn recipe(source: &str, entry: &str) -> Result<(StaticRecipe, Contract, mir::Program), Refusal> {
    let checked = check::check(
        front::parse(source.as_bytes()).map_err(|_| Refusal::Source)?,
        false,
    )
    .map_err(|_| Refusal::Source)?;
    let contract =
        Contract::from_checked(source.as_bytes(), &checked, entry).map_err(|_| Refusal::Shape)?;
    if contract.family() != Family::Gather {
        return Err(Refusal::Shape);
    }
    let source_mir = mir::lower(&checked).map_err(|_| Refusal::Binding)?;
    let net = general_net::Net::lower(&checked, general_net::Limits::default())
        .map_err(|_| Refusal::Binding)?;
    let proof = net.verify().map_err(|_| Refusal::Binding)?;
    let extracted = net.compiled_mir(&proof).map_err(|_| Refusal::Binding)?;
    let compiled = contract.verified_mir().map_err(|_| Refusal::Binding)?;
    if source_mir.contract_digest() != extracted.contract_digest()
        || source_mir.contract_digest() != compiled.contract_digest()
        || source_mir.contract_digest() != contract.source_contract()
    {
        return Err(Refusal::Binding);
    }
    let function = checked.program.function(entry).ok_or(Refusal::Shape)?;
    let region = function.body.statements.get(1).ok_or(Refusal::Shape)?;
    let StmtKind::For { body, .. } = &region.kind else {
        return Err(Refusal::Shape);
    };
    let [store] = body.statements.as_slice() else {
        return Err(Refusal::Shape);
    };
    let StmtKind::Assign {
        place: Place::Index { index, .. },
        value,
    } = &store.kind
    else {
        return Err(Refusal::Shape);
    };
    let mut ticks = vec![region.span, store.span]; // control, assignment
    expression_ticks(value, &mut ticks)?; // RHS precedes destination index
    expression_ticks(index, &mut ticks)?;
    let result = StaticRecipe {
        entry: entry.into(),
        source_contract: compiled.contract_digest(),
        loop_span: contract.loop_span(),
        ticks,
    };
    validate_recipe(&result, &compiled)?;
    Ok((result, contract, compiled))
}

fn validate_recipe(recipe: &StaticRecipe, current: &mir::Program) -> Result<(), Refusal> {
    mir::verify(current).map_err(|_| Refusal::Binding)?;
    if current.contract_digest() != recipe.source_contract {
        return Err(Refusal::Binding);
    }
    let function = current.function(&recipe.entry).ok_or(Refusal::Recipe)?;
    let (head_id, head, body_id) = function
        .blocks
        .iter()
        .enumerate()
        .find_map(|(id, b)| match &b.end {
            Some(mir::Terminator::Branch { yes, span, .. }) if *span == recipe.loop_span => {
                Some((id, b, *yes))
            }
            _ => None,
        })
        .ok_or(Refusal::Recipe)?;
    let body = function.blocks.get(body_id).ok_or(Refusal::Recipe)?;
    if !matches!(body.end, Some(mir::Terminator::Goto { target, .. }) if target == head_id)
        || head
            .instructions
            .iter()
            .any(|i| matches!(i.kind, mir::Operation::Tick))
    {
        return Err(Refusal::Recipe);
    }
    let actual: Vec<_> = body
        .instructions
        .iter()
        .filter(|i| matches!(i.kind, mir::Operation::Tick))
        .map(|i| i.span)
        .collect();
    if actual != recipe.ticks {
        return Err(Refusal::Recipe);
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct Cost {
    bytes: usize,
    visits: usize,
}

// Logical retained payload and element visits only. Allocator capacity,
// allocator overhead, RSS, source/AST/net/MIR storage and compiler work are
// excluded. No fabricated measurements or memory-safety permit results.
fn raw_cost(a: usize, ix: usize, n: usize, metadata: usize) -> Result<Cost, Refusal> {
    let inputs = a.checked_add(ix).ok_or(Refusal::Arithmetic)?;
    let words = inputs.checked_add(n).ok_or(Refusal::Arithmetic)?;
    let bytes = words
        .checked_mul(size_of::<u32>())
        .and_then(|v| {
            size_of::<Access>()
                .checked_mul(n)
                .and_then(|w| v.checked_add(w))
        })
        .and_then(|v| v.checked_add(metadata))
        .ok_or(Refusal::Arithmetic)?;
    let visits = inputs
        .checked_mul(2)
        .and_then(|v| n.checked_mul(6).and_then(|w| v.checked_add(w)))
        .ok_or(Refusal::Arithmetic)?;
    Ok(Cost { bytes, visits })
}

fn bounded_cost(
    a: usize,
    ix: usize,
    n: usize,
    metadata: usize,
    bytes_limit: usize,
    visits_limit: usize,
) -> Result<Cost, Refusal> {
    if n != ix {
        return Err(Refusal::OutputShape);
    }
    if metadata > METADATA_LIMIT {
        return Err(Refusal::Metadata);
    }
    let cost = raw_cost(a, ix, n, metadata)?;
    if cost.bytes > bytes_limit {
        return Err(Refusal::Bytes);
    }
    if cost.visits > visits_limit {
        return Err(Refusal::Visits);
    }
    Ok(cost)
}

fn remaining_ticks(n: u64, ticks_per_iteration: u64) -> Result<u64, Refusal> {
    n.checked_mul(ticks_per_iteration)
        .ok_or(Refusal::Arithmetic)
}

fn static_guard(
    recipe: &StaticRecipe,
    values: &[u32],
    indices: &[u32],
    output_len: usize,
    fuel: u64,
) -> Result<Cost, Refusal> {
    let cost = bounded_cost(
        values.len(),
        indices.len(),
        output_len,
        METADATA_LIMIT,
        BYTE_LIMIT,
        VISIT_LIMIT,
    )?;
    for (iteration, &index) in indices.iter().enumerate() {
        if !usize::try_from(index).is_ok_and(|i| i < values.len()) {
            return Err(Refusal::Index { iteration });
        }
    }
    let ticks = remaining_ticks(
        u64::try_from(indices.len()).map_err(|_| Refusal::Arithmetic)?,
        u64::try_from(recipe.ticks.len()).map_err(|_| Refusal::Arithmetic)?,
    )?;
    if ticks > fuel {
        return Err(Refusal::Fuel);
    }
    Ok(cost)
}

fn array(values: &[u32]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|n| format!("{n}u32"))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn sha256(source: &[u8]) -> String {
    Sha256::digest(source)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn source(leaf: &str, values: &[u32], indices: &[u32], expected: &[u32], alias: bool) -> String {
    let inputs = if alias {
        format!(
            "let a: Array<u32> = {}; let result = gather(a,a);",
            array(values)
        )
    } else {
        format!(
            "let a: Array<u32> = {}; let ix: Array<u32> = {}; let result = gather(a,ix);",
            array(values),
            array(indices)
        )
    };
    format!(
        "{leaf} fn main() -> i32 {{ {inputs} let wanted: Array<u32> = {}; if len(result) != len(wanted) {{ return 0; }} for k in 0u32..len(result) {{ if result[k] != wanted[k] {{ return 0; }} }} return 7; }}",
        array(expected)
    )
}

fn observe(source: &str, every_prefix: bool) -> eval::Observation {
    let checked = checked(source);
    let (_, _, compiled) = recipe(source, "gather").unwrap();
    let net = general_net::Net::lower(&checked, general_net::Limits::default()).unwrap();
    let proof = net.verify().unwrap();
    let complete = eval::observe(&checked, 1_000_000);
    let fuels: Vec<_> = if every_prefix {
        (0..=complete.steps_used + 1).collect()
    } else {
        vec![complete.steps_used, complete.steps_used + 1]
    };
    for fuel in &fuels {
        let reference = eval::observe(&checked, *fuel);
        let serial = mir_eval::observe(&compiled, *fuel);
        let interaction = general_net::oracle::observe(
            &net,
            &proof,
            general_net::oracle::Limits {
                fuel: *fuel,
                array_elements: invar::array_limits::MAX_LIVE_ELEMENTS,
                ..general_net::oracle::Limits::default()
            },
        )
        .unwrap();
        for actual in [&serial, &interaction.semantic] {
            assert_eq!(actual.result, reference.result, "fuel={fuel}");
            assert_eq!(actual.steps_used, reference.steps_used, "fuel={fuel}");
            assert_eq!(actual.cleanup_trace, reference.cleanup_trace, "fuel={fuel}");
        }
    }
    println!(
        "preflight source/MIR/net observations per engine: {}; full steps: {}",
        fuels.len(),
        complete.steps_used
    );
    complete
}

#[test]
fn gather_scatter_frontier_preflight_source_and_recipe_mutations() {
    let (plan, contract, compiled) = recipe(GATHER, "gather").unwrap();
    assert_eq!(plan.ticks.len(), 8); // independently derived AST vector, not an input
    assert_eq!(plan.source_contract, contract.source_contract());
    assert_eq!(remaining_ticks(4096, plan.ticks.len() as u64), Ok(32_768));
    assert_eq!(remaining_ticks(u64::MAX, 8), Err(Refusal::Arithmetic));
    let snippets: Vec<_> = plan.ticks.iter().map(|s| &GATHER[s.start..s.end]).collect();
    assert_eq!(&snippets[2..], &["a[ix[i]]", "a", "ix[i]", "ix", "i", "i"]);
    let mut stale = checked(GATHER);
    stale.expression_types.clear();
    assert!(matches!(
        Contract::from_checked(GATHER.as_bytes(), &stale, "gather"),
        Err(indexed_source::Error::CheckedMismatch)
    ));
    let changed = GATHER.replace("0u32);", "1u32);");
    assert!(recipe(&changed, "gather").is_err());
    for changed in [
        GATHER.replace("out[i] = a[ix[i]]", "out[i] = a[i]"),
        GATHER.replace("0u32..len(ix)", "1u32..len(ix)"),
        GATHER.replace(
            "out[i] = a[ix[i]];",
            "out[i] = a[ix[i]]; out[i] = a[ix[i]];",
        ),
    ] {
        assert!(recipe(&changed, "gather").is_err());
    }
    let renamed = GATHER.replace("gather", "indexed").replace("out", "answer");
    let (_, _, renamed_mir) = recipe(&renamed, "indexed").unwrap();
    assert_eq!(validate_recipe(&plan, &renamed_mir), Err(Refusal::Binding));
    let scatter = GATHER
        .replace("ix: read Array<u32>)", "ix: read Array<u32>, n: u32)")
        .replace("filled_u32(len(ix),0u32)", "filled_u32(n,0u32)")
        .replace("out[i] = a[ix[i]]", "out[ix[i]] = a[i]");
    assert!(matches!(recipe(&scatter, "gather"), Err(Refusal::Shape)));
    for mutation in 0..3 {
        let mut stale = StaticRecipe {
            entry: plan.entry.clone(),
            source_contract: plan.source_contract,
            loop_span: plan.loop_span,
            ticks: plan.ticks.clone(),
        };
        match mutation {
            0 => stale.source_contract[0] ^= 1,
            1 => {
                stale.ticks.pop();
            }
            _ => stale.ticks[2].start += 1,
        }
        assert!(validate_recipe(&stale, &compiled).is_err());
    }
}

#[test]
fn gather_scatter_frontier_preflight_logical_budget_boundaries() {
    let cost = bounded_cost(4096, 4096, 4096, 4096, BYTE_LIMIT, VISIT_LIMIT).unwrap();
    assert_eq!(size_of::<Access>(), 3 * size_of::<usize>());
    if usize::BITS == 64 {
        assert_eq!(cost.bytes, 151_552);
    }
    assert_eq!(cost.visits, 40_960);
    assert_eq!(
        bounded_cost(4096, 4096, 4096, 4096, cost.bytes, cost.visits),
        Ok(cost)
    );
    let cost = raw_cost(4096, 4096, 4096, 4096).unwrap();
    assert_eq!(
        bounded_cost(4096, 4096, 4096, 4096, cost.bytes - 1, VISIT_LIMIT),
        Err(Refusal::Bytes)
    );
    assert_eq!(
        bounded_cost(4096, 4096, 4096, 4096, BYTE_LIMIT, cost.visits - 1),
        Err(Refusal::Visits)
    );
    assert_eq!(
        bounded_cost(0, 0, 0, 4097, BYTE_LIMIT, VISIT_LIMIT),
        Err(Refusal::Metadata)
    );
    assert_eq!(
        bounded_cost(0, 0, 1, 0, BYTE_LIMIT, VISIT_LIMIT),
        Err(Refusal::OutputShape)
    );
    // The old 4096-element cap is gone: sizes past it are bounded only by bytes and visits.
    for (a, ix, n, expected) in [
        (4097, 0, 0, Ok(())),
        (0, 4097, 4097, Ok(())),
        (0, 0, 4097, Err(Refusal::OutputShape)),
        (0, 6000, 6000, Err(Refusal::Bytes)),
        (usize::MAX, 0, 0, Err(Refusal::Arithmetic)),
    ] {
        assert_eq!(
            bounded_cost(a, ix, n, 0, BYTE_LIMIT, VISIT_LIMIT).map(|_| ()),
            expected
        );
    }
    for (a, ix, n, metadata) in [
        (usize::MAX, 1, 0, 0),
        (0, 0, usize::MAX, 0),
        (1, 0, 0, usize::MAX),
        (usize::MAX / 4, 0, 1, 0),
    ] {
        assert_eq!(raw_cost(a, ix, n, metadata), Err(Refusal::Arithmetic));
    }
    assert_eq!(
        bounded_cost(0, 0, 0, 0, 0, 0),
        Ok(Cost {
            bytes: 0,
            visits: 0
        })
    );
}

#[test]
fn gather_scatter_frontier_preflight_inputs_bounds_and_fuel() {
    let (plan, contract, _) = recipe(GATHER, "gather").unwrap();
    for (values, indices, expected) in [
        (&[][..], &[][..], &[][..]),
        (&[41][..], &[0][..], &[41][..]),
        (&[7, 11, 19][..], &[2, 0, 2, 1][..], &[19, 7, 19, 11][..]),
    ] {
        let required = remaining_ticks(indices.len() as u64, plan.ticks.len() as u64).unwrap();
        assert!(static_guard(&plan, values, indices, indices.len(), required).is_ok());
        let candidate = contract.candidate(values, indices, None).unwrap();
        assert_eq!(candidate.output(), expected);
        assert_eq!(candidate.accesses().len(), indices.len());
        if required > 0 {
            assert_eq!(
                static_guard(&plan, values, indices, indices.len(), required - 1),
                Err(Refusal::Fuel)
            );
        }
    }
    let shared = [2, 0, 2];
    assert!(static_guard(&plan, &shared, &shared, 3, 24).is_ok());
    assert_eq!(
        contract.candidate(&shared, &shared, None).unwrap().output(),
        &[2, 2, 2]
    );
    for (values, indices, iteration) in [
        (&[][..], &[0][..], 0),
        (&[7][..], &[1][..], 0),
        (&[7][..], &[0, u32::MAX][..], 1),
    ] {
        assert_eq!(
            static_guard(&plan, values, indices, indices.len(), 1000),
            Err(Refusal::Index { iteration })
        );
        assert_eq!(
            contract.candidate(values, indices, None).unwrap_err(),
            indexed_source::Error::Bounds { iteration }
        );
    }
    assert_eq!(
        static_guard(&plan, &[1], &[0], 0, 100),
        Err(Refusal::OutputShape)
    );
}

#[test]
fn gather_scatter_frontier_preflight_group_move_and_prefixes() {
    let grouped = GATHER
        .replace("0u32..len(ix)", "(0u32)..(len(ix))")
        .replace("out[i] = a[ix[i]]", "out[(i)] = (a[(ix[(i)])])");
    let mut baseline = None;
    for (leaf, delta) in [
        (GATHER.to_owned(), 0),
        (grouped, 0),
        (GATHER.replace("return out;", "return move out;"), 0),
        (GATHER.replace("return out;", "return (move out);"), 0),
    ] {
        let (plan, _, _) = recipe(&leaf, "gather").unwrap();
        assert_eq!(plan.ticks.len(), 8);
        let whole = source(&leaf, &[7, 11, 19], &[2, 0, 2], &[19, 7, 19], false);
        let complete = observe(&whole, true);
        assert_eq!(complete.result, Ok(eval::Value::I32(7)));
        assert!(!complete.cleanup_trace.is_empty());
        let original = *baseline.get_or_insert(complete.steps_used);
        assert_eq!(complete.steps_used, original + delta);
        println!(
            "preflight return receipt: {}",
            serde_json::json!({
                "leaf_source": leaf,
                "leaf_source_sha256": sha256(leaf.as_bytes()),
                "whole_source_sha256": sha256(whole.as_bytes()),
                "leaf_mir_contract": plan.source_contract,
                "whole_mir_contract": mir::lower(&checked(&whole)).unwrap().contract_digest(),
                "loop_ticks": plan.ticks.len(), "n3_loop_fuel": 3 * plan.ticks.len(),
                "full_source_steps": complete.steps_used, "return_delta": delta,
                "cleanup_events": complete.cleanup_trace.len(),
                "scope": "source_mir_net_oracle_only_no_replacement"
            })
        );
    }
    let unsupported = GATHER.replace("return out;", "return (out);");
    assert_eq!(
        check::check(front::parse(unsupported.as_bytes()).unwrap(), false)
            .unwrap_err()
            .code,
        "OWN_MOVE_REQUIRED"
    );
    for (values, indices, expected, alias) in [
        (&[][..], &[][..], &[][..], false),
        (&[41][..], &[0][..], &[41][..], false),
        (&[2, 0, 2][..], &[2, 0, 2][..], &[2, 2, 2][..], true),
    ] {
        assert_eq!(
            observe(&source(GATHER, values, indices, expected, alias), true).result,
            Ok(eval::Value::I32(7))
        );
    }
}

#[test]
fn gather_scatter_frontier_preflight_maximum_arrays() {
    let values: Vec<_> = (0..MAX_ARRAY as u32).map(|i| 17 * i + 3).collect();
    let indices: Vec<_> = (0..MAX_ARRAY as u32).rev().collect();
    let expected: Vec<_> = (0..MAX_ARRAY as u32)
        .map(|position| 3 + 17 * (MAX_ARRAY as u32 - 1 - position))
        .collect();
    let (plan, contract, _) = recipe(GATHER, "gather").unwrap();
    assert!(static_guard(&plan, &values, &indices, MAX_ARRAY, 32_768).is_ok());
    assert_eq!(
        contract
            .candidate(&values, &indices, None)
            .unwrap()
            .output(),
        expected
    );
    assert_eq!(
        observe(&source(GATHER, &values, &indices, &expected, false), false).result,
        Ok(eval::Value::I32(7))
    );
    assert!(static_guard(&plan, &values, &vec![0; 4097], 4097, 1_000_000).is_ok());
    assert_eq!(
        static_guard(&plan, &values, &vec![0; 6000], 6000, 1_000_000),
        Err(Refusal::Bytes)
    );
}

#[test]
fn gather_scatter_frontier_preflight_traps_and_language_limits() {
    for (values, indices) in [(&[7][..], &[1][..]), (&[7][..], &[0, u32::MAX][..])] {
        let whole = source(GATHER, values, indices, &[], false);
        let complete = observe(&whole, true);
        let Err(eval::EvalError::Trap {
            category: eval::TrapCategory::IndexOutOfBounds,
            span,
        }) = complete.result
        else {
            panic!("expected retained source trap")
        };
        assert_eq!(&whole[span.start..span.end], "a[ix[i]]");
        assert!(complete.cleanup_trace.is_empty());
    }
    let bad = GATHER.replace("a[ix[i]]", "a[-1]");
    assert_eq!(
        check::check(front::parse(bad.as_bytes()).unwrap(), false)
            .unwrap_err()
            .code,
        "TYPE_MISMATCH"
    );
    let output_alias = GATHER.replace("filled_u32(len(ix),0u32)", "a");
    assert_eq!(
        check::check(front::parse(output_alias.as_bytes()).unwrap(), false)
            .unwrap_err()
            .code,
        "OWN_MOVE_REQUIRED"
    );
    let pad = language_limits::padding(invar::array_limits::MAX_LIVE_ELEMENTS - 2);
    for (main, every_prefix) in [
        (
            "fn main() -> i32 { let a = filled_u32(16777217u32,7u32); return 7; }".to_owned(),
            true,
        ),
        (
            format!(
                "fn main() -> i32 {{ {pad} let a: Array<u32> = [7u32]; let ix: Array<u32> = [0u32]; let result = gather(a,ix); return 7; }}"
            ),
            false,
        ),
    ] {
        // Large live-boundary fixtures check completion; small fixtures retain every prefix.
        let complete = observe(&format!("{GATHER} {main}"), every_prefix);
        assert!(matches!(
            complete.result,
            Err(eval::EvalError::Tool {
                code: "ARRAY_SIZE_LIMIT",
                ..
            })
        ));
        assert!(complete.cleanup_trace.is_empty());
    }
}
