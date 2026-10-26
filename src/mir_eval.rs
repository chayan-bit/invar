//! Independent evaluator for verified typed MIR blocks.

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::{
    eval::{
        CleanupEvent, EvalError, Observation, Outcome, TrapCategory, Value, admit_live, make_view,
        view_index, view_len,
    },
    front::{BinaryOp, Mode, Span, Type, UnaryOp},
    mir::{self, Function, Operation, Terminator, ValueId},
};

mod bulk;
mod f32_source;
mod frame;
pub(crate) mod gather;
pub(crate) mod histogram;
mod interpreter;
mod operation;
mod reduce;
pub(crate) mod region;
pub(crate) mod scatter;
mod stencil;
mod summary;
mod tensor;
mod transpose;
mod values;
pub(crate) use bulk::{LiveLoop, LoopOverride};
use bulk::{live_loop, live_trips};
use frame::{Frame, Slot, copied, park, record_cleanup};
pub use reduce::{ReduceCall, Reducer, serial_reduction};
#[cfg(test)]
use values::mir_host_index;
use values::{array_length, binary, filled, indexed, mir_len_u32, store_index, unary};

const MAX_CALL_DEPTH: usize = 64;
const MAX_ARRAY_LEN: usize = crate::array_limits::MAX_ARRAY_LEN;

fn tool(code: &'static str, message: impl Into<String>) -> EvalError {
    EvalError::Tool {
        code,
        message: message.into(),
    }
}

fn trap(category: TrapCategory, span: Span) -> EvalError {
    EvalError::Trap { category, span }
}

type BulkProvider<'a> = dyn FnMut(&LiveLoop<'_>) -> Result<Option<LoopOverride>, EvalError> + 'a;
pub(crate) type SortProvider<'a> =
    dyn FnMut(&str, &str, &[Value], Span, &mut usize) -> Result<Value, EvalError> + 'a;

struct Interpreter<'a> {
    program: &'a mir::Program,
    provider: Option<&'a mut BulkProvider<'a>>,
    sorts: Option<&'a mut SortProvider<'a>>,
    /// Engine hook for reached `sum_tree`/`dot_tree`/`sum_exact`/`dot_exact` calls.
    reducer: Option<&'a mut Reducer<'a>>,
    fuel_limit: u64,
    steps_used: u64,
    call_depth: usize,
    trace: Vec<CleanupEvent>,
    live_elements: usize,
    used_overrides: usize,
    /// One entry per plan the provider offered, in offer order: whether MIR
    /// published it. A plan whose publication ended the run is recorded too.
    published: Vec<bool>,
}

/// Run verified MIR with the same source step convention as the M0 evaluator.
pub fn run(program: &mir::Program, fuel_limit: u64) -> Result<Outcome, EvalError> {
    let observation = observe(program, fuel_limit);
    observation.result.map(|value| Outcome {
        value,
        steps_used: observation.steps_used,
        cleanup_trace: observation.cleanup_trace,
    })
}

/// Each MIR call recurses on the host stack, so the root runs on the reference
/// evaluator's sized stack and the 64-call `CALL_DEPTH_LIMIT` is reached first.
pub fn observe(program: &mir::Program, fuel_limit: u64) -> Observation {
    crate::eval::stack::observe_on_own_stack(|| observe_with_scheduler(program, fuel_limit, None).0)
}

/// Like [`observe_offers`], but reports only how many offered plans were published.
pub(crate) fn observe_with_scheduler<'a>(
    program: &'a mir::Program,
    fuel_limit: u64,
    provider: Option<&'a mut BulkProvider<'a>>,
) -> (Observation, usize) {
    let (observation, published) = observe_offers(program, fuel_limit, provider);
    let count = published.iter().filter(|published| **published).count();
    (observation, count)
}

/// Run with a provider; the second value says, for each plan the provider
/// offered in offer order, whether MIR published it. A refused offer falls
/// back to the serial loop and the run continues, so a caller attributing
/// offered plans must use these outcomes, not a published count.
pub(crate) fn observe_offers<'a>(
    program: &'a mir::Program,
    fuel_limit: u64,
    provider: Option<&'a mut BulkProvider<'a>>,
) -> (Observation, Vec<bool>) {
    observe_all(program, fuel_limit, provider, None, None)
}

/// [`observe_offers`] with an engine reducer for reached float reductions.
/// The reducer returns `None` to keep the serial MIR result for one call.
pub(crate) fn observe_engine<'a>(
    program: &'a mir::Program,
    fuel_limit: u64,
    provider: Option<&'a mut BulkProvider<'a>>,
    reducer: Option<&'a mut Reducer<'a>>,
) -> (Observation, Vec<bool>) {
    observe_all(program, fuel_limit, provider, None, reducer)
}

/// Run with every hook: bulk plans, explicit sorts (sort and spatial runs,
/// which also pass the engine reducer for their float reductions).
pub(crate) fn observe_all<'a>(
    program: &'a mir::Program,
    fuel_limit: u64,
    provider: Option<&'a mut BulkProvider<'a>>,
    sorts: Option<&'a mut SortProvider<'a>>,
    reducer: Option<&'a mut Reducer<'a>>,
) -> (Observation, Vec<bool>) {
    if fuel_limit == 0 {
        return (
            Observation {
                result: Err(tool("INVALID_FUEL", "fuel must be positive")),
                steps_used: 0,
                cleanup_trace: Vec::new(),
            },
            Vec::new(),
        );
    }
    if let Err(error) = mir::verify(program) {
        return (
            Observation {
                result: Err(tool(error.code, error.message)),
                steps_used: 0,
                cleanup_trace: Vec::new(),
            },
            Vec::new(),
        );
    }
    let mut interpreter = Interpreter {
        program,
        provider,
        sorts,
        reducer,
        fuel_limit,
        steps_used: 0,
        call_depth: 0,
        trace: Vec::new(),
        live_elements: 0,
        used_overrides: 0,
        published: Vec::new(),
    };
    let result = interpreter.call("main", Vec::new());
    (
        Observation {
            result,
            steps_used: interpreter.steps_used,
            cleanup_trace: interpreter.trace,
        },
        interpreter.published,
    )
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;

    #[test]
    fn issue_ai_01_machine_diagnostics_mir_size_index_faults() {
        let mut cases = vec![(
            mir_host_index(65_536, u16::MAX as u64).unwrap_err(),
            "MIR_INDEX",
        )];
        #[cfg(target_pointer_width = "64")]
        cases.push((mir_len_u32(u32::MAX as usize + 1).unwrap_err(), "MIR_SIZE"));
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/invar-diagnostics-v3.schema.json"))
                .unwrap();
        for (error, code) in cases {
            let document = error.to_diagnostics_json("fixture.inv", b"");
            assert_eq!(document["diagnostics"][0]["code"], code);
            assert!(jsonschema::is_valid(&schema, &document), "{document}");
        }
    }

    #[test]
    fn issue_ai_01_machine_diagnostics_mir_fault_boundaries() {
        let source = b"fn main() -> i32 { return 1; }";
        let checked = crate::check::check(crate::front::parse(source).unwrap(), true).unwrap();
        let mut program = mir::lower(&checked).unwrap();
        let mut frame = Frame::default();
        let mut errors = vec![
            frame.value(0).unwrap_err(),
            frame.slot("missing").unwrap_err(),
        ];
        frame.slots.insert(
            "scalar".into(),
            Slot {
                value: Some(Value::I32(1)),
                owned: true,
            },
        );
        errors.push(frame.array_ref("scalar").unwrap_err());

        let array = Value::ArrayI32(Rc::new(RefCell::new(vec![1])));
        let Value::ArrayI32(cell) = &array else {
            unreachable!();
        };
        let loan = cell.borrow_mut();
        errors.push(indexed(&array, 0, Span { start: 0, end: 0 }).unwrap_err());
        drop(loan);

        program.functions[0].entry = usize::MAX;
        let mut interpreter = Interpreter {
            program: &program,
            provider: None,
            sorts: None,
            reducer: None,
            fuel_limit: 100,
            steps_used: 0,
            call_depth: 0,
            trace: Vec::new(),
            live_elements: 0,
            used_overrides: 0,
            published: Vec::new(),
        };
        errors.push(interpreter.call("missing", Vec::new()).unwrap_err());
        errors.push(interpreter.call("main", Vec::new()).unwrap_err());
        frame.values.insert(0, Value::U32(u32::MAX));
        errors.push(
            interpreter
                .operation(
                    "main",
                    &Operation::ForNext(0),
                    Some(&Type::U32),
                    Span { start: 0, end: 0 },
                    &mut frame,
                )
                .unwrap_err(),
        );

        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/invar-diagnostics-v3.schema.json"))
                .unwrap();
        for (error, expected) in errors.into_iter().zip([
            "MIR_VALUE",
            "MIR_PLACE",
            "MIR_TYPE",
            "MIR_LOAN",
            "MIR_CALL",
            "MIR_BLOCK",
            "MIR_FOR",
        ]) {
            let document = error.to_diagnostics_json("fixture.inv", source);
            assert_eq!(document["diagnostics"][0]["code"], expected);
            assert!(jsonschema::is_valid(&schema, &document), "{document}");
        }
    }

    #[test]
    fn issue_eval_call_depth_limit_is_typed_in_every_mir_entry_on_a_default_thread() {
        for n in [10u32, 500] {
            let source = format!(
                "fn r(n: u32) -> i32 {{ if n == 0u32 {{ return 1; }} return r(n - 1u32) + 1; }} fn main() -> i32 {{ return r({n}u32); }}"
            );
            let checked =
                crate::check::check(crate::front::parse(source.as_bytes()).unwrap(), true).unwrap();
            let program = mir::lower(&checked).unwrap();
            let expected = format!("{:?}", observe(&program, 1_000_000));
            assert_eq!(
                expected.contains("CALL_DEPTH_LIMIT"),
                n == 500,
                "{expected}"
            );
            // Debug strings cross the thread boundary: an `Observation` holds `Rc` values.
            let entries = std::thread::scope(|scope| {
                std::thread::Builder::new()
                    .stack_size(2 * 1024 * 1024)
                    .spawn_scoped(scope, || {
                        [
                            observe_with_scheduler(&program, 1_000_000, None).0,
                            observe_offers(&program, 1_000_000, None).0,
                            observe_engine(&program, 1_000_000, None, None).0,
                            observe_all(&program, 1_000_000, None, None, None).0,
                        ]
                        .map(|observation| format!("{observation:?}"))
                    })
                    .unwrap()
                    .join()
                    .expect("every MIR entry must not overflow a 2 MiB caller thread")
            });
            assert_eq!(
                entries,
                [
                    expected.clone(),
                    expected.clone(),
                    expected.clone(),
                    expected
                ]
            );
        }
    }
}
