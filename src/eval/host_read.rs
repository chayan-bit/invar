//! Reference semantics for host array arguments (HA-1.1,
//! `docs/host-array-arguments-ha1.md`).
//!
//! Two entry points call one checked function:
//! - `observe_host_read_call`: exact scalars and `read Array<i32|u32|f32|f64|Bool>`
//!   buffers only; any `mut` or `take` parameter or `Result` result is refused.
//! - `observe_host_call` (stage 1): additionally `mut` buffers and scalar
//!   `Result<A,B>` results. `take` is refused with `HOST_TAKE_UNAVAILABLE`.
//!
//! Admission (fuel, function, signature, arity, exact types, per-array limit,
//! checked extents, alias rules, live elements, staging) completes before any
//! source step; a refused call reports zero steps, records no cleanup and
//! leaves every caller buffer unchanged.
//!
//! Aliasing uses each buffer's checked half-open byte extent, computed from its
//! address as an integer without dereferencing. Empty extents never alias.
//! Several read arguments may cover the same memory (read loans coexist):
//! - an exact alias (same element type, address and length) shares one owner,
//!   exactly like an in-source call passing one owner to both parameters;
//! - a partial overlap of the same element type gets one private owner per
//!   argument (read-only, so no source step can observe the difference);
//! - overlapping extents of different element types are refused;
//! - any overlap involving a `mut` buffer is refused.
//!
//! Host arrays count once toward the live-element limit: the union of their
//! extents, in elements. This engine is a copying semantic oracle, not a
//! zero-copy binding and not a native engine: the private owners are host
//! staging outside the language limit, reserved fallibly within a separate
//! byte bound. A `mut` buffer is written back only after a normal return
//! (including a returned `Result::Err`) and a precheck of every `mut` owner.
use super::{CheckedProgram, EvalError, Evaluator, MAX_ARRAY_LEN, Observation, Value, admit_live};
use crate::{
    front::{Mode, Type},
    numeric_contract::{F32Bits, F64Bits},
};
use std::{
    cell::{Ref, RefCell},
    rc::Rc,
};

/// Maximum parameters, matching the scalar export facade.
pub const MAX_PARAMETERS: usize = 8;

/// The widest admitted element (`f64`): the staging bound counts every element at this size.
const MAX_ELEMENT_BYTES: usize = 8;

/// Bytes of one element of an admitted host array type (HA-1.4): `Array<f64>` is eight, `Array<Bool>` one
/// (a `bool` is exactly 0 or 1), every other admitted array four.
fn element_bytes(ty: &Type) -> usize {
    match ty {
        Type::ArrayF64 => 8,
        Type::ArrayBool => 1,
        _ => 4,
    }
}

/// Host staging bound for one call (HA-1.1 §6), outside the language
/// live-element limit. It covers reference copies (each exact-alias group
/// once, at most every parameter at the raised per-array limit) plus future
/// native rollback snapshots of disjoint `mut` buffers within the language
/// live-element bound. This reference engine never needs snapshots.
const STAGING_LIMIT_BYTES: usize = MAX_PARAMETERS * MAX_ARRAY_LEN * MAX_ELEMENT_BYTES
    + super::MAX_LIVE_ELEMENTS * MAX_ELEMENT_BYTES;

/// One borrowed host buffer of exact element type.
#[derive(Clone, Copy, Debug)]
pub enum HostReadArray<'a> {
    I32(&'a [i32]),
    U32(&'a [u32]),
    F32(&'a [F32Bits]),
    /// Borrow native binary32 storage directly; reference execution copies exact bits.
    Float32(&'a [f32]),
    /// Borrow native binary64 storage directly; reference execution copies exact bits (HA-1.4).
    Float64(&'a [f64]),
    /// Borrow bool storage. The host proves every byte is 0 or 1 before it builds this slice (HA-1.4).
    Bool(&'a [bool]),
}

impl HostReadArray<'_> {
    pub fn len(&self) -> usize {
        match self {
            Self::I32(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::Float32(v) => v.len(),
            Self::Float64(v) => v.len(),
            Self::Bool(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn array_type(&self) -> Type {
        match self {
            Self::I32(_) => Type::ArrayI32,
            Self::U32(_) => Type::ArrayU32,
            Self::F32(_) | Self::Float32(_) => Type::ArrayF32,
            Self::Float64(_) => Type::ArrayF64,
            Self::Bool(_) => Type::ArrayBool,
        }
    }

    /// The buffer's address as an integer; never dereferenced.
    fn address(&self) -> usize {
        match self {
            Self::I32(v) => v.as_ptr().addr(),
            Self::U32(v) => v.as_ptr().addr(),
            Self::F32(v) => v.as_ptr().addr(),
            Self::Float32(v) => v.as_ptr().addr(),
            Self::Float64(v) => v.as_ptr().addr(),
            Self::Bool(v) => v.as_ptr().addr(),
        }
    }

    /// A private owner holding a copy, reserved fallibly.
    fn to_value(self) -> Option<Value> {
        fn copy<T: Copy>(v: &[T]) -> Option<Rc<RefCell<Vec<T>>>> {
            let mut out = Vec::new();
            out.try_reserve_exact(v.len()).ok()?;
            out.extend_from_slice(v);
            Some(Rc::new(RefCell::new(out)))
        }
        Some(match self {
            Self::I32(v) => Value::ArrayI32(copy(v)?),
            Self::U32(v) => Value::ArrayU32(copy(v)?),
            Self::F32(v) => Value::ArrayF32(copy(v)?),
            Self::Float32(v) => {
                let mut out = Vec::new();
                out.try_reserve_exact(v.len()).ok()?;
                out.extend(v.iter().map(|x| F32Bits::from_bits(x.to_bits())));
                Value::ArrayF32(Rc::new(RefCell::new(out)))
            }
            Self::Float64(v) => {
                let mut out = Vec::new();
                out.try_reserve_exact(v.len()).ok()?;
                out.extend(v.iter().map(|x| F64Bits::from_bits(x.to_bits())));
                Value::ArrayF64(Rc::new(RefCell::new(out)))
            }
            Self::Bool(v) => Value::ArrayBool(copy(v)?),
        })
    }

    /// Whether `value` still holds exactly this buffer's elements.
    fn same_as(&self, value: &Value) -> bool {
        match (self, value) {
            (Self::I32(a), Value::ArrayI32(b)) => **a == *b.borrow(),
            (Self::U32(a), Value::ArrayU32(b)) => **a == *b.borrow(),
            (Self::F32(a), Value::ArrayF32(b)) => **a == *b.borrow(),
            (Self::Float32(a), Value::ArrayF32(b)) => {
                let b = b.borrow();
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits())
            }
            (Self::Float64(a), Value::ArrayF64(b)) => {
                let b = b.borrow();
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits())
            }
            (Self::Bool(a), Value::ArrayBool(b)) => **a == *b.borrow(),
            _ => false,
        }
    }
}

impl PartialEq for HostReadArray<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::I32(a), Self::I32(b)) => a == b,
            (Self::U32(a), Self::U32(b)) => a == b,
            (Self::F32(a), Self::F32(b)) => a == b,
            (Self::Float32(a), Self::Float32(b)) => {
                a.len() == b.len() && a.iter().zip(*b).all(|(a, b)| a.to_bits() == b.to_bits())
            }
            (Self::Float64(a), Self::Float64(b)) => {
                a.len() == b.len() && a.iter().zip(*b).all(|(a, b)| a.to_bits() == b.to_bits())
            }
            (Self::Bool(a), Self::Bool(b)) => a == b,
            _ => false,
        }
    }
}
impl Eq for HostReadArray<'_> {}

/// One host argument; scalars are exact `i32`, `u32`, `f32`, `f64`, `Bool` or `Unit`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostReadArgument<'a> {
    I32(i32),
    U32(u32),
    F32(F32Bits),
    F64(F64Bits),
    Bool(bool),
    Unit,
    Array(HostReadArray<'a>),
}

/// One exclusively borrowed host buffer for a `mut` parameter. The callee may
/// update elements but never resize (spec N3.2.6; a `mut`
/// parameter cannot be reassigned).
#[derive(Debug)]
pub enum HostMutArray<'a> {
    I32(&'a mut [i32]),
    U32(&'a mut [u32]),
    F32(&'a mut [F32Bits]),
    /// Borrow native binary32 storage directly, with exact-bit rollback.
    Float32(&'a mut [f32]),
    /// Borrow native binary64 storage directly, with exact-bit rollback (HA-1.4).
    Float64(&'a mut [f64]),
    /// Borrow bool storage (HA-1.4); see [`HostReadArray::Bool`].
    Bool(&'a mut [bool]),
}

impl HostMutArray<'_> {
    pub fn len(&self) -> usize {
        self.view().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A shared view of the current contents.
    fn view(&self) -> HostReadArray<'_> {
        match self {
            Self::I32(v) => HostReadArray::I32(v),
            Self::U32(v) => HostReadArray::U32(v),
            Self::F32(v) => HostReadArray::F32(v),
            Self::Float32(v) => HostReadArray::Float32(v),
            Self::Float64(v) => HostReadArray::Float64(v),
            Self::Bool(v) => HostReadArray::Bool(v),
        }
    }
}

impl PartialEq for HostMutArray<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.view() == other.view()
    }
}
impl Eq for HostMutArray<'_> {}

/// One host argument for [`observe_host_call`]: exact scalars, `read` buffers
/// and `mut` buffers.
#[derive(Debug, PartialEq, Eq)]
pub enum HostArgument<'a> {
    I32(i32),
    U32(u32),
    F32(F32Bits),
    F64(F64Bits),
    Bool(bool),
    Unit,
    Read(HostReadArray<'a>),
    Mut(HostMutArray<'a>),
}

impl<'a> From<HostReadArgument<'a>> for HostArgument<'a> {
    fn from(arg: HostReadArgument<'a>) -> Self {
        match arg {
            HostReadArgument::I32(x) => Self::I32(x),
            HostReadArgument::U32(x) => Self::U32(x),
            HostReadArgument::F32(x) => Self::F32(x),
            HostReadArgument::F64(x) => Self::F64(x),
            HostReadArgument::Bool(x) => Self::Bool(x),
            HostReadArgument::Unit => Self::Unit,
            HostReadArgument::Array(array) => Self::Read(array),
        }
    }
}

fn scalar(ty: &Type) -> bool {
    matches!(
        ty,
        Type::I32 | Type::U32 | Type::F32 | Type::F64 | Type::Bool | Type::Unit
    )
}

fn owner(ty: &Type) -> bool {
    matches!(
        ty,
        Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::ArrayF64 | Type::ArrayBool
    )
}

/// `Result<A,B>` with `A` and `B` each `i32`, `u32` or `Bool`, not nested.
fn scalar_result(ty: &Type) -> bool {
    let Type::Named(name) = ty else {
        return false;
    };
    let payload = |t: &Type| matches!(t, Type::I32 | Type::U32 | Type::Bool);
    crate::front::result_parts(name).is_some_and(|(ok, err)| payload(&ok) && payload(&err))
}

fn refuse(code: &'static str, message: &'static str) -> Observation {
    Observation {
        result: Err(EvalError::tool(code, message)),
        steps_used: 0,
        cleanup_trace: Vec::new(),
    }
}

/// Call `name` with scalars and read-only host buffers.
pub fn observe_host_read_call(
    program: &CheckedProgram,
    name: &str,
    args: &[HostReadArgument<'_>],
    fuel_limit: u64,
) -> Observation {
    observe_bounded(program, name, args, fuel_limit, STAGING_LIMIT_BYTES)
}

fn observe_bounded(
    program: &CheckedProgram,
    name: &str,
    args: &[HostReadArgument<'_>],
    fuel_limit: u64,
    staging_limit: usize,
) -> Observation {
    let mut args: Vec<HostArgument<'_>> = args.iter().map(|&arg| arg.into()).collect();
    observe_inner(
        program,
        name,
        &mut args,
        fuel_limit,
        staging_limit,
        Policy::ReadOnly,
        Inject::None,
    )
}

/// Call `name` with scalars, `read` and `mut` host buffers (HA-1.1 reference
/// engine; it copies and never runs at the caller's address).
///
/// A normal return, including a returned `Result::Err` value, commits every
/// `mut` buffer. Any other outcome (trap, fuel exhaustion, tool limit,
/// internal error) and every refusal leaves all caller buffers unchanged:
/// they are written only after the call returned normally and every `mut`
/// owner passed its precheck. `take` parameters are refused before any step.
pub fn observe_host_call(
    program: &CheckedProgram,
    name: &str,
    args: &mut [HostArgument<'_>],
    fuel_limit: u64,
) -> Observation {
    observe_inner(
        program,
        name,
        args,
        fuel_limit,
        STAGING_LIMIT_BYTES,
        Policy::HostCall,
        Inject::None,
    )
}

/// Which entry point is admitting: the approved read-only API keeps its exact
/// refusals; the host-call API admits `mut` and scalar `Result` results.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Policy {
    ReadOnly,
    HostCall,
}

/// Private fault-injection seam. Public entry points pass `None`.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Inject {
    None,
    /// The `n`th staging reservation (0-based) fails.
    ReserveFailAt(usize),
    PanicAfterCopyIn,
    PanicAfterEval,
    /// After every precheck passed, before the first caller write.
    PanicBeforeCommit,
    /// A foreign borrow of the first `mut` owner is live at precheck.
    BorrowedAtCommit,
    /// The first `mut` owner has a different length at precheck.
    ResizedAtCommit,
    /// The first `read` owner is altered after evaluation, before the read
    /// post-check (an invariant checked source cannot violate).
    ChangedReadAtEnd,
}

/// One prechecked writeback: a caller slice and an owner borrow of equal
/// length and element type.
enum Commit<'h, 'g> {
    I32(&'h mut [i32], Ref<'g, Vec<i32>>),
    U32(&'h mut [u32], Ref<'g, Vec<u32>>),
    F32(&'h mut [F32Bits], Ref<'g, Vec<F32Bits>>),
    Float32(&'h mut [f32], Ref<'g, Vec<F32Bits>>),
    Float64(&'h mut [f64], Ref<'g, Vec<F64Bits>>),
    Bool(&'h mut [bool], Ref<'g, Vec<bool>>),
}

fn observe_inner(
    program: &CheckedProgram,
    name: &str,
    args: &mut [HostArgument<'_>],
    fuel_limit: u64,
    staging_limit: usize,
    policy: Policy,
    inject: Inject,
) -> Observation {
    // Staging, evaluation and commit all run on the evaluator's own stack.
    super::stack::observe_on_own_stack(move || {
        observe_on_stack(
            program,
            name,
            args,
            fuel_limit,
            staging_limit,
            policy,
            inject,
        )
    })
}

fn observe_on_stack(
    program: &CheckedProgram,
    name: &str,
    args: &mut [HostArgument<'_>],
    fuel_limit: u64,
    staging_limit: usize,
    policy: Policy,
    inject: Inject,
) -> Observation {
    if fuel_limit == 0 {
        return refuse("INVALID_FUEL", "fuel must be positive");
    }
    let Some(function) = program.program.functions.iter().find(|f| f.name == name) else {
        return refuse("NATIVE_ABI", "unknown function");
    };
    let host_call = policy == Policy::HostCall;
    let signature_ok = (scalar(&function.result)
        || owner(&function.result)
        || host_call && scalar_result(&function.result))
        && function.params.len() <= MAX_PARAMETERS
        && function.params.iter().all(|p| match p.mode {
            None => scalar(&p.ty),
            Some(Mode::Read) => owner(&p.ty),
            Some(Mode::Mut | Mode::Take) if host_call => owner(&p.ty),
            Some(Mode::Mut | Mode::Take) => false,
        });
    if !signature_ok {
        return refuse(
            "NATIVE_ABI",
            if host_call {
                "signature is outside host reference admission"
            } else {
                "signature is outside read-only host reference admission"
            },
        );
    }
    if function.params.iter().any(|p| p.mode == Some(Mode::Take)) {
        return refuse(
            "HOST_TAKE_UNAVAILABLE",
            "host buffers cannot be consumed; take has no host-boundary contract",
        );
    }
    if args.len() != function.params.len() {
        return refuse("NATIVE_ABI", "argument count does not match");
    }
    let mut values = Vec::with_capacity(args.len());
    for (param, arg) in function.params.iter().zip(args.iter()) {
        let value = match (&param.ty, param.mode, arg) {
            (Type::I32, None, HostArgument::I32(x)) => Value::I32(*x),
            (Type::U32, None, HostArgument::U32(x)) => Value::U32(*x),
            (Type::F32, None, HostArgument::F32(x)) => Value::F32(*x),
            (Type::F64, None, HostArgument::F64(x)) => Value::F64(*x),
            (Type::Bool, None, HostArgument::Bool(x)) => Value::Bool(*x),
            (Type::Unit, None, HostArgument::Unit) => Value::Unit,
            (ty, Some(Mode::Read), HostArgument::Read(array)) if *ty == array.array_type() => {
                Value::Unit // placeholder; arrays are built after admission
            }
            (ty, Some(Mode::Mut), HostArgument::Mut(array)) if *ty == array.view().array_type() => {
                Value::Unit // placeholder; arrays are built after admission
            }
            _ => return refuse("NATIVE_ABI", "argument type does not match"),
        };
        values.push(value);
    }
    // Shared views of every array argument, in argument order, and whether
    // each is `mut`.
    let arrays: Vec<(HostReadArray<'_>, bool)> = args
        .iter()
        .filter_map(|arg| match arg {
            HostArgument::Read(array) => Some((*array, false)),
            HostArgument::Mut(array) => Some((array.view(), true)),
            _ => None,
        })
        .collect();
    if arrays.iter().any(|(array, _)| array.len() > MAX_ARRAY_LEN) {
        return refuse("ARRAY_SIZE_LIMIT", "host array exceeds element limits");
    }
    let extents: Vec<Extent> = arrays
        .iter()
        .map(|(array, mutable)| Extent {
            element: array.array_type(),
            address: array.address(),
            length: array.len(),
            mutable: *mutable,
        })
        .collect();
    let plan = match admit(&extents, staging_limit) {
        Ok(plan) => plan,
        Err((code, message)) => return refuse(code, message),
    };
    // Admission passed: one private owner per exact-alias group (a `mut`
    // buffer is always its own group). A group leader always precedes its
    // aliases. The staging bound is also enforced where copies are reserved,
    // so it covers what is actually copied.
    let mutable_count = arrays.iter().filter(|(_, mutable)| *mutable).count();
    let mut owners: Vec<Value> = Vec::with_capacity(arrays.len());
    let mut handles = Vec::new();
    let mut mut_owners: Vec<Value> = Vec::new();
    let mut commits: Vec<Commit<'_, '_>> = Vec::new();
    if mut_owners.try_reserve_exact(mutable_count).is_err()
        || commits.try_reserve_exact(mutable_count).is_err()
    {
        return refuse("HOST_STAGING_LIMIT", "host staging reservation failed");
    }
    let mut staged = 0usize;
    let mut reservations = 0usize;
    for (index, (array, mutable)) in arrays.iter().enumerate() {
        let leader = plan.group[index];
        let owner = if leader == index {
            staged = staged.saturating_add(
                array
                    .len()
                    .saturating_mul(element_bytes(&array.array_type())),
            );
            if staged > staging_limit {
                return refuse("HOST_STAGING_LIMIT", "host staging exceeds its byte bound");
            }
            let injected = inject == Inject::ReserveFailAt(reservations);
            reservations += 1;
            let Some(value) = array.to_value().filter(|_| !injected) else {
                return refuse("HOST_STAGING_LIMIT", "host staging reservation failed");
            };
            if *mutable {
                mut_owners.push(share(&value));
            } else {
                handles.push((*array, share(&value)));
            }
            value
        } else {
            share(&owners[leader])
        };
        owners.push(owner);
    }
    let mut owners = owners.into_iter();
    for (slot, arg) in values.iter_mut().zip(args.iter()) {
        if matches!(arg, HostArgument::Read(_) | HostArgument::Mut(_))
            && let Some(owner) = owners.next()
        {
            *slot = owner;
        }
    }
    if inject == Inject::PanicAfterCopyIn {
        panic!("injected panic after copy-in");
    }
    let live = plan.live_elements;
    let mut evaluator = Evaluator {
        program,
        types: program.types_for(name),
        fuel_limit,
        steps_used: 0,
        call_depth: 0,
        trace: Vec::new(),
        live_elements: live,
        pending_result_error: None,
    };
    let mut result = evaluator.call_user(name, values);
    if inject == Inject::PanicAfterEval {
        panic!("injected panic after evaluation");
    }
    if let (Inject::ChangedReadAtEnd, Some((_, Value::ArrayI32(v)))) = (inject, handles.first()) {
        v.borrow_mut().push(0);
    }
    if handles.iter().any(|(array, value)| !array.same_as(value)) {
        result = Err(EvalError::tool(
            "INTERNAL_HOST_READ",
            "read argument changed during the call",
        ));
    }
    // Only a normal return commits; a returned `Result::Err` is a normal value.
    if result.is_ok() {
        let foreign = match (inject, mut_owners.first()) {
            (Inject::BorrowedAtCommit, Some(Value::ArrayI32(v))) => Some(v.borrow_mut()),
            _ => None,
        };
        if let (Inject::ResizedAtCommit, Some(Value::ArrayI32(v))) = (inject, mut_owners.first()) {
            v.borrow_mut().push(0);
        }
        // Phase 1: precheck every `mut` owner before any caller write. Pushes
        // stay within the capacity reserved above.
        let prechecked = precheck(args, &mut_owners, &mut commits);
        drop(foreign);
        if prechecked.is_err() {
            commits.clear();
            result = Err(EvalError::tool(
                "INTERNAL_HOST_COMMIT",
                "a mut owner failed its writeback precheck",
            ));
        } else {
            if inject == Inject::PanicBeforeCommit {
                panic!("injected panic before commit");
            }
            // Phase 2: equal-length element copies. `copy_from_slice` panics
            // only on a length mismatch, which phase 1 excluded; nothing here
            // allocates or acquires a borrow.
            for commit in commits.drain(..) {
                match commit {
                    Commit::I32(host, owner) => host.copy_from_slice(&owner),
                    Commit::U32(host, owner) => host.copy_from_slice(&owner),
                    Commit::F32(host, owner) => host.copy_from_slice(&owner),
                    Commit::Float32(host, owner) => {
                        for (slot, bits) in host.iter_mut().zip(owner.iter()) {
                            *slot = f32::from_bits(bits.to_bits());
                        }
                    }
                    Commit::Float64(host, owner) => {
                        for (slot, bits) in host.iter_mut().zip(owner.iter()) {
                            *slot = f64::from_bits(bits.to_bits());
                        }
                    }
                    Commit::Bool(host, owner) => host.copy_from_slice(&owner),
                }
            }
        }
    }
    Observation {
        result,
        steps_used: evaluator.steps_used,
        cleanup_trace: evaluator.trace,
    }
}

/// Pair every `mut` argument with its owner: same element type, owner not
/// borrowed, equal length. Writes nothing; on failure no commit is kept.
fn precheck<'h, 'g>(
    args: &'h mut [HostArgument<'_>],
    owners: &'g [Value],
    commits: &mut Vec<Commit<'h, 'g>>,
) -> Result<(), ()> {
    let mut owners = owners.iter();
    for arg in args.iter_mut() {
        let HostArgument::Mut(host) = arg else {
            continue;
        };
        let owner = owners.next().ok_or(())?;
        let commit = match (host, owner) {
            (HostMutArray::I32(host), Value::ArrayI32(v)) => {
                Commit::I32(host, v.try_borrow().map_err(|_| ())?)
            }
            (HostMutArray::U32(host), Value::ArrayU32(v)) => {
                Commit::U32(host, v.try_borrow().map_err(|_| ())?)
            }
            (HostMutArray::F32(host), Value::ArrayF32(v)) => {
                Commit::F32(host, v.try_borrow().map_err(|_| ())?)
            }
            (HostMutArray::Float32(host), Value::ArrayF32(v)) => {
                Commit::Float32(host, v.try_borrow().map_err(|_| ())?)
            }
            (HostMutArray::Float64(host), Value::ArrayF64(v)) => {
                Commit::Float64(host, v.try_borrow().map_err(|_| ())?)
            }
            (HostMutArray::Bool(host), Value::ArrayBool(v)) => {
                Commit::Bool(host, v.try_borrow().map_err(|_| ())?)
            }
            _ => return Err(()),
        };
        let equal = match &commit {
            Commit::I32(host, owner) => host.len() == owner.len(),
            Commit::U32(host, owner) => host.len() == owner.len(),
            Commit::F32(host, owner) => host.len() == owner.len(),
            Commit::Float32(host, owner) => host.len() == owner.len(),
            Commit::Float64(host, owner) => host.len() == owner.len(),
            Commit::Bool(host, owner) => host.len() == owner.len(),
        };
        if !equal || commits.len() == commits.capacity() {
            return Err(());
        }
        commits.push(commit);
    }
    if owners.next().is_some() {
        return Err(());
    }
    Ok(())
}

/// Integer metadata of one host buffer: element type, address and length.
#[derive(Clone, Debug)]
struct Extent {
    element: Type,
    address: usize,
    length: usize,
    /// A `mut` buffer: it may overlap nothing.
    mutable: bool,
}

#[derive(Debug, PartialEq, Eq)]
struct Plan {
    /// For each array, the index of the first array it exactly aliases
    /// (itself when it starts a group). Only group leaders are staged.
    group: Vec<usize>,
    /// Union of all extents, in elements: the host's live contribution.
    live_elements: usize,
}

type Refusal = (&'static str, &'static str);

/// Checked half-open byte range `[address, address + length * element size)`.
fn range(extent: &Extent) -> Result<(usize, usize), Refusal> {
    let size = element_bytes(&extent.element);
    let end = extent
        .length
        .checked_mul(size)
        .and_then(|bytes| extent.address.checked_add(bytes))
        .ok_or(("HOST_ARRAY_EXTENT", "host array extent overflows"))?;
    if !extent.address.is_multiple_of(size) {
        return Err(("HOST_ARRAY_EXTENT", "host array is not element aligned"));
    }
    Ok((extent.address, end))
}

/// HA-1 admission over integer metadata only, in contract order: checked
/// extents, alias rules, live elements, staging bound.
fn admit(extents: &[Extent], staging_limit: usize) -> Result<Plan, Refusal> {
    let ranges = extents.iter().map(range).collect::<Result<Vec<_>, _>>()?;
    let mut group: Vec<usize> = (0..extents.len()).collect();
    for right in 0..extents.len() {
        for left in 0..right {
            let (a, b) = (ranges[left], ranges[right]);
            // Empty extents never alias.
            let overlap = a.0 < a.1 && b.0 < b.1 && a.0 < b.1 && b.0 < a.1;
            if !overlap {
                continue;
            }
            if extents[left].mutable || extents[right].mutable {
                return Err((
                    "HOST_ARRAY_ALIAS",
                    "a mut host array overlaps another argument",
                ));
            }
            if extents[left].element != extents[right].element {
                return Err((
                    "HOST_ARRAY_ALIAS",
                    "host arrays of different element types overlap",
                ));
            }
            if a == b && group[right] == right {
                group[right] = group[left];
            }
        }
    }
    // Overlap of different element types was refused above, so overlapping extents share one element size and a
    // merged range has one; extents that only touch merge when their element sizes agree. The union counts
    // elements of that size (HA-1.4).
    let mut sorted: Vec<(usize, usize, usize)> = ranges
        .iter()
        .zip(extents)
        .filter(|(r, _)| r.0 < r.1)
        .map(|(r, e)| (r.0, r.1, element_bytes(&e.element)))
        .collect();
    sorted.sort_unstable();
    let union_overflow = ("HOST_ARRAY_EXTENT", "host array union overflows");
    let mut union_elements = 0usize;
    let mut current: Option<(usize, usize, usize)> = None;
    for (start, end, size) in sorted {
        current = match current {
            Some((s, e, z)) if start < e || (start == e && size == z) => Some((s, e.max(end), z)),
            Some((s, e, z)) => {
                union_elements = union_elements
                    .checked_add((e - s) / z)
                    .ok_or(union_overflow)?;
                Some((start, end, size))
            }
            None => Some((start, end, size)),
        };
    }
    if let Some((s, e, z)) = current {
        union_elements = union_elements
            .checked_add((e - s) / z)
            .ok_or(union_overflow)?;
    }
    let mut live_elements = 0;
    admit_live(&mut live_elements, union_elements).map_err(|_| {
        (
            "ARRAY_SIZE_LIMIT",
            "host arrays exceed the live element limit",
        )
    })?;
    (0..extents.len())
        .filter(|&index| group[index] == index)
        .try_fold(0usize, |total, index| {
            total.checked_add(ranges[index].1 - ranges[index].0)
        })
        .filter(|&total| total <= staging_limit)
        .ok_or(("HOST_STAGING_LIMIT", "host staging exceeds its byte bound"))?;
    Ok(Plan {
        group,
        live_elements,
    })
}

/// A second handle to the same array storage.
fn share(value: &Value) -> Value {
    match value {
        Value::ArrayI32(v) => Value::ArrayI32(Rc::clone(v)),
        Value::ArrayU32(v) => Value::ArrayU32(Rc::clone(v)),
        Value::ArrayF32(v) => Value::ArrayF32(Rc::clone(v)),
        Value::ArrayF64(v) => Value::ArrayF64(Rc::clone(v)),
        Value::ArrayBool(v) => Value::ArrayBool(Rc::clone(v)),
        _ => Value::Unit,
    }
}

#[cfg(test)]
mod tests;
