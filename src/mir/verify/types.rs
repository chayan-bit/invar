#[derive(Clone)]
struct Definition {
    block: BlockId,
    position: Option<usize>,
    ty: Type,
}

#[derive(Clone, PartialEq, Eq)]
struct FlowState {
    available: HashSet<String>,
    loans: HashMap<ValueId, (String, Mode)>,
    views: HashMap<String, ViewFact>,
    view_values: HashMap<ValueId, ViewFact>,
    partial: HashMap<String, HashSet<usize>>,
    taken_payloads: HashSet<ValueId>,
    dropped_values: HashSet<ValueId>,
}

#[derive(Clone, PartialEq, Eq)]
struct ViewFact {
    root: String,
    interval: Option<(u64, u64)>,
    stride: Option<u64>,
    mutable: bool,
    parent: Option<String>,
}

impl Operation {
    fn uses(&self) -> Vec<ValueId> {
        match self {
            Self::Unary { value, .. }
            | Self::ForNext(value)
            | Self::VariantTag(value)
            | Self::VariantTakePayload { value, .. }
            | Self::DropValue(value)
            | Self::Bind { value, .. }
            | Self::Store { value, .. } => vec![*value],
            Self::Binary { left, right, .. } => vec![*left, *right],
            Self::Index { array, index } => vec![*array, *index],
            Self::Field { record, .. } => vec![*record],
            Self::VariantPayload { value, .. } => vec![*value],
            Self::Variant { payload, .. } => payload.iter().copied().collect(),
            Self::Record { fields, .. } => fields.iter().map(|(_, value)| *value).collect(),
            Self::Array(values) | Self::Call { args: values, .. } => values.clone(),
            Self::StoreIndex { index, value, .. } => vec![*value, *index],
            _ => Vec::new(),
        }
    }
}

impl Terminator {
    fn uses(&self) -> Vec<ValueId> {
        match self {
            Self::Goto { args, .. } => args.clone(),
            Self::Branch { condition, .. } => vec![*condition],
            Self::Return { value, .. } => value.iter().copied().collect(),
        }
    }

    fn targets(&self) -> Vec<(BlockId, &[ValueId])> {
        match self {
            Self::Goto { target, args, .. } => vec![(*target, args)],
            Self::Branch { yes, no, .. } => vec![(*yes, &[]), (*no, &[])],
            Self::Return { .. } => Vec::new(),
        }
    }

    fn span(&self) -> Span {
        match self {
            Self::Goto { span, .. } | Self::Branch { span, .. } | Self::Return { span, .. } => {
                *span
            }
        }
    }
}

fn contained(inner: Span, outer: Span) -> bool {
    inner.start >= outer.start && inner.end <= outer.end && inner.start < inner.end
}

fn result_type(op: BinaryOp, left: &Type, right: &Type) -> Option<Type> {
    if left != right {
        return None;
    }
    match op {
        BinaryOp::Eq | BinaryOp::Ne
            if matches!(left, Type::I32 | Type::U32 | Type::F32 | Type::F64 | Type::Bool) =>
        {
            Some(Type::Bool)
        }
        BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
            if matches!(left, Type::I32 | Type::U32 | Type::F32 | Type::F64) =>
        {
            Some(Type::Bool)
        }
        BinaryOp::Rem if matches!(left, Type::F32 | Type::F64) => None,
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem
            if matches!(left, Type::I32 | Type::U32 | Type::F32 | Type::F64) =>
        {
            Some(left.clone())
        }
        _ => None,
    }
}

/// The result type of a float intrinsic over `args`, or `None` when the
/// operands do not match its signature.
fn float_intrinsic(name: &str, args: &[&Type]) -> Option<Type> {
    if let Some((input, output)) = crate::check::conversion(name) {
        return (args == [&input]).then_some(output);
    }
    match (name, args) {
        ("filled_f32", [Type::U32, Type::F32]) => Some(Type::ArrayF32),
        ("filled_f64", [Type::U32, Type::F64]) => Some(Type::ArrayF64),
        ("tc_mul_add", [Type::F32, Type::F32, Type::F32]) => Some(Type::F32),
        ("tc_mul_add", _) | (crate::math::TRIG_F32_SIN | crate::math::TRIG_F32_COS, [Type::F64]) => None,
        _ => {
            let arity = crate::math::intrinsic_arity(name);
            let first = *args.first()?;
            (matches!(first, Type::F32 | Type::F64) && args.len() == arity && args.iter().all(|arg| *arg == first))
                .then(|| first.clone())
        }
    }
}

fn affine(ty: &Type) -> bool {
    matches!(ty, Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::ArrayF64 | Type::ArrayBool | Type::Named(_))
}

fn reference(op: &Operation) -> Option<Option<Mode>> {
    match op {
        Operation::ArrayRef { loan, .. } | Operation::RecordRef { loan, .. } => Some(*loan),
        _ => None,
    }
}
