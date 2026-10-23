//! The `invar-math-cr-v1` source library for `f32` and `f64`.
//!
//! Every function returns the correctly rounded (round-to-nearest-even) value
//! of the exact mathematical result, so every engine that implements it
//! returns the same encoding by construction. NaN results are the canonical
//! quiet NaN of the type. The transcendental functions are CORE-MATH
//! (vendored in `crates/invar-math`); `sqrt` and the exact functions are the
//! strict numeric oracles in [`crate::numeric_contract`].

use crate::numeric_contract::{F32Bits, F64Bits};

/// Contract identifier recorded in docs, receipts and device artifacts.
pub const CONTRACT: &str = "invar-math-cr-v1";

/// The operand count of a float library intrinsic (3 for `fma`).
pub fn intrinsic_arity(name: &str) -> usize {
    MathFn::from_name(name).map_or(3, |f| f.arity())
}

mod device;
pub use device::{CUDA_KERNEL, cuda_map_min_nvrtc, cuda_map_source, msl_map_source};
#[cfg(feature = "cuda")]
pub use device::{load_cuda_map, nvrtc_admits};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MathFn {
    Sqrt,
    Cbrt,
    Hypot,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Sinh,
    Cosh,
    Tanh,
    Exp,
    Exp2,
    Expm1,
    Log,
    Log2,
    Log10,
    Log1p,
    Pow,
    Floor,
    Ceil,
    Trunc,
    Round,
    Abs,
    Copysign,
    Min,
    Max,
}

impl MathFn {
    pub const ALL: [Self; 29] = [
        Self::Sqrt,
        Self::Cbrt,
        Self::Hypot,
        Self::Sin,
        Self::Cos,
        Self::Tan,
        Self::Asin,
        Self::Acos,
        Self::Atan,
        Self::Atan2,
        Self::Sinh,
        Self::Cosh,
        Self::Tanh,
        Self::Exp,
        Self::Exp2,
        Self::Expm1,
        Self::Log,
        Self::Log2,
        Self::Log10,
        Self::Log1p,
        Self::Pow,
        Self::Floor,
        Self::Ceil,
        Self::Trunc,
        Self::Round,
        Self::Abs,
        Self::Copysign,
        Self::Min,
        Self::Max,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Sqrt => "sqrt",
            Self::Cbrt => "cbrt",
            Self::Hypot => "hypot",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Atan2 => "atan2",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Exp => "exp",
            Self::Exp2 => "exp2",
            Self::Expm1 => "expm1",
            Self::Log => "log",
            Self::Log2 => "log2",
            Self::Log10 => "log10",
            Self::Log1p => "log1p",
            Self::Pow => "pow",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Trunc => "trunc",
            Self::Round => "round",
            Self::Abs => "abs",
            Self::Copysign => "copysign",
            Self::Min => "min",
            Self::Max => "max",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|f| f.name() == name)
    }

    pub const fn arity(self) -> usize {
        match self {
            Self::Hypot | Self::Atan2 | Self::Pow | Self::Copysign | Self::Min | Self::Max => 2,
            _ => 1,
        }
    }
}

/// Reference evaluation in binary32. `args` has exactly `f.arity()` items;
/// `atan2` takes `(y, x)`.
pub fn eval_f32(f: MathFn, args: &[F32Bits]) -> F32Bits {
    let (x, y) = (args[0], args.get(1).copied().unwrap_or(F32Bits::POS_ZERO));
    let result = match core_math(f) {
        Some(c) => F32Bits::from_bits(invar_math::f32_bits(c, x.to_bits(), y.to_bits())),
        None => match f {
            MathFn::Floor => x.floor(),
            MathFn::Ceil => x.ceil(),
            MathFn::Trunc => x.trunc(),
            MathFn::Round => x.round(),
            MathFn::Abs => x.abs(),
            MathFn::Copysign => x.copysign(y),
            MathFn::Min => x.min(y),
            MathFn::Max => x.max(y),
            _ => x.sqrt(),
        },
    };
    if result.is_nan() {
        F32Bits::NAN
    } else {
        result
    }
}

/// Reference evaluation in binary64; see [`eval_f32`].
pub fn eval_f64(f: MathFn, args: &[F64Bits]) -> F64Bits {
    let (x, y) = (args[0], args.get(1).copied().unwrap_or(F64Bits::POS_ZERO));
    let result = match core_math(f) {
        Some(c) => F64Bits::from_bits(invar_math::f64_bits(c, x.to_bits(), y.to_bits())),
        None => match f {
            MathFn::Floor => x.floor(),
            MathFn::Ceil => x.ceil(),
            MathFn::Trunc => x.trunc(),
            MathFn::Round => x.round(),
            MathFn::Abs => x.abs(),
            MathFn::Copysign => x.copysign(y),
            MathFn::Min => x.min(y),
            MathFn::Max => x.max(y),
            _ => x.sqrt(),
        },
    };
    if result.is_nan() {
        F64Bits::NAN
    } else {
        result
    }
}

/// The `numeric_contracts.fold` receipt of explain-v5 and invar-diagnostics-v3
/// (ENG-03, D12): each declared tier a lowered fold used (`tree`, `exact`, or
/// `relaxed`, which a Strict function's `reassoc_add` folds also record), to
/// the algorithm that ran. `None` when no fold was lowered.
pub fn fold_receipt(program: &crate::mir::Program) -> Option<serde_json::Value> {
    (!program.lowered_folds.is_empty()).then(|| {
        program
            .lowered_folds
            .iter()
            .map(|(declared, ran)| (declared.name().to_owned(), ran.name().into()))
            .collect::<serde_json::Map<_, _>>()
            .into()
    })
}

/// The numeric contracts a lowered program relies on beyond `strict-f32-v1`,
/// for explain and run receipts: `binary64` when any value is f64 and `math`
/// when it calls a library function. `None` for programs that use neither.
pub fn contracts(program: &crate::mir::Program) -> Option<serde_json::Value> {
    use crate::{front::Type, mir::Operation};
    let instructions = || {
        program
            .functions
            .iter()
            .flat_map(|f| &f.blocks)
            .flat_map(|b| &b.instructions)
    };
    let binary64 =
        instructions().any(|i| matches!(i.output, Some((_, Type::F64 | Type::ArrayF64))));
    // C4 interim: f32 sin, cos and sqrt are invar-trig-f32-v1.
    let trig_f32 = |i: &crate::mir::Instruction| {
        matches!(&i.kind, Operation::Call { name, .. } if matches!(name.as_str(), "sin" | "cos" | "sqrt"))
            && matches!(i.output, Some((_, Type::F32)))
    };
    let trig = instructions().any(trig_f32);
    let math = instructions().any(|i| {
        matches!(&i.kind, Operation::Call { name, .. } if MathFn::from_name(name).is_some())
            && !trig_f32(i)
    });
    let mut record = serde_json::Map::new();
    if binary64 {
        record.insert(
            "binary64".into(),
            crate::numeric_contract::NumericPolicy::STRICT_F64
                .id()
                .into(),
        );
    }
    if math {
        record.insert("math".into(), CONTRACT.into());
    }
    if trig {
        record.insert("trig".into(), invar_device::trig_f32::CONTRACT.into());
    }
    (!record.is_empty()).then_some(serde_json::Value::Object(record))
}

/// The CORE-MATH entry point of a correctly rounded transcendental, or `None`
/// for `sqrt` and the exact functions.
pub fn core_math(f: MathFn) -> Option<invar_math::Function> {
    use invar_math::Function as C;
    Some(match f {
        MathFn::Cbrt => C::Cbrt,
        MathFn::Hypot => C::Hypot,
        MathFn::Sin => C::Sin,
        MathFn::Cos => C::Cos,
        MathFn::Tan => C::Tan,
        MathFn::Asin => C::Asin,
        MathFn::Acos => C::Acos,
        MathFn::Atan => C::Atan,
        MathFn::Atan2 => C::Atan2,
        MathFn::Sinh => C::Sinh,
        MathFn::Cosh => C::Cosh,
        MathFn::Tanh => C::Tanh,
        MathFn::Exp => C::Exp,
        MathFn::Exp2 => C::Exp2,
        MathFn::Expm1 => C::Expm1,
        MathFn::Log => C::Log,
        MathFn::Log2 => C::Log2,
        MathFn::Log10 => C::Log10,
        MathFn::Log1p => C::Log1p,
        MathFn::Pow => C::Pow,
        MathFn::Sqrt
        | MathFn::Floor
        | MathFn::Ceil
        | MathFn::Trunc
        | MathFn::Round
        | MathFn::Abs
        | MathFn::Copysign
        | MathFn::Min
        | MathFn::Max => return None,
    })
}

/// Explain reason for f64 code under a Metal request: Metal has no binary64,
/// so placement keeps every f64 computation on the CPU (no emulation).
pub const METAL_NO_F64: &str = "metal-no-f64";

/// One `(function, span)` per function that computes an f64 value, at its
/// first such instruction: the sites `metal-no-f64` keeps on the CPU.
pub fn f64_sites(program: &crate::mir::Program) -> Vec<(String, crate::front::Span)> {
    use crate::front::Type;
    program
        .functions
        .iter()
        .filter_map(|function| {
            function
                .blocks
                .iter()
                .flat_map(|b| &b.instructions)
                .find(|i| matches!(i.output, Some((_, Type::F64 | Type::ArrayF64))))
                .map(|i| (function.name.clone(), i.span))
        })
        .collect()
}
