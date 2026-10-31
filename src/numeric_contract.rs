//! Independent software oracle for the frozen M4-01 strict binary32 contract
//! (`strict-f32-v1`) and the v0.2 strict binary64 contract (`strict-f64-v1`).
//!
//! Exact rational arithmetic and integer rounding avoid dependence on host FP
//! flags, excess precision, implicit contraction, or denormal handling. This
//! module supplies source-independent decimal conversion and does not admit
//! device operations or parse program syntax.

mod binary64;
pub mod fold;
mod ieee;
pub mod storage;
pub use binary64::F64Bits;
pub use fold::{FoldTier, LoweredFolds};

/// A versioned numeric contract. `strict-f32-v1` stays the default;
/// `strict-f64-v1` applies the same rules to binary64. `TC_F16_F32` names the
/// explicit `tc_mul_add` contraction contract (D5 as amended); its reference
/// and bound live in `invar_device::tc`, and every strict consumer compares
/// against `STRICT_F32` or `STRICT_F64` and so refuses it. Version 2
/// (`docs/numeric.md`) adds the binary32 fold tiers `TREE_F32`,
/// `EXACT_F32` and `RELAXED_F32`; strict v2 spellings resolve to the v1
/// constants, so a strict program has one identity.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct NumericPolicy {
    version: u32,
    binary64: bool,
    tc: bool,
    fold: FoldTier,
}

/// The strict binary32 spelling stays `NumericPolicy { version: 1 }`: the
/// general-net receipt snapshot embeds this `Debug` text, so frozen ABI1-11
/// receipt ids stay byte-identical (`policy_v2_accepts_the_four_tiers_and_refuses_the_rest`).
impl std::fmt::Debug for NumericPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("NumericPolicy");
        debug.field("version", &self.version);
        if self.binary64 {
            debug.field("binary64", &true);
        }
        if self.tc {
            debug.field("tc", &true);
        }
        if self.fold != FoldTier::Strict {
            debug.field("fold", &self.fold);
        }
        debug.finish()
    }
}

/// A serialized policy cannot silently select an unknown numeric contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyError {
    UnsupportedVersion(u32),
    UnsupportedKind,
}

impl Default for NumericPolicy {
    fn default() -> Self {
        Self::STRICT_F32
    }
}

impl NumericPolicy {
    pub const STRICT_F32: Self = Self {
        version: 1,
        binary64: false,
        tc: false,
        fold: FoldTier::Strict,
    };
    pub const STRICT_F64: Self = Self {
        version: 1,
        binary64: true,
        tc: false,
        fold: FoldTier::Strict,
    };
    pub const TC_F16_F32: Self = Self {
        version: 1,
        binary64: false,
        tc: true,
        fold: FoldTier::Strict,
    };
    pub const TREE_F32: Self = Self::f32_fold(FoldTier::Tree);
    pub const EXACT_F32: Self = Self::f32_fold(FoldTier::Exact);
    pub const RELAXED_F32: Self = Self::f32_fold(FoldTier::Relaxed);

    /// The binary32 policy whose float folds follow `fold`; strict is v1.
    pub const fn f32_fold(fold: FoldTier) -> Self {
        match fold {
            FoldTier::Strict => Self::STRICT_F32,
            _ => Self {
                version: 2,
                binary64: false,
                tc: false,
                fold,
            },
        }
    }

    /// Validate the proposed policy fields before use. No other policy is
    /// approximated by strict arithmetic.
    pub fn from_parts(version: u32, kind: &str) -> Result<Self, PolicyError> {
        // A serialized field never selects `tc-f16-f32`: only a recognized
        // `tc_mul_add` source contraction constructs `TC_F16_F32`.
        match (version, kind) {
            (1, "strict-f32") | (2, "f32-strict") => Ok(Self::STRICT_F32),
            (1, "strict-f64") | (2, "f64-strict") => Ok(Self::STRICT_F64),
            (2, _) => kind
                .strip_prefix("f32-")
                .and_then(FoldTier::from_name)
                .filter(|&fold| fold != FoldTier::Strict)
                .map(Self::f32_fold)
                .ok_or(PolicyError::UnsupportedKind),
            (1, _) => Err(PolicyError::UnsupportedKind),
            _ => Err(PolicyError::UnsupportedVersion(version)),
        }
    }

    pub const fn version(self) -> u32 {
        self.version
    }

    /// How float folds under this policy may associate.
    pub const fn fold(self) -> FoldTier {
        self.fold
    }

    pub const fn kind(self) -> &'static str {
        match self.fold {
            FoldTier::Tree => "f32-tree",
            FoldTier::Exact => "f32-exact",
            FoldTier::Relaxed => "f32-relaxed",
            FoldTier::Strict if self.tc => "tc-f16-f32",
            FoldTier::Strict if self.binary64 => "strict-f64",
            FoldTier::Strict => "strict-f32",
        }
    }

    pub const fn id(self) -> &'static str {
        match self.fold {
            FoldTier::Tree => "f32-tree-v2",
            FoldTier::Exact => "f32-exact-v2",
            FoldTier::Relaxed => "f32-relaxed-v2",
            FoldTier::Strict if self.tc => "tc-f16-f32-v1",
            FoldTier::Strict if self.binary64 => "strict-f64-v1",
            FoldTier::Strict => "strict-f32-v1",
        }
    }

    /// Canonical fields for the lane's proposed schema, not device IR wiring.
    pub fn to_json(self) -> serde_json::Value {
        serde_json::json!({"version": self.version(), "kind": self.kind()})
    }
}

/// Local conversion outcome; integration must attach the original source span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericTrap {
    IntegerOverflow,
}

impl NumericTrap {
    pub const fn category(self) -> &'static str {
        match self {
            Self::IntegerOverflow => "INTEGER_OVERFLOW",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Comparison {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Binary32 oracle encoding. Bit access is test infrastructure, not a public
/// Invar bit-cast intrinsic. Derived equality compares encodings; `strict_eq`
/// compares engine outcomes and `compare` implements source predicates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F32Bits(u32);

/// Bounded source decimal conversion errors, before any floating operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecimalError {
    Malformed,
    DigitLimit,
}

pub const MAX_DECIMAL_DIGITS: usize = 256;

/// Split a source decimal token into its digits and its optional `f32` or
/// `f64` suffix. An unsuffixed literal takes its float type from context.
pub fn decimal_digits(raw: &str) -> (&str, Option<&'static str>) {
    for suffix in ["f32", "f64"] {
        if let Some(digits) = raw.strip_suffix(suffix) {
            return (digits, Some(suffix));
        }
    }
    (raw, None)
}

const SIGN: u32 = 0x8000_0000;
const F32: ieee::Format = ieee::BINARY32;

impl F32Bits {
    pub const POS_ZERO: Self = Self(0);
    pub const NEG_ZERO: Self = Self(SIGN);
    pub const INFINITY: Self = Self(0x7f80_0000);
    pub const NEG_INFINITY: Self = Self(SIGN | 0x7f80_0000);
    /// A representative of the unobservable NaN payload/sign class.
    pub const NAN: Self = Self(0x7fc0_0000);

    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    const fn wide(self) -> u64 {
        self.0 as u64
    }

    // Every ieee result for binary32 fits the low 32 bits.
    const fn narrow(bits: u64) -> Self {
        Self(bits as u32)
    }

    /// Round a positive `digits.digits` spelling once, exactly, to binary32.
    /// Unary source negation is separate, including for signed zero. Neither
    /// decimal conversion nor rounding consults the host floating environment.
    /// An `f32` suffix is accepted; any other suffix is malformed.
    pub fn from_decimal(raw: &str) -> Result<Self, DecimalError> {
        match decimal_digits(raw) {
            (digits, None | Some("f32")) => F32.parse_decimal(digits).map(Self::narrow),
            _ => Err(DecimalError::Malformed),
        }
    }

    pub const fn to_bits(self) -> u32 {
        self.0
    }

    pub const fn is_nan(self) -> bool {
        F32.is_nan(self.wide())
    }

    pub const fn is_infinite(self) -> bool {
        F32.is_infinite(self.wide())
    }

    pub const fn is_zero(self) -> bool {
        F32.is_zero(self.wide())
    }

    pub const fn is_negative(self) -> bool {
        self.0 & SIGN != 0
    }

    /// Strict engine agreement: identical non-NaN encodings, all NaNs equal.
    pub const fn strict_eq(self, other: Self) -> bool {
        self.0 == other.0 || (self.is_nan() && other.is_nan())
    }

    /// IEEE predicates: signed zeros equal; NaNs unordered (including with
    /// themselves). This is deliberately distinct from `strict_eq`.
    pub fn compare(self, predicate: Comparison, other: Self) -> bool {
        F32.compare(self.wide(), predicate, other.wide())
    }

    /// Negation preserves zeros and finite magnitude without rounding.
    pub const fn negate(self) -> Self {
        Self(self.0 ^ SIGN)
    }

    /// Addition rounds once; exact cancellation is +0 under nearest-even.
    #[allow(
        clippy::should_implement_trait,
        reason = "Named oracle calls make each strict rounding site explicit; no overloaded host operators"
    )]
    pub fn add(self, other: Self) -> Self {
        Self::narrow(F32.add(self.wide(), other.wide()))
    }

    #[allow(
        clippy::should_implement_trait,
        reason = "Named oracle calls make each strict rounding site explicit; no overloaded host operators"
    )]
    pub fn sub(self, other: Self) -> Self {
        self.add(other.negate())
    }

    /// Multiplication rounds independently of any later addition.
    #[allow(
        clippy::should_implement_trait,
        reason = "Named oracle calls make each strict rounding site explicit; no overloaded host operators"
    )]
    pub fn mul(self, other: Self) -> Self {
        Self::narrow(F32.mul(self.wide(), other.wide()))
    }

    /// Correctly rounded CPU division. Floating zero divisors follow IEEE
    /// outcomes and do not use the source integer division trap.
    #[allow(
        clippy::should_implement_trait,
        reason = "Named oracle calls make each strict rounding site explicit; no overloaded host operators"
    )]
    pub fn div(self, other: Self) -> Self {
        Self::narrow(F32.div(self.wide(), other.wide()))
    }

    /// Explicit fused multiply-add: compute exact a*b+c and round only once.
    /// No call to `mul` or `add` is allowed on its finite path.
    pub fn fma(self, multiplier: Self, addend: Self) -> Self {
        Self::narrow(F32.fma(self.wide(), multiplier.wide(), addend.wide()))
    }

    /// Correctly rounded CPU sqrt, with -0 preserved and negative inputs NaN.
    pub fn sqrt(self) -> Self {
        Self::narrow(F32.sqrt(self.wide()))
    }

    pub fn from_i32(value: i32) -> Self {
        Self::narrow(F32.encode_integer(i64::from(value)))
    }

    pub fn from_u32(value: u32) -> Self {
        Self::narrow(F32.encode_integer(i64::from(value)))
    }

    /// Validate representability before truncating toward zero. Never use a
    /// saturating host float cast to implement a source overflow trap.
    pub fn trunc_i32(self) -> Result<i32, NumericTrap> {
        F32.trunc_integer(self.wide(), i64::from(i32::MIN), i64::from(i32::MAX))
            .map(|value| value as i32)
    }

    /// Truncate toward zero into `u32`; NaN, infinities and values whose
    /// truncation is negative or above `u32::MAX` trap.
    pub fn trunc_u32(self) -> Result<u32, NumericTrap> {
        F32.trunc_integer(self.wide(), 0, i64::from(u32::MAX))
            .map(|value| value as u32)
    }

    /// A strict source-order left fold; empty input returns the initial bits.
    pub fn ordered_sum(initial: Self, inputs: &[Self]) -> Self {
        inputs.iter().fold(initial, |acc, &value| acc.add(value))
    }
}

/// One round-to-nearest-even of an exact finite rational; `negative_zero`
/// selects the sign of an exactly zero value. Used by exact reductions.
pub fn round_rational(value: num_rational::BigRational, negative_zero: bool) -> F32Bits {
    F32Bits::narrow(F32.round(&value, negative_zero))
}
