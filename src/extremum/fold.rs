//! The selection fold of `min_of`, `max_of`, `argmin` and `argmax` (#207),
//! free of crate dependencies so the native host can include it verbatim.
//!
//! Every non-NaN element has a key that orders it as IEEE 754 totalOrder
//! does (exact for the integers; `-0 < +0` for floats), which on non-NaN
//! operands is the order of minimumNumber and maximumNumber. The fold keeps
//! the first element of least (or greatest) key and skips NaN, so it equals
//! the left fold of the library `min` (or `max`), and it is associative:
//! chunk results combine by the same step in chunk order.

/// The four intrinsics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    MinOf,
    MaxOf,
    Argmin,
    Argmax,
}

impl Kind {
    pub const ALL: [Self; 4] = [Self::MinOf, Self::MaxOf, Self::Argmin, Self::Argmax];

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::MinOf => "min_of",
            Self::MaxOf => "max_of",
            Self::Argmin => "argmin",
            Self::Argmax => "argmax",
        }
    }

    /// Stable code of the native host ABI.
    pub fn code(self) -> u32 {
        match self {
            Self::MinOf => 1,
            Self::MaxOf => 2,
            Self::Argmin => 3,
            Self::Argmax => 4,
        }
    }

    pub fn from_code(code: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.code() == code)
    }

    /// Whether the result is the index (`u32`) rather than the element.
    pub fn is_index(self) -> bool {
        matches!(self, Self::Argmin | Self::Argmax)
    }

    pub fn is_max(self) -> bool {
        matches!(self, Self::MaxOf | Self::Argmax)
    }
}

/// An element type the intrinsics admit.
pub trait Elem: Copy + Send + Sync {
    /// The totalOrder key, injective on encodings; `None` for NaN.
    fn key(self) -> Option<i128>;
    /// The canonical encoding of a result (the quiet NaN for any NaN).
    fn canonical(self) -> Self {
        self
    }
}

impl Elem for i32 {
    fn key(self) -> Option<i128> {
        Some(i128::from(self))
    }
}

impl Elem for u32 {
    fn key(self) -> Option<i128> {
        Some(i128::from(self))
    }
}

/// A binary32 encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F32Word(pub u32);

/// A binary64 encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F64Word(pub u64);

impl Elem for F32Word {
    fn key(self) -> Option<i128> {
        let bits = self.0;
        if bits & 0x7fff_ffff > 0x7f80_0000 {
            return None;
        }
        let ordered = if bits >> 31 == 1 {
            !bits
        } else {
            bits | 0x8000_0000
        };
        Some(i128::from(ordered))
    }
    fn canonical(self) -> Self {
        if self.key().is_none() {
            Self(0x7fc0_0000)
        } else {
            self
        }
    }
}

impl Elem for F64Word {
    fn key(self) -> Option<i128> {
        let bits = self.0;
        if bits & 0x7fff_ffff_ffff_ffff > 0x7ff0_0000_0000_0000 {
            return None;
        }
        let ordered = if bits >> 63 == 1 {
            !bits
        } else {
            bits | 0x8000_0000_0000_0000
        };
        Some(i128::from(ordered))
    }
    fn canonical(self) -> Self {
        if self.key().is_none() {
            Self(0x7ff8_0000_0000_0000)
        } else {
            self
        }
    }
}

/// One fold step: `next` (an element, or a later chunk's result) replaces
/// `best` iff it is not NaN and `best` is NaN or has a greater key (a lesser
/// one for `max`), so the first of equal elements stays. `right_ties` is the
/// injected mutation that lets the later operand win a tie.
pub fn step<T: Elem>(
    max: bool,
    best: (usize, T),
    next: (usize, T),
    right_ties: bool,
) -> (usize, T) {
    let Some(key) = next.1.key() else {
        return best;
    };
    match best.1.key() {
        None => next,
        Some(held) if (max && key > held) || (!max && key < held) => next,
        Some(held) if right_ties && key == held => next,
        Some(_) => best,
    }
}

/// Index and raw value of the extremum of `values`, whose first element has
/// index `start`; `None` for an empty slice. An all-NaN slice gives its first
/// element.
pub fn extremum<T: Elem>(max: bool, values: &[T], start: usize) -> Option<(usize, T)> {
    let (&first, rest) = values.split_first()?;
    Some(
        rest.iter()
            .enumerate()
            .fold((start, first), |best, (k, &x)| {
                step(max, best, (start + 1 + k, x), false)
            }),
    )
}

/// Ordered combine of the chunks' extrema (empty chunks are `None`).
pub fn combine<T: Elem>(
    max: bool,
    parts: &[Option<(usize, T)>],
    right_ties: bool,
) -> Option<(usize, T)> {
    parts
        .iter()
        .flatten()
        .copied()
        .reduce(|best, next| step(max, best, next, right_ties))
}

/// The intrinsic's result: the index, or the canonical element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome<T> {
    Index(u32),
    Value(T),
}

/// The result of `kind` from the extremum of the whole input, or `None`
/// (the empty-input trap). An index past `u32` cannot come from a live
/// array; it is reported as `None` rather than truncated.
pub fn outcome<T: Elem>(kind: Kind, found: Option<(usize, T)>) -> Option<Outcome<T>> {
    let (index, value) = found?;
    Some(if kind.is_index() {
        Outcome::Index(u32::try_from(index).ok()?)
    } else {
        Outcome::Value(value.canonical())
    })
}

/// The serial reference of `kind` over `values`.
pub fn reference<T: Elem>(kind: Kind, values: &[T]) -> Option<Outcome<T>> {
    reference_iter(kind, values.iter().copied())
}

/// [`reference`] over an iterator, so a caller that adapts element words
/// (the native host) scans them in place instead of copying the array.
pub fn reference_iter<T: Elem>(
    kind: Kind,
    values: impl IntoIterator<Item = T>,
) -> Option<Outcome<T>> {
    let max = kind.is_max();
    let found = values
        .into_iter()
        .enumerate()
        .reduce(|best, next| step(max, best, next, false));
    outcome(kind, found)
}
